"""Check that every pipeline artifact is complete and consistent (e.g. after a crash or power/
network loss). Read-only; prints OK / PROBLEM lines and exits 1 if anything needs fixing.

Checks: images vs metadata, splits and labels for every photo, no site in two splits,
no near-identical photo (DINOv2 cosine >= 0.95) across train/val and test, leftover .tmp/.lock
files, feature caches for each backbone (photos + phone-style copies), VLM coverage of the test
split, models trained on the current split/labels (content fingerprint) and on no test photo,
results newer than the models they report on, and that models/selected.json points to existing
models of the right backbone.

Usage:  python tools/check_integrity.py
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (DATA, IMAGES, LABELS_DIR, METADATA_CSV, MODELS, RESULTS, SPLITS_CSV, VLM_CSV,  # noqa: E402
                    data_fingerprint, labeled_dataset, read_csv)

problems = []


def ok(msg):
    print(f"OK       {msg}")


def bad(msg):
    print(f"PROBLEM  {msg}")
    problems.append(msg)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            epilog='See the module docstring for details.').parse_args()
    meta = read_csv(METADATA_CSV)
    files = {p.name for p in IMAGES.glob("*.jpg")}
    rows = set(meta.file.str.split("/").str[-1])
    (ok if not rows - files else bad)(f"{len(meta)} metadata rows, {len(rows - files)} without an image file")
    orphans = files - rows
    print(f"note     {len(orphans)} image files without a metadata row (interrupted downloads; harmless, ignored)")
    (ok if meta.image_id.is_unique else bad)("image_id unique")

    splits = read_csv(SPLITS_CSV)
    missing = set(meta.image_id) - set(splits.image_id)
    (ok if not missing else bad)(f"split assigned for every photo ({len(missing)} missing)")
    site_split = splits[["image_id", "split"]].merge(meta[["image_id", "site_id"]], on="image_id").groupby("site_id").split.nunique()
    (ok if (site_split == 1).all() else bad)(f"no site spans two splits ({(site_split > 1).sum()} do)")

    labels = [read_csv(p) for p in LABELS_DIR.glob("*.csv")]
    labeled = set().union(*[set(l.image_id) for l in labels]) if labels else set()
    unl = set(meta.image_id) - labeled
    (ok if not unl else bad)(f"every photo labeled ({len(unl)} unlabeled)")

    leftovers = [p for p in DATA.rglob("*") if p.suffix in (".tmp", ".lock")]
    (ok if not leftovers else bad)(f"no half-written .tmp / stale .lock files ({[str(p) for p in leftovers][:5]})")

    df = labeled_dataset(include_reject=True)
    fit_ids = set(df.image_id[df.split.isin(["train", "val"])])
    test_ids = set(df.image_id[df.split == "test"])
    for bb in ("facebook_dinov2-base", "facebook_dinov2-large", "google_siglip2-base-patch16-224"):
        path = DATA / "cache" / f"emb_{bb}.npz"
        if not path.exists():
            bad(f"feature cache missing: {path.name}")
            continue
        keys = set(np.load(path).files)
        m0 = len(set(df.image_id) - keys)
        maug = sum(len({f"{i}#{k}" for i in fit_ids} - keys) for k in (1, 2, 3))
        mtest = len({f"{i}#1" for i in test_ids} - keys)
        (ok if m0 == maug == mtest == 0 else bad)(
            f"{bb}: missing photos {m0}, phone copies (train/val) {maug}, phone copies (test) {mtest}")

    vlm = read_csv(VLM_CSV)
    usable_test = set(df.image_id[(df.split == "test") & (df.label != "reject")])
    miss = usable_test - set(vlm.image_id) if vlm is not None else usable_test
    (ok if not miss else bad)(f"VLM baseline covers usable test photos ({len(miss)} missing)")

    # near-identical photos across splits (site grouping can miss renamed places)
    cache_b = DATA / "cache" / "emb_facebook_dinov2-base.npz"
    if cache_b.exists():
        c = np.load(cache_b)
        u = df[df.label != "reject"]
        X = np.stack([c[i] for i in u.image_id]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        te, tv = (u.split == "test").values, (u.split != "test").values
        near = int(((X[te] @ X[tv].T).max(1) >= 0.95).sum()) if te.any() and tv.any() else 0
        (ok if near == 0 else bad)(f"usable test photos with a near-identical train/val photo (cos>=0.95): {near}")

    # Staleness is judged by CONTENT: every model stores a fingerprint of the split/label table it
    # was trained on (common.data_fingerprint), so touching a file cannot cause or hide staleness.
    fp = data_fingerprint()
    test_all = set(df.image_id[df.split == "test"])
    model_time = {}
    for p in sorted(MODELS.glob("probe_*.joblib")):
        b = joblib.load(p)
        model_time[p.stem] = p.stat().st_mtime
        (ok if b.get("data_fingerprint") == fp else bad)(
            f"{p.name} trained on the current split/labels (fingerprint {b.get('data_fingerprint')} vs {fp})")
        overlap = set(b.get("train_ids", [])) & test_all
        (ok if not overlap else bad)(f"{p.name} trained on no test photo ({len(overlap)} overlap)")
    cnn = sorted(p for p in MODELS.glob("cnn_*.pt") if not p.name.endswith((".best.pt", ".state.pt")))
    if cnn:
        import torch
        ck = torch.load(cnn[-1], map_location="cpu")
        model_time["cnn"] = cnn[-1].stat().st_mtime
        overlap = set(ck.get("train_ids", [])) & test_all
        (ok if not overlap else bad)(f"{cnn[-1].name} trained on no test photo ({len(overlap)} overlap)")
    sel = MODELS / "selected.json"
    if sel.exists():
        s = json.loads(sel.read_text())
        members = s["members"]
        exist = all((MODELS / f"{m}.joblib").exists() or m == "cnn" for m in members)
        (ok if exist else bad)(f"selected.json members exist: {members}")
        for m, bb in s.get("backbones", {}).items():
            if (MODELS / f"{m}.joblib").exists():
                got = joblib.load(MODELS / f"{m}.joblib")["backbone"]
                (ok if got == bb else bad)(f"{m} backbone {got} matches selection ({bb})")
        (ok if "confident_min" in s else bad)("selected.json has a calibrated confident_min")
        (ok if s.get("data_fingerprint") == fp else bad)("selected.json chosen on the current split/labels")
        model_time["members"] = max(model_time.get(m, 0) for m in members)   # what robustness.md evaluates
        model_time["selected"] = max(sel.stat().st_mtime, model_time["members"])
    else:
        bad("models/selected.json missing (run tools/robustness_eval.py)")
    # each result must be newer than the model(s) it reports on
    for name, depends in (("pred_probe_base.csv", "probe_base"), ("pred_probe_large.csv", "probe_large"),
                          ("pred_probe_siglip.csv", "probe_siglip"), ("pred_cnn.csv", "cnn"),
                          ("pred_selected.csv", "selected"), ("robustness.md", "members"),
                          ("system_test.md", "selected"), ("summary.md", "selected")):
        p = RESULTS / name
        if not p.exists():
            bad(f"results/{name} missing")
        elif depends in model_time:
            (ok if p.stat().st_mtime >= model_time[depends] - 1 else bad)(f"results/{name} newer than {depends}")
    print(f"\n{'ALL OK' if not problems else f'{len(problems)} PROBLEM(S)'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
