"""End-to-end backtest of the deployed pipeline (exactly what the app runs) on the test split.

Runs src/predict.Predictor (models from models/selected.json) on every test photo, including
those that should be rejected, once on the original file and once on its fixed phone-degraded
copy (the same copy probe.py/train_cnn.py score as *_phonesim: probe.variant(.., k=1)).
Refuses to run if any deployed model was trained on a test photo.

Reports, over ALL test photos and over usable photos only:
  - photo check: does it accept usable ground photos and reject unusable ones?
  - outcome mix: answered confidently / uncertain (incl. family answers) / asked for a new photo
  - end-to-end precision: share of confident answers that are on a usable photo AND correct
  - type accuracy on confident answers for usable photos
Ground truth = final labels (AI-made unless people re-labeled; see report.md).

Usage:  python tools/system_test.py [--model auto]
Output: results/system_test.md, results/system_test.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402,F401
from common import DATA, REJECT, RESULTS, labeled_dataset  # noqa: E402
from predict import FAMILIES, Predictor, load_image  # noqa: E402
from probe import variant  # noqa: E402


def phone_copy(image_id, file):
    img = variant(image_id, file, 1)       # same seeded copy as the *_phonesim evaluations
    img.thumbnail((1024, 1024))            # what load_image does to an upload
    return img


def run(pred, df, phone, batch=16):
    rows = []
    for i in range(0, len(df), batch):
        chunk = df.iloc[i:i + batch]
        imgs = [phone_copy(r.image_id, r.file) if phone else load_image(DATA / r.file) for r in chunk.itertuples()]
        for (_, r), res in zip(chunk.iterrows(), pred(imgs)):
            rows.append({"image_id": r.image_id, "label": r.label, "status": res["status"], "pred": res["label"],
                         "family": res["family"] or "", "conf": round(res["confidence"], 4),
                         "usable_p": round(res["usable"], 4), "phone": int(phone)})
    return pd.DataFrame(rows)


def summarize(d):
    usable = d.label != REJECT
    accepted = d.status != "bad_photo"
    ok = d.status == "ok"
    out = {"n": len(d), "n_usable": int(usable.sum()),
           "photo_check_acc": (usable == accepted).mean(),
           "usable_accepted": accepted[usable].mean(), "unusable_rejected": (~accepted[~usable]).mean(),
           # over ALL photos
           "confident_answers": int(ok.sum()),
           "confident_on_unusable": (ok & ~usable).sum() / max(1, ok.sum()),
           "e2e_precision": (ok & usable & (d.pred == d.label)).sum() / max(1, ok.sum()),
           # usable photos only
           "answered_share": (ok & usable).sum() / max(1, usable.sum()),
           "uncertain_share": ((d.status == "uncertain") & usable).sum() / max(1, usable.sum()),
           "acc_answered": (d.pred[ok & usable] == d.label[ok & usable]).mean() if (ok & usable).any() else np.nan,
           "acc_all_accepted_top1": (d.pred[accepted & usable] == d.label[accepted & usable]).mean()}
    fam = d[(d.status == "uncertain") & (d.family != "") & usable]
    out["family_answers"] = len(fam)
    out["family_correct"] = np.mean([lab in FAMILIES[f] for lab, f in zip(fam.label, fam.family)]) if len(fam) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto")
    args = ap.parse_args()
    df = labeled_dataset(include_reject=True)
    df = df[df.split == "test"].reset_index(drop=True)
    pred = Predictor(args.model)
    leak = pred.train_ids & set(df.image_id)
    if leak:
        raise SystemExit(f"{len(leak)} test photos were used to train the deployed models; retrain first")
    res = pd.concat([run(pred, df, False), run(pred, df, True)], ignore_index=True)
    res.to_csv(RESULTS / "system_test.csv", index=False)
    s = {name: summarize(res[res.phone == k]) for name, k in (("original", 0), ("phone-degraded", 1))}
    lines = [f"# End-to-end system test (test split, model: {pred.model_name}, confident ≥ {pred.confident_min})\n",
             f"{s['original']['n']} test photos, {s['original']['n_usable']} usable ground photos of slopes. "
             "Training sets of the deployed models are disjoint from these photos (checked).\n",
             "| | original | phone-degraded |", "|---|---|---|"]
    names = [("photo_check_acc", "Photo check accuracy (usable vs reject)"),
             ("usable_accepted", "Usable photos accepted"), ("unusable_rejected", "Unusable photos rejected"),
             ("confident_answers", "**All photos:** confident answers given"),
             ("confident_on_unusable", "**All photos:** share of confident answers on photos that should have been rejected"),
             ("e2e_precision", "**All photos: end-to-end precision** (confident answer on a usable photo AND correct type)"),
             ("answered_share", "Usable photos answered confidently"),
             ("uncertain_share", "Usable photos marked uncertain"),
             ("acc_answered", "**Usable photos: type accuracy when answered**"),
             ("acc_all_accepted_top1", "Usable photos: top-guess accuracy on all accepted"),
             ("family_answers", "Uncertain answers given as a family (flow/slide)"),
             ("family_correct", "Family answers correct")]
    for k, label in names:
        a, b = s["original"][k], s["phone-degraded"][k]
        fmt = (lambda v: f"{v}") if k in ("family_answers", "confident_answers") else (lambda v: f"{v:.3f}")
        lines.append(f"| {label} | {fmt(a)} | {fmt(b)} |")
    lines.append("\nThe test split keeps the collection's mix (about 3 rejects per usable photo); in the field, "
                 "where most uploads are slope photos, end-to-end precision will be closer to the usable-photo figure.")
    (RESULTS / "system_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
