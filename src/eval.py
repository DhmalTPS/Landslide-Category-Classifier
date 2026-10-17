"""Score every model on the same (untouched) test split and write a comparison report.

Models found automatically:
  - results/pred_<name>.csv             (probe_base, probe_large, probe_siglip, cnn, ...)
  - results/pred_selected.csv           written here from models/selected.json: the average of
                                        the members' class probabilities (= what the app runs)
  - data/vlm_predictions.csv            one entry per VLM model; the VLM's "reject" option is
                                        excluded and the remaining class probabilities used
Safety checks: prediction files older than data/splits.csv or the labels are skipped (stale),
and a model whose stored training ids overlap the test split is refused (leak).
Macro-F1 is averaged over the classes present in the scored set (also per subgroup); 95% CIs
come from a site-clustered bootstrap. Also reports inter-annotator agreement (Cohen's kappa).

Usage:  python src/eval.py
Output: results/summary.md, results/confusion_<model>.png
"""
import argparse
import itertools
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score

from common import (CLASS_NAMES, LABELS_DIR, MODELS, RESULTS, SPLITS_CSV, VLM_CSV, all_labels, is_south_asia,
                    labeled_dataset, read_csv, write_csv)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P_COLS = [f"p_{c}" for c in CLASS_NAMES]


def write_selected():
    """results/pred_selected[_phonesim].csv = mean probabilities of the selected members."""
    sel_path = MODELS / "selected.json"
    if not sel_path.exists():
        return
    members = json.loads(sel_path.read_text(encoding="utf-8"))["members"]
    newest_input = max([SPLITS_CSV.stat().st_mtime] + [p.stat().st_mtime for p in LABELS_DIR.glob("*.csv")])
    for suffix in ("", "_phonesim"):
        paths = [RESULTS / f"pred_{m}{suffix}.csv" for m in members]
        frames = [read_csv(p) for p in paths]
        if any(f is None or not set(P_COLS) <= set(f.columns) for f in frames):
            print(f"note: cannot build pred_selected{suffix}.csv (missing member predictions)")
            continue
        if any(p.stat().st_mtime < newest_input for p in paths) or len({frozenset(f.image_id) for f in frames}) > 1:
            print(f"note: pred_selected{suffix}.csv not built (member predictions stale or for different photos)")
            (RESULTS / f"pred_selected{suffix}.csv").unlink(missing_ok=True)
            continue
        ids = frames[0].image_id.values
        P = np.mean([f.set_index("image_id").loc[ids, P_COLS].astype(float).values for f in frames], axis=0)
        write_csv(pd.DataFrame({"image_id": ids, "pred": np.array(CLASS_NAMES)[P.argmax(1)], "conf": P.max(1).round(4),
                                **{c: P[:, i].round(4) for i, c in enumerate(P_COLS)}}),
                  RESULTS / f"pred_selected{suffix}.csv")


def trained_on(name):
    """Training image ids stored with a model (empty set if unknown)."""
    base = name.removesuffix("_phonesim")
    if (MODELS / f"{base}.joblib").exists():
        return set(joblib.load(MODELS / f"{base}.joblib").get("train_ids", []))
    if base == "selected" and (MODELS / "selected.json").exists():
        members = json.loads((MODELS / "selected.json").read_text(encoding="utf-8"))["members"]
        return set().union(*[trained_on(m) for m in members])
    if base == "cnn":
        import torch
        ck = sorted((p for p in MODELS.glob("cnn_*.pt") if not p.name.endswith((".best.pt", ".state.pt"))),
                    key=lambda p: p.stat().st_mtime)   # same choice as predict.py
        if ck:
            return set(torch.load(ck[-1], map_location="cpu").get("train_ids", []))
    return set()


def collect_predictions(test_ids):
    newest_input = max([SPLITS_CSV.stat().st_mtime] + [p.stat().st_mtime for p in LABELS_DIR.glob("*.csv")])
    preds, notes = {}, []
    for p in sorted(RESULTS.glob("pred_*.csv")):
        name = p.stem.removeprefix("pred_")
        if p.stat().st_mtime < newest_input:
            notes.append(f"skipped `{name}`: predictions older than the current split/labels (rerun that model)")
            continue
        leak = trained_on(name) & test_ids
        if leak:
            notes.append(f"refused `{name}`: trained on {len(leak)} test photos")
            continue
        preds[name] = read_csv(p)[["image_id", "pred"]]
    vlm = read_csv(VLM_CSV)
    if vlm is not None:
        for model, g in vlm.groupby("model"):
            g = g.drop_duplicates("image_id", keep="last")
            g = g[(g[P_COLS] != "").all(axis=1)]
            probs = g[P_COLS].astype(float).values
            preds["vlm_" + model.split("/")[-1]] = pd.DataFrame(
                {"image_id": g.image_id.values, "pred": [CLASS_NAMES[i] for i in probs.argmax(1)]})
    return preds, notes


def macro(y, p):
    return f1_score(y, p, average="macro", labels=sorted(set(y)), zero_division=0)


def site_bootstrap_ci(y, p, sites, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    idx_of = {s: np.where(sites == s)[0] for s in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_of[s] for s in rng.choice(uniq, len(uniq))])
        vals.append(macro(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def plot_confusion(y, p, name):
    cm = confusion_matrix(y, p, labels=CLASS_NAMES)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i, j in itertools.product(range(len(CLASS_NAMES)), repeat=2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name} (row-normalized colour, counts shown)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def annotator_agreement():
    lab = all_labels()
    lines = []
    for a, b in itertools.combinations(sorted(lab.reviewer.unique()), 2):
        m = lab[lab.reviewer == a].merge(lab[lab.reviewer == b], on="image_id")
        if len(m) >= 10:
            k = cohen_kappa_score(m.label_x, m.label_y)
            lines.append(f"| {a} vs {b} | {len(m)} | {(m.label_x == m.label_y).mean():.3f} | {k:.3f} |")
    if not lines:
        return ""
    return ("\n## Label agreement between reviewers\n\n| Pair | Shared images | Raw agreement | Cohen's kappa |\n"
            "|---|---|---|---|\n" + "\n".join(lines) + "\n")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            epilog='See the module docstring for details.').parse_args()
    test = labeled_dataset()
    test = test[test.split == "test"].reset_index(drop=True)
    if test.empty:
        raise SystemExit("no labeled test images")
    RESULTS.mkdir(exist_ok=True)
    write_selected()
    preds, notes = collect_predictions(set(labeled_dataset(include_reject=True).query("split == 'test'").image_id))

    rows, per_class = [], {}
    for name, pred in preds.items():
        m = test[["image_id", "label", "site_id"]].merge(pred, on="image_id")
        if m.empty:
            continue
        y, p = m.label.values, m.pred.values
        lo, hi = site_bootstrap_ci(y, p, m.site_id.values)
        rows.append(f"| {name} | {len(m)}/{len(test)} | {accuracy_score(y, p):.3f} | {macro(y, p):.3f} [{lo:.3f}, {hi:.3f}] |")
        per_class[name] = recall_score(y, p, labels=CLASS_NAMES, average=None, zero_division=0)
        plot_confusion(y, p, name)
    if not rows:
        raise SystemExit(f"{len(test)} labeled test images, but no current model predictions for them. "
                         "Run probe.py / train_cnn.py / vlm_label.py --split test first.")

    def subset(sub, title, note):
        out = [f"\n## {title}\n", f"{note} n = {len(sub)}; classes: {sub.label.value_counts().to_dict()}. "
               "Macro-F1 over the classes present in this subset.\n",
               "| Model | n | Accuracy | Macro-F1 |", "|---|---|---|---|"]
        for name, pred in preds.items():
            m = sub[["image_id", "label"]].merge(pred, on="image_id")
            if len(m):
                out.append(f"| {name} | {len(m)} | {accuracy_score(m.label, m.pred):.3f} | {macro(m.label.values, m.pred.values):.3f} |")
        return out

    ai_only = test.reviewers.str.split("+").map(lambda rs: all(r.endswith("_ai") for r in rs))
    counts = test.label.value_counts().reindex(CLASS_NAMES, fill_value=0)
    source = (f"**Label source:** {int(ai_only.sum())} of {len(test)} test labels were made only by an AI annotator "
              "(reviewer `*_ai`); these numbers measure agreement with those labels, not verified ground truth. "
              "Have a person re-label the test split blind "
              "(`python src/review.py --reviewer <name> --split test --relabel-others`) to replace them.\n"
              if ai_only.any() else "**Label source:** all test labels made by people.\n")
    md = [f"# Test results\n\nTest split: {len(test)} usable photos from {test.site_id.nunique()} sites "
          "(no site also appears in train/val). "
          f"Labels made with VLM suggestion visible: {(test.vlm_shown == '1').sum()} (should be 0).\n", source,
          *([("Notes: " + "; ".join(notes) + ".\n")] if notes else []),
          "| Model | Scored | Accuracy | Macro-F1 [95% CI, site bootstrap] |", "|---|---|---|---|", *rows,
          *subset(test[test.viewpoint == "up"], "Deployment case: ground photos looking up/across at the slope", ""),
          *subset(test[test.apply(is_south_asia, axis=1)], "South Asia (deployment region)",
                  "Tagged from category/title/description place names (common.is_south_asia)."),
          *subset(test[test.get("is_phone", pd.Series("", index=test.index)) == "1"],
                  "Real phone photos (EXIF camera = phone)", "Small n -> wide uncertainty."),
          "\n## Recall per class\n", "| Class | n | " + " | ".join(per_class) + " |",
          "|---|---|" + "---|" * len(per_class)]
    for i, c in enumerate(CLASS_NAMES):
        md.append(f"| {c} | {counts[c]} | " + " | ".join(f"{r[i]:.2f}" for r in per_class.values()) + " |")
    md.append(annotator_agreement())
    md.append("\n`<model>_phonesim` = the same model scored on fixed phone-degraded copies of the test photos "
              "(crop, tilt, blur, noise, JPEG; src/phone_aug.py). `selected` = the app's model (models/selected.json).\n")
    (RESULTS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nconfusion matrices in {RESULTS}")


if __name__ == "__main__":
    main()
