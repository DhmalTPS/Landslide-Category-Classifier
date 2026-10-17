"""Flag likely label errors in train/val (never test) with cross-validated predictions.

5-fold cross-validation grouped by site on cached DINOv2 features; an image is flagged when
the model gives its current label < --max-own probability while another class gets > --min-other.
Flagged images are written as an id list for tools/contact_sheet.py, so a person (or the AI
annotator) can look at them and fix real mistakes with review.py / tools/ai_labels.py.
Test images are excluded on purpose: correcting test labels toward the model's opinion would
inflate the test score.

Usage:  python tools/find_label_issues.py OUT_DIR [--backbone facebook/dinov2-base]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import DATA, labeled_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--max-own", type=float, default=0.15)
    ap.add_argument("--min-other", type=float, default=0.6)
    args = ap.parse_args()

    df = labeled_dataset(include_reject=True)
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    cache = np.load(DATA / "cache" / f"emb_{args.backbone.replace('/', '_')}.npz")
    missing = [i for i in df.image_id if i not in cache.files]
    if missing:
        raise SystemExit(f"{len(missing)} photos have no cached features; run tools/precompute_embeddings.py first")
    X = np.stack([cache[i] for i in df.image_id])
    y = df.label.values
    P = np.zeros((len(df), len(np.unique(y))))
    classes = np.unique(y)
    for tr, te in GroupKFold(5).split(X, y, df.site_id):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        P[np.ix_(te, [list(classes).index(c) for c in m.classes_])] = m.predict_proba(X[te])
    own = P[np.arange(len(df)), [list(classes).index(c) for c in y]]
    other = np.where(np.arange(P.shape[1])[None, :] == np.array([list(classes).index(c) for c in y])[:, None], 0, P)
    flag = (own < args.max_own) & (other.max(1) > args.min_other)
    out = df.loc[flag, ["image_id", "label", "split", "title"]].assign(
        model_says=classes[other.argmax(1)][flag], p_model=other.max(1)[flag].round(2), p_label=own[flag].round(2))
    out = out.sort_values("p_label")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out) / "flagged.csv", index=False)
    (Path(args.out) / "ids.txt").write_text("\n".join(out.image_id), encoding="utf-8")
    print(f"{flag.sum()} of {len(df)} train/val labels flagged")
    print(out.groupby(["label", "model_says"]).size().sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
