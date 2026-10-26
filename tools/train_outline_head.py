"""Self-train a landslide-outline head on frozen DINOv2 patch features, with no human masks and no leakage.

Pseudo-labels come from CONSENSUS. A patch is "landslide" only where CLIPSeg (p > 0.6), the jittered SAM
soft mask (> 0.6) and the pixel prior (not vegetation/sky) all agree. It is "background" where CLIPSeg
and SAM both say no (< 0.2), or the pixel prior is sure it is vegetation/sky (> 0.8). Everything else is
left unlabelled. A logistic-regression head (standardised 768-d features, balanced classes) learns from
the confident patches.

Self-training round 2: unlabelled TRAIN patches that the round-1 head is very sure about (p > 0.95 or
< 0.05) are added as labels, and the head is retrained. The round is kept only if validation AUC improves.

Leakage control:
  - fit on the train split only; C (regularisation) and the round are chosen on the val split;
  - the test split is never read;
  - splits.csv separates sites, so photos of one landslide are never in both train and val.
The saved bundle stores the train image ids so this can be audited (tools/check_integrity.py style).

  python tools/train_outline_head.py --max-train 500 --max-val 150
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dim_stats as S
from common import IMAGES, labeled_dataset
from dimension import HEAD_BACKBONE, HEAD_FILE, PROMPTS, Dimensioner
from predict import load_image

SIZE, PATCH = 448, 14
G = SIZE // PATCH


def pool(m):
    """Mask (H, W) -> per-patch fraction on the DINOv2 grid."""
    r = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
    return r.reshape(G, PATCH, G, PATCH).mean((1, 3)).ravel()


def collect(dim, rows, rng, n_lab=200, n_unl=100):
    X, y, U, ids = [], [], [], []
    for i, r in enumerate(rows.itertuples(), 1):
        try:
            img = load_image(next(IMAGES.glob(f"{r.image_id}.*")))
        except StopIteration:
            continue
        p = dim.clip_prob(img, r.label)
        m0 = S.clean_mask(p > 0.5)
        if not 0.005 < m0.mean() < 0.92:
            continue
        emb, inp = dim.sam_embed(img)
        soft, _ = dim.sam_soft(img, m0, p, emb, inp)
        if soft is None:
            continue
        bad = S.bad_prior(img)
        pos = pool((p > 0.6) & (soft > 0.6) & (bad < 0.3))
        neg = pool(((p < 0.2) & (soft < 0.2)) | (bad > 0.8))
        f = dim.patch_features(img, SIZE).reshape(-1, 768)[: G * G]
        lab = np.full(G * G, -1)
        lab[pos >= 0.8] = 1
        lab[neg >= 0.8] = 0
        li, ui = np.nonzero(lab >= 0)[0], np.nonzero(lab < 0)[0]
        if len(li) == 0:
            continue
        li = rng.choice(li, min(n_lab, len(li)), replace=False)
        ui = rng.choice(ui, min(n_unl, len(ui)), replace=False) if len(ui) else ui
        X.append(f[li].astype(np.float16))
        y.append(lab[li])
        U.append(f[ui].astype(np.float16))
        ids.append(r.image_id)
        if i % 25 == 0:
            print(f"    {i}/{len(rows)}", flush=True)
    return np.concatenate(X), np.concatenate(y), np.concatenate(U), ids


def evaluate(clf, X, y):
    p = clf.predict_proba(X.astype(np.float32))[:, 1]
    pr, gt = p > 0.5, y == 1
    return float(roc_auc_score(y, p)), float((pr & gt).sum() / max((pr | gt).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=500)
    ap.add_argument("--max-val", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    df = labeled_dataset()
    df = df[df.label.isin(list(PROMPTS))]
    tr = df[df.split == "train"].sample(frac=1, random_state=args.seed).head(args.max_train)
    va = df[df.split == "val"].sample(frac=1, random_state=args.seed).head(args.max_val)
    assert not set(tr.site_id) & set(va.site_id), "site leakage between train and val"
    assert "test" not in set(tr.split) | set(va.split)
    dim = Dimensioner(depth=False, extras=False)
    dim.head = None
    import transformers as T
    dim.head_bb = T.AutoModel.from_pretrained(HEAD_BACKBONE).to(dim.device).eval()
    dim.head_proc = T.AutoImageProcessor.from_pretrained(HEAD_BACKBONE)
    t0 = time.time()
    print(f"collecting consensus pseudo-labels: train {len(tr)} photos, val {len(va)} photos")
    Xtr, ytr, Utr, tr_ids = collect(dim, tr, rng)
    Xva, yva, _, va_ids = collect(dim, va, rng)
    print(f"  train patches {len(ytr)} ({ytr.mean():.0%} landslide) from {len(tr_ids)} photos; "
          f"val patches {len(yva)} from {len(va_ids)} photos ({time.time() - t0:.0f}s)")
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        clf.fit(Xtr.astype(np.float32), ytr)
        auc, iou = evaluate(clf, Xva, yva)
        print(f"  round 1  C={C:<5} val AUC {auc:.3f}  IoU {iou:.3f}")
        if best is None or auc > best["val_auc"]:
            best = {"clf": clf, "C": C, "val_auc": auc, "val_iou": iou, "round": 1}
    # round 2: self-training on confident unlabelled train patches
    pu = best["clf"].predict_proba(Utr.astype(np.float32))[:, 1]
    sel = (pu > 0.95) | (pu < 0.05)
    X2 = np.concatenate([Xtr, Utr[sel]])
    y2 = np.concatenate([ytr, (pu[sel] > 0.5).astype(int)])
    clf2 = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], class_weight="balanced", max_iter=2000))
    clf2.fit(X2.astype(np.float32), y2)
    auc2, iou2 = evaluate(clf2, Xva, yva)
    print(f"  round 2  +{sel.sum()} self-labelled patches: val AUC {auc2:.3f}  IoU {iou2:.3f}")
    if auc2 > best["val_auc"]:
        best.update(clf=clf2, val_auc=auc2, val_iou=iou2, round=2)
    # fusion weight: the head only gets a full vote when it is clearly good on validation
    weight = float(np.clip((best["val_auc"] - 0.75) / 0.15, 0, 1))
    bundle = {**best, "backbone": HEAD_BACKBONE, "size": SIZE, "weight": round(weight, 2),
              "train_ids": tr_ids, "val_ids": va_ids, "seed": args.seed}
    if weight == 0:
        print("val AUC too low (<= 0.75): not saving the head")
        return
    joblib.dump(bundle, HEAD_FILE)
    print(f"saved {HEAD_FILE} (round {best['round']}, C={best['C']}, val AUC {best['val_auc']:.3f}, "
          f"IoU {best['val_iou']:.3f}, fusion weight {weight:.2f})")


if __name__ == "__main__":
    main()
