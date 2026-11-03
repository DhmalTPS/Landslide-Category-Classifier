"""Experiment: can the type classifier get more accurate without new labels? (OVERNIGHT_PLAN step 5)

Goal / success criterion: out-of-fold macro-F1 up by >= 1 point over the shipped system (3-backbone probe
ensemble, mean of probabilities), with the site-bootstrap 90 % interval of the difference above 0.

Protocol (eval-rules): usable train+val photos only, site-grouped stratified 5-fold (same splitter and seed as
tools/robustness_eval.py); per fold the C of each backbone is chosen inside the training folds (choose_C) and
reused by every variant of that fold, heads are refit per fold with the phone-style copies. Test is never read.

Variants
  base        shipped: per-backbone head, mean of probabilities
  logpool     geometric mean of probabilities (log opinion pool)
  stack       weights per backbone chosen inside the training folds (nested 3-fold) on log-loss
  concat      one head on the concatenated (standardised) features of all backbones
  mc_test     multi-crop TTA at prediction time only: embedding = mean of full photo + 5 crops (0.85 scale)
  mc_both     multi-crop embeddings for the training photos too (copies unchanged)

Phase 1 `--embed`: multi-crop embeddings -> data/cache/emb_mc_<backbone>.npz (resumable, GPU).
Phase 2 (default): the table -> results/accuracy_eval.md, json -> results/accuracy_eval.json.
Usage: python tools/accuracy_eval.py --embed --budget 540   (repeat until DONE), then python tools/accuracy_eval.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, DATA, RESULTS, labeled_dataset  # noqa: E402
from probe import choose_C, fit, with_copies  # noqa: E402
from robustness_eval import LABELS, load_features, macro, paired_bootstrap, proba, site_weights  # noqa: E402

BACKBONES = ["facebook/dinov2-base", "facebook/dinov2-large", "google/siglip2-base-patch16-224"]
SCALE = 0.85


def mc_path(bb):
    return DATA / "cache" / f"emb_mc_{bb.replace('/', '_')}.npz"


def crops(img):
    W, H = img.size
    w, h = int(W * SCALE), int(H * SCALE)
    boxes = [((W - w) // 2, (H - h) // 2), (0, 0), (W - w, 0), (0, H - h), (W - w, H - h)]
    return [img.crop((x, y, x + w, y + h)) for x, y in boxes]


def embed(df, budget):
    from probe import Embedder, save_cache, variant
    import torch
    t0 = time.time()
    for bb in BACKBONES:
        p = mc_path(bb)
        cache = dict(np.load(p)) if p.exists() else {}
        todo = [r for r in df.itertuples() if r.image_id not in cache]
        if not todo:
            continue
        emb = Embedder(bb)
        for k, r in enumerate(todo):
            if time.time() - t0 > budget:
                save_cache(p, cache)
                print(f"PARTIAL ({bb.split('/')[-1]} {len(cache)}/{len(df)}) - run again")
                return False
            cache[r.image_id] = emb(crops(variant(r.image_id, r.file, 0))).mean(0)
            if k % 100 == 99:
                save_cache(p, cache)
        save_cache(p, cache)
        del emb
        torch.cuda.empty_cache()
    print("embeddings DONE")
    return True


def inner_oof(y, groups, tr, Xs, augs, Cs):
    Q = [np.zeros((len(y), len(CLASS_NAMES))) for _ in Xs]
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=1).split(tr, y[tr], groups[tr]):
        for j, (X, aug) in enumerate(zip(Xs, augs)):
            Q[j][tr[b]] = proba(fit(*with_copies(X, y, tr[a], aug), Cs[j]), X[tr[b]])
    return Q


def class_offsets(P, y, idx, sweeps=2):
    """Per-class log-offsets b maximising macro-F1 of argmax(log P + b) on rows idx (coordinate ascent)."""
    L = np.log(np.clip(P[idx], 1e-9, 1))
    b = np.zeros(L.shape[1])
    score = lambda bb: macro(y[idx], LABELS[(L + bb).argmax(1)])
    best = score(b)
    for _ in range(sweeps):
        for c in range(len(b)):
            for v in np.arange(-1.0, 1.01, 0.1):
                bb = b.copy(); bb[c] = v
                sc = score(bb)
                if sc > best + 1e-9:
                    best, b = sc, bb
    return b


def stack_weights(Ps, y, groups, tr, Xs, augs, Cs, Q=None):
    """Backbone weights (simplex grid, step 0.1) minimising log-loss of nested 3-fold OOF probs inside tr."""
    Q = Q if Q is not None else inner_oof(y, groups, tr, Xs, augs, Cs)
    best, bw = 1e9, None
    grid = [(a / 10, b / 10, 1 - a / 10 - b / 10) for a in range(11) for b in range(11 - a)]
    for w in grid:
        P = sum(wi * q[tr] for wi, q in zip(w, Q))
        ll = log_loss(y[tr], np.clip(P, 1e-6, 1), labels=CLASS_NAMES)
        if ll < best:
            best, bw = ll, w
    return bw


def evaluate(df):
    ids, y, groups = df.image_id.values, df.label.values, df.site_id.values
    Xs, augs, Ms = [], [], []
    for bb in BACKBONES:
        X, aug = load_features(bb, ids)
        mc = np.load(mc_path(bb))
        M = np.stack([mc[i] for i in ids])
        Xs.append(X); augs.append(aug); Ms.append((X + 5 * M) / 6)       # full photo + 5 crops, equal weight
    n, k = len(y), len(CLASS_NAMES)
    P = {v: np.zeros((n, k)) for v in ("base", "logpool", "stack", "concat", "mc_test", "mc_both",
                                        "aug_test", "mc_aug", "offsets", "mc_aug_offsets")}
    log = {"C": [], "stack_w": []}
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ids, y, groups)):
        t = time.time()
        Cs = [choose_C(X, y, groups, tr, aug)[0] for X, aug in zip(Xs, augs)]
        heads = [fit(*with_copies(X, y, tr, aug), C) for X, aug, C in zip(Xs, augs, Cs)]
        parts = [proba(m, X[te]) for m, X in zip(heads, Xs)]
        P["base"][te] = np.mean(parts, 0)
        lp = np.exp(np.mean([np.log(np.clip(q, 1e-9, 1)) for q in parts], 0))
        P["logpool"][te] = lp / lp.sum(1, keepdims=True)
        Q = inner_oof(y, groups, tr, Xs, augs, Cs)
        w = stack_weights(parts, y, groups, tr, Xs, augs, Cs, Q)
        # TTA over the cached phone-style copies (+ multi-crop): average probabilities over the views
        def tta(m, X, aug, M=None):
            views = [X[te]] + [A[te] for A in aug if not np.isnan(A[te]).any()] + ([M[te]] if M is not None else [])
            return np.mean([proba(m, v) for v in views], 0)
        P["aug_test"][te] = np.mean([tta(m, X, aug) for m, X, aug in zip(heads, Xs, augs)], 0)
        P["mc_aug"][te] = np.mean([tta(m, X, aug, M) for m, X, aug, M in zip(heads, Xs, augs, Ms)], 0)
        # per-class decision offsets fitted on the inner OOF probabilities (never on the outer fold)
        bo = class_offsets(np.mean(Q, 0), y, tr)
        sm = lambda Pv: (lambda Z: Z / Z.sum(1, keepdims=True))(np.exp(np.log(np.clip(Pv, 1e-9, 1)) + bo))
        P["offsets"][te] = sm(P["base"][te])
        P["mc_aug_offsets"][te] = sm(P["mc_aug"][te])
        log.setdefault("offsets", []).append([round(float(v), 2) for v in bo])
        P["stack"][te] = sum(wi * q for wi, q in zip(w, parts))
        Xc = [np.concatenate([X for X in Xs], 1)]
        ac = [np.concatenate([a[j] for a in augs], 1) for j in range(len(augs[0]))]
        Cc = choose_C(Xc[0], y, groups, tr, ac)[0]
        P["concat"][te] = proba(fit(*with_copies(Xc[0], y, tr, ac), Cc), Xc[0][te])
        P["mc_test"][te] = np.mean([proba(m, M[te]) for m, M in zip(heads, Ms)], 0)
        P["mc_both"][te] = np.mean([proba(fit(*with_copies(M, y, tr, aug), C), M[te])
                                    for M, aug, C in zip(Ms, augs, Cs)], 0)
        log["C"].append(Cs + [Cc]); log["stack_w"].append(w)
        print(f"  fold {f + 1}/5 ({time.time() - t:.0f} s) C={Cs} concat C={Cc} stack w={w}", flush=True)
    sw = site_weights(groups)
    rows, out = [], {}
    for v, Pv in P.items():
        pred = LABELS[Pv.argmax(1)]
        d = paired_bootstrap(Pv, P["base"], y, groups) if v != "base" else (0, 0, 0, 1)
        conf = Pv.max(1)
        ans = conf >= 0.6
        out[v] = {"f1": macro(y, pred), "f1_site": macro(y, pred, sw), "acc": float((pred == y).mean()),
                  "answered": float(ans.mean()), "acc_answered": float((pred[ans] == y[ans]).mean()),
                  "diff": d[0], "diff_lo": d[1], "diff_hi": d[2]}
        o = out[v]
        rows.append(f"| {v} | {o['f1']:.3f} | {o['f1_site']:.3f} | {o['acc']:.1%} | {o['answered']:.0%} / {o['acc_answered']:.1%} | "
                    + ("-" if v == "base" else f"{o['diff']:+.3f} [{o['diff_lo']:+.3f}, {o['diff_hi']:+.3f}]") + " |")
    lines = ["# Type accuracy: cheap improvements, out-of-fold (site-grouped 5-fold, usable train+val)", "",
             f"Made by `tools/accuracy_eval.py`; n = {n} photos, {len(set(groups))} sites. Test split not read.", "",
             "| variant | macro-F1 | site-weighted F1 | accuracy | answered (conf >= 0.6) / acc there | F1 diff vs base [90 % site bootstrap] |",
             "|---|---|---|---|---|---|"] + rows
    (RESULTS / "accuracy_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "accuracy_eval.json").write_text(json.dumps({"results": out, "log": log}, indent=1), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--budget", type=float, default=540)
    a = ap.parse_args()
    df = labeled_dataset()
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    if a.embed:
        if embed(df, a.budget):
            print("DONE")
        return
    evaluate(df)
    print("DONE")


if __name__ == "__main__":
    main()
