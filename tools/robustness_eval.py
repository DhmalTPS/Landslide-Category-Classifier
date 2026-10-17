"""How well does the classifier generalise to photos it has never seen? (robustness report + model choice)

One fixed test split can be lucky. This script uses cached image features (fast, CPU) to
estimate generalisation in harder ways, for the landslide-type head, for every feature
backbone and every average of them (ensembles). Only train+val photos are used (unless
--include-test, report only), so the test split stays untouched by model selection.

1. Site-grouped, class-stratified 5-fold cross-validation (shuffled, seeded) over the usable
   train+val photos: every photo is predicted by a model that never saw its site.
2. Unseen-source tests (sites never span both sides):
   - Wikimedia Commons -> Openverse/Flickr photos
   - outside South Asia -> South Asia (deployment region; common.is_south_asia)
   - non-phone cameras -> phone photos (Commons photos with known EXIF camera only)
3. Phone degradation: the CV models scored on fixed phone-style copies (src/phone_aug.py).
4. Calibration (ECE) and selective prediction (accuracy when answering only above a threshold).

Metrics in the robust score are SITE-BALANCED (each site weighs the same in total), so one
big event (e.g. 130 photos of one debris flow) cannot dominate a test. Regularisation C is
chosen exactly as in src/probe.py (probe.choose_C: grouped, stratified 3-fold CV), so the
models evaluated here are the models the app deploys.

Robustness score = mean site-balanced macro-F1 of CV and the three shift tests. The best
system becomes the app's model (models/selected.json, members probe_<name>; ties at 3 decimals
go to fewer members); the threshold is the first confidence level with >= --target-acc CV
accuracy at >= 50% coverage. A paired site bootstrap compares every system with the winner.

Usage:  python tools/robustness_eval.py [--backbones facebook/dinov2-base ...]
Output: results/robustness.md, results/robustness_cv.csv, models/selected.json
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import CLASS_NAMES, DATA, MODELS, RESULTS, data_fingerprint, is_south_asia, labeled_dataset  # noqa: E402
from probe import choose_C, fit, with_copies  # noqa: E402

SHORT = {"facebook/dinov2-base": "base", "facebook/dinov2-large": "large",
         "google/siglip2-base-patch16-224": "siglip"}
LABELS = np.array(CLASS_NAMES)


def site_weights(sites):
    s = pd.Series(sites)
    return (1.0 / s.map(s.value_counts())).values


def macro(y, p, w=None):
    return f1_score(y, p, average="macro", labels=sorted(set(y)), zero_division=0, sample_weight=w)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return out


def load_features(backbone, ids):
    cache = np.load(DATA / "cache" / f"emb_{backbone.replace('/', '_')}.npz")
    keys = set(cache.files)
    missing = [i for i in ids if i not in keys]
    if missing:
        raise KeyError(f"{len(missing)} photos without features")
    X = np.stack([cache[i] for i in ids])
    aug = []
    for k in (1, 2, 3):
        A = np.full_like(X, np.nan)
        for j, i in enumerate(ids):
            if f"{i}#{k}" in keys:
                A[j] = cache[f"{i}#{k}"]
        aug.append(A)
    return X, aug


def proba(model, X):
    P = np.zeros((len(X), len(CLASS_NAMES)))
    P[:, [CLASS_NAMES.index(c) for c in model.classes_]] = model.predict_proba(X)
    return P


def run_backbone(X, aug, y, groups, shifts):
    out = {"cv": np.zeros((len(y), len(CLASS_NAMES))), "phone": np.full((len(y), len(CLASS_NAMES)), np.nan),
           "folds": [], "C": [], "shift": {}}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        C, _ = choose_C(X, y, groups, tr, aug)
        m = fit(*with_copies(X, y, tr, aug), C)
        out["cv"][te] = proba(m, X[te])
        have = te[~np.isnan(aug[0][te]).any(1)]
        if len(have):
            out["phone"][have] = proba(m, aug[0][have])
        out["folds"].append(te)
        out["C"].append(C)
    for name, (trm, tem) in shifts.items():
        tr = np.where(trm)[0]
        if tem.sum() < 10 or len(set(y[tr])) < 2:
            continue
        C, _ = choose_C(X, y, groups, tr, aug)
        out["shift"][name] = proba(fit(*with_copies(X, y, tr, aug), C), X[tem])
    return out


def summarize(P, Pphone, folds, shift_P, y, groups, shifts):
    pred = LABELS[P.argmax(1)]
    w = site_weights(groups)
    fold_f1 = [macro(y[te], pred[te], site_weights(groups[te])) for te in folds]
    ph = ~np.isnan(Pphone).any(1)
    pph = LABELS[np.nan_to_num(Pphone, nan=-1)[ph].argmax(1)]
    s = {"cv_acc": float((pred == y).mean()), "cv_f1": float(macro(y, pred)),
         "cv_f1_site": float(macro(y, pred, w)),
         "fold_f1_mean": float(np.mean(fold_f1)), "fold_f1_sd": float(np.std(fold_f1)), "fold_f1_min": float(min(fold_f1)),
         "phone_f1": float(macro(y[ph], pph)) if ph.any() else float("nan"),
         "ece": float(ece(P.max(1), pred == y))}
    parts = [s["cv_f1_site"]]
    for name, (trm, tem) in shifts.items():
        if name not in shift_P:
            s[f"{name}_acc"] = s[f"{name}_f1"] = float("nan")
            continue
        p = LABELS[shift_P[name].argmax(1)]
        s[f"{name}_acc"] = float((p == y[tem]).mean())
        s[f"{name}_f1"] = float(macro(y[tem], p, site_weights(groups[tem])))
        parts.append(s[f"{name}_f1"])
    s["robust_score"] = float(np.mean(parts))
    return s


def paired_bootstrap(P_a, P_b, y, groups, n=1000, seed=0):
    """Site bootstrap of CV macro-F1(a) - macro-F1(b): mean, 90% interval, P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    sites = np.unique(groups)
    idx_of = {s: np.where(groups == s)[0] for s in sites}
    pa, pb = LABELS[P_a.argmax(1)], LABELS[P_b.argmax(1)]
    d = []
    for _ in range(n):
        idx = np.concatenate([idx_of[s] for s in rng.choice(sites, len(sites))])
        d.append(macro(y[idx], pa[idx]) - macro(y[idx], pb[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95)), float((d <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(SHORT))
    ap.add_argument("--target-acc", type=float, default=0.85, help="accuracy the 'confident' answers should reach")
    ap.add_argument("--no-select", action="store_true", help="report only; do not write models/selected.json")
    ap.add_argument("--include-test", action="store_true",
                    help="also use test photos (report only: selection would see test, so --no-select is forced)")
    args = ap.parse_args()

    df = labeled_dataset()
    if args.include_test:
        args.no_select = True
    else:
        df = df[df.split.isin(["train", "val"])]
    df = df.reset_index(drop=True)
    ids, y, groups = df.image_id.values, df.label.values, df.site_id.values
    south = df.apply(is_south_asia, axis=1).values
    commons = (df.source == "wikimedia_commons").values
    known_cam = (df.get("camera_checked", pd.Series("", index=df.index)) == "1").values
    phone = (df.get("is_phone", pd.Series("", index=df.index)) == "1").values

    def no_leak(trm, tem):   # a site never spans both sides of a shift test
        return trm & ~df.site_id.isin(set(df.site_id[tem])).values, tem
    shifts = {"flickr": no_leak(commons, ~commons), "south_asia": no_leak(~south, south),
              "phone_cam": no_leak(known_cam & ~phone, phone)}

    runs, skipped = {}, []
    for bb in args.backbones:
        try:
            X, aug = load_features(bb, ids)
        except (FileNotFoundError, KeyError) as e:
            skipped.append(f"{bb} ({e})")
            print(f"skip {bb}: {e}")
            continue
        print(f"running {bb} ...", flush=True)
        runs[SHORT.get(bb, bb)] = run_backbone(X, aug, y, groups, shifts) | {"backbone": bb}
    if not runs:
        raise SystemExit("no backbone has cached features; run tools/precompute_embeddings.py")

    rows, systems = [], {}
    for r in range(1, len(runs) + 1):
        for combo in itertools.combinations(sorted(runs), r):
            P = np.mean([runs[b]["cv"] for b in combo], axis=0)
            Pph = np.mean([runs[b]["phone"] for b in combo], axis=0)
            names = set.intersection(*[set(runs[b]["shift"]) for b in combo])
            sP = {n: np.mean([runs[b]["shift"][n] for b in combo], axis=0) for n in names}
            s = summarize(P, Pph, runs[combo[0]]["folds"], sP, y, groups, shifts)
            systems[combo] = (P, s)
            rows.append({"system": "+".join(combo), **{k: round(v, 3) for k, v in s.items()}})
    table = pd.DataFrame(rows)
    table["members"] = table.system.str.count("[+]") + 1
    table = table.assign(rs=table.robust_score.round(3)).sort_values(["rs", "members"], ascending=[False, True]).drop(columns="rs")
    best = tuple(table.iloc[0].system.split("+"))
    P, s = systems[best]
    pred, conf = LABELS[P.argmax(1)], P.max(1)
    pd.DataFrame({"image_id": ids, "label": y, "pred": pred, "conf": conf.round(4)}).to_csv(
        RESULTS / "robustness_cv.csv", index=False)
    boots = {sys_: paired_bootstrap(P, systems[tuple(sys_.split("+"))][0], y, groups)
             for sys_ in table.system if tuple(sys_.split("+")) != best}

    thresholds = []
    for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= t
        thresholds.append((t, m.mean(), (pred[m] == y[m]).mean(), macro(y[m], pred[m])) if m.any()
                          else (t, 0.0, float("nan"), float("nan")))
    conf_min = next((t for t, cov, acc, _ in thresholds if acc >= args.target_acc and cov >= 0.5), 0.5)

    top_site = pd.Series(groups[phone]).value_counts()
    scope = "train+val+test (report only)" if args.include_test else "train+val only (test split untouched)"
    md = ["# Robustness report\n",
          f"Photos used: {scope}. Usable labeled photos: {len(df)} from {df.site_id.nunique()} sites "
          f"(South Asia {int(south.sum())}, Openverse {int((~commons).sum())}, phone EXIF {int(phone.sum())}; "
          f"known non-phone camera {int((known_cam & ~phone).sum())}). "
          "All numbers measure agreement with the AI-made labels (see report.md, label caveat).\n",
          *([f"Skipped backbones: {', '.join(skipped)}\n"] if skipped else []),
          "## All systems (sorted by robustness score)\n",
          "robust score = mean **site-balanced** macro-F1 of: CV, Commons→Openverse, outside→South Asia, "
          "known non-phone cameras→phone. Δ vs selected = paired site bootstrap of CV macro-F1 "
          "(selected − system): mean [90% interval], P(Δ ≤ 0).\n",
          "| System | robust score | CV macro-F1 site-bal. (plain; fold mean ± sd) | CV acc | phone-degraded F1 | "
          "→Openverse F1 | →South Asia F1 (acc) | →phone cam F1 | ECE | Δ vs selected |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        b = boots.get(r.system)
        bs = f"{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}], P={b[3]:.2f}" if b else "selected"
        md.append(f"| {r.system} | **{r.robust_score:.3f}** | {r.cv_f1_site:.3f} ({r.cv_f1:.3f}; {r.fold_f1_mean:.3f} ± "
                  f"{r.fold_f1_sd:.3f}) | {r.cv_acc:.3f} | {r.phone_f1:.3f} | {r.flickr_f1:.3f} | "
                  f"{r.south_asia_f1:.3f} ({r.south_asia_acc:.3f}) | {r.phone_cam_f1:.3f} | {r.ece:.3f} | {bs} |")
    md += [f"\nLargest site among phone photos: {top_site.index[0]} ({top_site.iloc[0]} of {phone.sum()})"
           if len(top_site) else "",
           f"\n## Selected: `{'+'.join(best)}`\n",
           "### Per class (site-grouped CV)\n", "| Class | n | recall | precision |", "|---|---|---|---|"]
    for c in CLASS_NAMES:
        n = int((y == c).sum())
        md.append(f"| {c} | {n} | {(pred[y == c] == c).mean() if n else float('nan'):.2f} | "
                  f"{(y[pred == c] == c).mean() if (pred == c).any() else float('nan'):.2f} |")
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted")).reindex(
        index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
    cm_md = ["| true \\ predicted | " + " | ".join(CLASS_NAMES) + " |", "|---" * (len(CLASS_NAMES) + 1) + "|"]
    cm_md += [f"| {c} | " + " | ".join(str(v) for v in cm.loc[c]) + " |" for c in CLASS_NAMES]
    md += ["\n### Confusion (CV)\n", "\n".join(cm_md),
           "\n### Selective prediction (CV, type head only, before the photo check)\n",
           "| Confidence ≥ | Coverage | Accuracy on answered | Macro-F1 on answered |", "|---|---|---|---|"]
    md += [f"| {t:.1f} | {cov:.2f} | {acc:.3f} | {f1:.3f} |" for t, cov, acc, f1 in thresholds]
    md.append(f"\nApp threshold: answer confidently at ≥ {conf_min:.1f} (first threshold with accuracy ≥ "
              f"{args.target_acc:.2f} at ≥ 50% coverage); below it the app says 'uncertain' and shows the top two.")
    md.append(f"\nC chosen per CV fold: " + "; ".join(f"{b}: {runs[b]['C']}" for b in sorted(runs)))
    (RESULTS / "robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    if not args.no_select:
        code_hash = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        sel = {"members": [f"probe_{b}" for b in best],
               "backbones": {f"probe_{b}": runs[b]["backbone"] for b in best},
               "confident_min": conf_min, "robust_score": s["robust_score"],
               "selected_by": "highest site-balanced robust score (CV + 3 shift tests) on train+val; "
                              "tools/robustness_eval.py",
               "selection_code_sha1": code_hash, "date": time.strftime("%Y-%m-%d %H:%M"),
               "data_fingerprint": data_fingerprint(),
               "systems": table.to_dict("records")}
        (MODELS / "selected.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        print(f"\nselected {sel['members']} (robust score {s['robust_score']:.3f}), confident_min {conf_min}")


if __name__ == "__main__":
    main()
