"""Calibrate and test the overlapping-set check (src/overlap.py) on simulated sets made from real photos.

Real overlapping sets of our sites do not exist yet, so sets are simulated from single labeled photos with
known geometry:
  turn   virtual camera rotations: view = K R K^-1 of the photo (exact for a camera turning on the spot),
         with roll jitter (+-2 deg), exposure jitter (x0.85-1.15), JPEG re-compression and noise
  walk   virtual sideways steps: every pixel shifts by f*t/Z with a smooth random depth field Z (near rocks,
         far ridge), so no homography fits (parallax); a planar depth would be a homography and is avoided
  none   two photos of different sites (must not be linked)
Train photos set the thresholds (written to src/overlap_check.json):
  min_inliers  smallest count with <= 0.5 % of unrelated pairs linked (at least 30)
  parallax_px  95th percentile of turn pairs (so <= 5 % of correct turning sets are called "moved")
  weak         smallest true overlap where >= 95 % of turn pairs link (below it joining may fail)
mosaic_weight: train+val photos are split into 3 overlapping views, joined, and the type taken from
w * joined + (1 - w) * mean(views), w in {0, .25, .5, .75, 1}, best macro-F1. The deployed heads saw every
train+val photo, so here the heads are refit out of fold (site-grouped 5-fold). Test photos are never read.

Usage: python tools/overlap_eval.py [--n-train 120] [--n-val 150] [--no-type]
Writes results/overlap_eval.md and src/overlap_check.json.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import IMAGES, RESULTS, labeled_dataset  # noqa: E402
import overlap as O  # noqa: E402

rng = np.random.default_rng(0)
STATE = RESULTS / "overlap_eval_state"          # resumable chunks (gitignored results/)
T0 = time.time()
BUDGET = 1e9                                     # seconds per call; set by --budget


def seed(key):
    """Per-item random stream, so a resumed run draws exactly what an uninterrupted one would."""
    global rng
    # full hash of the key (ids share long prefixes like 'commons_', so truncating the key repeats streams)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little"))


def over_budget():
    return time.time() - T0 > BUDGET


def cached(name):
    """(done dict from <name>.jsonl, appender)."""
    STATE.mkdir(parents=True, exist_ok=True)
    f = STATE / f"{name}.jsonl"
    done = {}
    if f.exists():
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r["val"]
    fh = open(f, "a", encoding="utf-8")

    def add(key, val):
        done[key] = val
        fh.write(json.dumps({"key": key, "val": val}) + "\n")
        fh.flush()
    return done, add


class Partial(Exception):
    pass


def load(image_id, max_side=1600):
    im = ImageOps.exif_transpose(Image.open(next(IMAGES.glob(f"{image_id}.*")))).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def rot(yaw, pitch=0.0, roll=0.0):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def K(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def degrade(a):
    a = np.clip(a.astype(float) * rng.uniform(0.85, 1.15) + rng.normal(0, 2.0, a.shape), 0, 255).astype(np.uint8)
    ok, b = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(80, 95))])
    return cv2.imdecode(b, cv2.IMREAD_UNCHANGED)


def distort(a, k1):
    """Residual radial lens distortion (phones correct most of it): views stop being exact homographies."""
    h, w = a.shape[:2]
    f = float(max(w, h))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xs - w / 2) / f, (ys - h / 2) / f
    s = 1 + k1 * (xn * xn + yn * yn)
    return cv2.remap(a, xn * s * f + w / 2, yn * s * f + h / 2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def turn_views(img, yaws, frac=0.55, pitch=0.0, jitter=True):
    """Views of a camera turning on the spot. Returns views and the true homographies view_i -> view_0
    (true up to the simulated lens distortion, which is the same lens for the whole set)."""
    a = np.asarray(img)
    k1 = rng.uniform(-0.08, 0.04) if jitter else 0.0
    H0, W0 = a.shape[:2]
    f = O.DEFAULT_F35 / 36 * max(W0, H0)
    w, h = int(W0 * frac), int(H0 * 0.8)
    Ks, Ko = K(f, W0, H0), K(f, w, h)
    views, Rs = [], []
    for y in yaws:
        R = rot(y, pitch + (rng.normal(0, 0.7) if jitter else 0), rng.normal(0, 1.0) if jitter else 0)
        M = Ks @ R @ np.linalg.inv(Ko)                       # view pixel -> source pixel
        v = cv2.warpPerspective(a, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        views.append(Image.fromarray(degrade(distort(v, k1)) if jitter else v))
        Rs.append(R)
    Hs = [Ko @ Rs[0].T @ R @ np.linalg.inv(Ko) for R in Rs]   # view_i -> view_0
    return views, Hs, (w, h), f


def max_yaw(img, frac=0.55):
    W0, H0 = img.size
    f = O.DEFAULT_F35 / 36 * max(W0, H0)
    return math.degrees(math.atan(W0 / 2 / f) - math.atan(W0 * frac / 2 / f)) * 0.9


def true_overlap(Hab, w, h):
    sa, _ = O._inter(O._quad(Hab, w, h), w, h)
    sb, _ = O._inter(O._quad(np.linalg.inv(Hab), w, h), w, h)
    return (sa + sb) / (2 * w * h)


def walk_pair(img, shift_frac, frac=0.6, amp=0.35):
    """Two views a sideways step apart over a bumpy depth field (parallax)."""
    a = np.asarray(img).astype(np.float32)
    H0, W0 = a.shape[:2]
    w = int(W0 * frac)
    ys, xs = np.mgrid[0:H0, 0:W0].astype(np.float32)
    z = 1.0 + 0.8 * (1 - ys / H0)                                      # farther towards the top
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H0, W0)).astype(np.float32), (0, 0), max(H0, W0) / 12)
    z = z * (1 + amp * noise / (noise.std() + 1e-6))
    z = np.clip(z, 0.4, None)
    disp = shift_frac * W0 / z                                        # pixels, larger for near ground
    disp = disp * (shift_frac * W0 / np.median(disp))                 # median shift = shift_frac * W0
    x0 = (W0 - w - shift_frac * W0) / 2
    va = a[:, int(x0):int(x0) + w]
    mapx = (xs[:, :w] + x0 + disp[:, :w]).astype(np.float32)
    vb = cv2.remap(a, mapx, ys[:, :w], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(degrade(va.astype(np.uint8))), Image.fromarray(degrade(vb.astype(np.uint8)))


def calibrate(train, n):
    ids = list(train.image_id)
    random.Random(0).shuffle(ids)
    rows_turn, rows_walk, rows_none = [], [], []
    t0 = time.time()
    done, add = cached(f"calibrate_{n}")
    for k, iid in enumerate(ids[:n]):
        if iid in done:
            rows_turn += done[iid][0]; rows_walk += done[iid][1]
            continue
        if over_budget():
            raise Partial(f"calibrate {len(done)}/{n}")
        seed(iid)
        nt, nw = len(rows_turn), len(rows_walk)
        img = load(iid)
        frac = 0.35
        ym = max_yaw(img, frac)
        fov = 2 * math.degrees(math.atan(img.size[0] * frac / 2 / (O.DEFAULT_F35 / 36 * max(img.size))))
        if ym < 4:
            add(iid, [[], []])
            continue
        for _ in range(3):                           # turn pairs with target overlap 2-92 %
            d = fov * (1 - rng.uniform(0.02, 0.92))
            if d > 2 * ym:
                d = rng.uniform(0.1, 2 * ym)
            c = rng.uniform(-ym + d / 2, ym - d / 2)
            views, Hs, (w, h), f = turn_views(img, [c - d / 2, c + d / 2], frac=frac)
            tov = true_overlap(np.linalg.inv(Hs[1]), w, h)   # view_0 -> view_1
            fa, fb = O.features(views[0]), O.features(views[1])
            p = O.pair(fa, fb)
            rows_turn.append({"true": tov, "est": p["overlap"], "inliers": p["inliers"], "parallax": p["parallax"],
                              "linked": p["inliers"] >= 12 and p["H"] is not None})
        for s in (0.08, 0.15):                       # walk pairs
            va, vb = walk_pair(img, s)
            p = O.pair(O.features(va), O.features(vb))
            rows_walk.append({"shift": s, "inliers": p["inliers"], "parallax": p["parallax"], "motion": p["motion"]})
        add(iid, [rows_turn[nt:], rows_walk[nw:]])
        if k % 20 == 0:
            print(f"  calibrate {k}/{n} ({time.time() - t0:.0f} s)", flush=True)
    # unrelated pairs: different sites
    sub = train.drop_duplicates("site_id").sample(min(len(train.site_id.unique()), 2 * n), random_state=0)
    keys = list(sub.image_id)
    done_n, add_n = cached(f"unrelated_{n}")
    for i in range(0, len(keys) - 1, 2):
        key = f"{keys[i]}|{keys[i + 1]}"
        if key not in done_n:
            if over_budget():
                raise Partial(f"unrelated {len(done_n)}/{len(keys) // 2}")
            p = O.pair(O.features(load(keys[i], 1280)), O.features(load(keys[i + 1], 1280)))
            add_n(key, {"inliers": p["inliers"], "overlap": p["overlap"]})
        rows_none.append(done_n[key])
    return rows_turn, rows_walk, rows_none


def choose(rows_turn, rows_walk, rows_none):
    none_in = np.array([r["inliers"] for r in rows_none])
    min_inl = 30
    while (none_in >= min_inl).mean() > 0.005:
        min_inl += 5
    good = [r for r in rows_turn if r["parallax"] is not None and math.isfinite(r["parallax"]) and r["inliers"] >= min_inl]
    par = float(np.percentile([r["parallax"] for r in good], 95)) if good else O._DEFAULT["parallax_px"]
    par = round(max(par, 1.5), 2)              # never tighter than 1.5 px (real lenses, hand shake)
    bins = np.arange(0.0, 0.95, 0.05)
    curve = []
    for b in bins:
        sel = [r for r in rows_turn if b <= r["true"] < b + 0.05]
        if sel:
            curve.append((round(b, 2), len(sel), float(np.mean([r["inliers"] >= min_inl and r["linked"] for r in sel]))))
    weak = next((b for b, nb, s in curve if nb >= 5 and s >= 0.95 and all(s2 >= 0.9 for b2, nb2, s2 in curve if b2 > b and nb2 >= 5)), 0.25)
    gap = next((b + 0.05 for b, nb, s in reversed(curve) if nb >= 5 and s < 0.5), 0.10)
    return min_inl, par, max(0.15, round(weak + 0.05, 2)), max(0.05, round(gap, 2)), curve


def type_eval(tv, n, folds=5):
    """Type from a 3-photo turning set vs the whole photo, with OUT-OF-FOLD heads: the deployed probe heads
    were refit on all train+val, so they have seen every val photo. Here each sampled photo is classified by
    heads (same backbones, same C) refit without its site (site-grouped 5-fold over train+val, no phone-style
    copies). Test photos are never used."""
    from predict import Predictor
    from probe import fit
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from common import CLASS_NAMES, DATA
    pred = Predictor("auto")
    members = list(pred.type_probes)
    bund = {m: pred.bundles[m] for m in members}
    caches = {m: np.load(DATA / "cache" / f"emb_{bund[m]['backbone'].replace('/', '_')}.npz") for m in members}
    keys = {m: set(c.files) for m, c in caches.items()}
    tv = tv[[all(i in keys[m] for m in members) for i in tv.image_id]].reset_index(drop=True)
    X = {m: np.stack([caches[m][i] for i in tv.image_id]) for m in members}
    y, groups = tv.label.values, tv.site_id.values
    sample = set(tv.sample(min(n, len(tv)), random_state=0).index)
    feats, skipped, t0 = {}, 0, time.time()
    tdir = STATE / f"type_{n}"
    tdir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(sorted(sample)):
        iid = tv.image_id[idx]
        fz = tdir / f"{iid}.npz"
        if fz.exists():
            z = np.load(fz)
            if "skip" in z.files:
                skipped += 1
            else:
                feats[idx] = {m: z[m] for m in members}
            continue
        if over_budget():
            raise Partial(f"type {len(list(tdir.glob('*.npz')))}/{len(sample)}")
        seed(iid)
        img = load(iid)
        ym = max_yaw(img, 0.5)
        if ym < 3:
            skipped += 1; np.savez(fz, skip=1)
            continue
        views, _, _, _ = turn_views(img, [-ym, 0, ym], frac=0.5)
        _, ex = O.analyse(views, "turn")
        if ex["mosaic"] is None:
            skipped += 1; np.savez(fz, skip=1)
            continue
        ims = [img] + views + [ex["mosaic"]]
        feats[idx] = {m: np.asarray(pred.embedders[bund[m]["backbone"]](ims)) for m in members}
        np.savez(fz, **feats[idx])
        if k % 25 == 0:
            print(f"  type {k}/{len(sample)} ({time.time() - t0:.0f} s)", flush=True)

    def align(P, classes):
        out = np.zeros((len(P), len(CLASS_NAMES)))
        for j, c in enumerate(classes):
            out[:, CLASS_NAMES.index(c)] = P[:, j]
        return out
    probs = {}
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=0).split(np.zeros(len(tv)), y, groups):
        te = [i for i in te if i in feats]
        if not te:
            continue
        heads = {m: fit(X[m][tr], y[tr], bund[m]["C"]["type"]) for m in members}
        for i in te:
            probs[i] = np.mean([align(heads[m].predict_proba(feats[i][m]), heads[m].classes_) for m in members], axis=0)
    ids = sorted(probs)
    ys = [y[i] for i in ids]
    lab = lambda v: CLASS_NAMES[int(np.argmax(v))]
    W = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = sorted(set(ys))
    f1 = lambda p: f1_score(ys, p, labels=labels, average="macro", zero_division=0)
    acc = lambda p: float(np.mean([a == b for a, b in zip(ys, p)]))
    orig = [lab(probs[i][0]) for i in ids]
    table = {"whole photo (reference)": (f1(orig), acc(orig), 1.0)}
    single = [lab(probs[i][1:4].mean(0)) for i in ids]
    for w in W:
        p = [lab(w * probs[i][4] + (1 - w) * probs[i][1:4].mean(0)) for i in ids]
        table[f"w = {w}"] = (f1(p), acc(p), float(np.mean([a == b for a, b in zip(p, orig)])))
    one = [lab(probs[i][2]) for i in ids]
    table["middle photo alone"] = (f1(one), acc(one), float(np.mean([a == b for a, b in zip(one, orig)])))
    best = max(W, key=lambda w: (round(table[f"w = {w}"][0], 3), -abs(w - 0.5)))
    del single
    return best, table, len(ids), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=300, help="train+val photos for the out-of-fold type check")
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--budget", type=float, default=1e9, help="stop after this many seconds and resume on the next call")
    args = ap.parse_args()
    global BUDGET
    BUDGET = args.budget
    try:
        run(args)
    except Partial as e:
        print(f"PARTIAL ({e}) - run the same command again to continue")
        return
    print("DONE")


def run(args):
    df = labeled_dataset()
    train, val = df[df.split == "train"], df[df.split == "val"]
    tv = df[df.split.isin(["train", "val"])]
    print(f"train {len(train)} photos, val {len(val)} (test not read)")
    rt, rw, rn = calibrate(train, args.n_train)
    min_inl, par, weak, gap, curve = choose(rt, rw, rn)
    O.CFG.update(min_inliers=min_inl, parallax_px=par)
    walk_det = {s: float(np.mean([r["motion"] == "moved" or r["inliers"] < min_inl for r in rw if r["shift"] == s])) for s in (0.08, 0.15)}
    walk_moved = {s: float(np.mean([r["parallax"] is not None and r["parallax"] > par for r in rw if r["shift"] == s])) for s in (0.08, 0.15)}
    lk = [r for r in rt if r["inliers"] >= min_inl and r["linked"] and r["true"] > 0.05]
    mae = float(np.mean([abs(r["est"] - r["true"]) for r in lk])) if lk else float("nan")
    false_link = float(np.mean([r["inliers"] >= min_inl for r in rn]))
    cfg = dict(O._DEFAULT)
    cfg["bands"] = json.loads(json.dumps(O._DEFAULT["bands"]))
    cfg.update(min_inliers=min_inl, parallax_px=par)
    cfg["bands"]["turn"]["weak"] = weak
    cfg["bands"]["turn"]["gap"] = min(gap, weak - 0.05)
    cfg["bands"]["turn"]["ideal"] = [max(0.30, weak), max(0.60, weak + 0.25)]   # the guide aims inside the safe zone
    lines = ["# Overlapping photo sets: calibration and test (simulated sets from real photos)", "",
             "Made by `tools/overlap_eval.py`. Train photos set the thresholds, out-of-fold train+val photos choose how the "
             "joined picture and the single photos are combined; test photos are not read.", "",
             f"## Train: {len(rt)} turn pairs, {len(rw)} walk pairs, {len(rn)} unrelated pairs", "",
             f"- Link threshold: **{min_inl} inliers** (unrelated pairs linked: {false_link:.1%})",
             f"- Overlap estimate error (linked turn pairs): mean absolute error **{mae * 100:.1f} points**",
             f"- Parallax threshold: **{par} px** (95th percentile of turn pairs at 1024 px)",
             f"- Walk pairs called moved or not linked: step 8 % of width {walk_det[0.08]:.0%}, 15 % {walk_det[0.15]:.0%} "
             f"(parallax above threshold: {walk_moved[0.08]:.0%}, {walk_moved[0.15]:.0%})",
             f"- 'Weak' overlap band below **{weak:.0%}** (95 % of turn pairs link from {weak - 0.05:.0%} up); "
             f"'gap' below **{cfg['bands']['turn']['gap']:.0%}**; ideal band {cfg['bands']['turn']['ideal'][0]:.0%}-{cfg['bands']['turn']['ideal'][1]:.0%}", "",
             "| true overlap | pairs | linked |", "|---|---|---|"]
    lines += [f"| {b:.0%}-{b + 0.05:.0%} | {nb} | {s:.0%} |" for b, nb, s in curve]
    cfg["calibrated"] = {"train_turn_pairs": len(rt), "train_walk_pairs": len(rw), "train_unrelated": len(rn),
                         "overlap_mae": round(mae, 4), "false_link": round(false_link, 4),
                         "walk_detected": walk_det}

    def write():
        O._CFG.write_text(json.dumps(cfg, indent=1))
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "overlap_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write()                                   # train calibration is kept even if the val stage fails
    print("\n".join(lines), flush=True)
    if not args.no_type and len(tv):
        O.CFG.update(cfg)
        best, table, nv, skipped = type_eval(tv, args.n_val)
        cfg["mosaic_weight"] = best
        extra = ["", f"## Type from a 3-photo turning set: {nv} train+val photos, out-of-fold heads (site-grouped 5-fold)", "",
                 "Each photo is split into 3 overlapping views (half its width each), joined, and classified by probe heads "
                 f"refit without its site ({skipped} photos skipped: too narrow to split or not joined).", "",
                 "| type from | macro-F1 | accuracy | same answer as whole photo |", "|---|---|---|---|"]
        extra += [f"| {k} | {f:.3f} | {a:.1%} | {g:.1%} |" for k, (f, a, g) in table.items()]
        extra += ["", f"Chosen: **w = {best}** (w = weight of the joined picture)."]
        lines += extra
        write()
        print("\n".join(extra))


if __name__ == "__main__":
    main()
