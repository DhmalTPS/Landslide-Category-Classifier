"""Validate the walk-along metric scale (overlap.stereo_points) on simulated stereo pairs with known depth.

Simulation (labelled as such in the report): a train photo is draped over a metric depth field
Z(x, y) (median distance 20-200 m, farther towards the top, smooth bumps); a second camera stands B metres
to the side (B = steps x 0.75 m, steps 2/4/8) and sees pixel x where the first sees x + f B / Z. Both views
get noise and JPEG re-compression. Field errors are simulated too: the walked distance the engineer reports
is off by a log-normal factor (sd 0.15) and the focal length is off by sd 0.02 (EXIF) or 0.25 (no EXIF).

What matters for sizes is the error of (metric depth / focal): size = Z * pixels / f, and the triangulated
Z scales with the assumed focal, so most of a focal error cancels. Reported per pair:
  err = median log(z_tri / z_true) - log(f_assumed / f_true) - ... (the scale error a size would carry)
  sd  = the uncertainty the app would state (baseline 0.15 + robust spread / sqrt(min(n, 10)) + 0.3 x focal sd)
Gate (OVERNIGHT_PLAN step 2): typical |err| well below the current object-prior sd 0.345, and 90 % coverage
of the stated interval between 85 and 95 %. Train photos only; test photos are not read.

Usage: python tools/stereo_eval.py [--n 160] [--budget 540]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, labeled_dataset  # noqa: E402
import overlap as O  # noqa: E402
import overlap_eval as E  # noqa: E402

STRIDE = 0.75
STATE = RESULTS / "stereo_eval_state.jsonl"


def stereo_views(img, z_med, B, f_px, rng, frac=0.7):
    """(view_ref, view_other, Z on the ref grid). The ref camera stands B m to the right of the other."""
    a = np.asarray(img).astype(np.float32)
    H0, W0 = a.shape[:2]
    ys, xs = np.mgrid[0:H0, 0:W0].astype(np.float32)
    z = 1.0 + 1.2 * (1 - ys / H0)
    bump = cv2.GaussianBlur(rng.normal(0, 1, (H0, W0)).astype(np.float32), (0, 0), max(H0, W0) / 10)
    z = z * (1 + 0.25 * bump / (bump.std() + 1e-6))
    z = np.clip(z, 0.3, None)
    Z = z * (z_med / np.median(z))
    disp = f_px * B / Z
    w = int(W0 * frac)
    x0 = int((W0 - w - disp.max()) / 2)
    if x0 < 0:
        return None
    other = a[:, x0:x0 + w]
    mapx = (xs[:, :w] + x0 + disp[:, x0:x0 + w]).astype(np.float32)
    ref = cv2.remap(a, mapx, ys[:, :w], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (Image.fromarray(E.degrade(ref.astype(np.uint8))), Image.fromarray(E.degrade(other.astype(np.uint8))),
            Z[:, x0:x0 + w])


def one(iid, k):
    E.seed(f"stereo|{iid}|{k}")
    rng = E.rng
    img = E.load(iid, 1600)
    W, H = img.size
    f35 = float(rng.uniform(24, 28))
    f_true = f35 / 36 * max(W, H) * 0.7          # crop keeps 70 % of the width -> same focal in px
    z_med = float(rng.choice([20, 50, 100, 200]))
    steps = int(rng.choice([2, 4, 8]))
    B = steps * STRIDE
    sv = stereo_views(img, z_med, B, f35 / 36 * max(W, H), rng)
    if sv is None:
        return {"ok": False, "why": "disparity too large"}
    ref, oth, Z = sv
    fsd = float(rng.choice([0.02, 0.25]))
    f_ass = f35 / 36 * max(W, H) * math.exp(rng.normal(0, fsd))
    B_ass = B * math.exp(rng.normal(0, 0.15))
    st = O.stereo_points(ref, oth, f_ass, B_ass)
    out = {"ok": st is not None, "z_med": z_med, "steps": steps, "fsd": fsd, "bz": B / z_med}
    if st is None:
        return out
    xy = np.array(st["xy"]); z = np.array(st["z"])
    zt = Z[xy[:, 1].astype(int).clip(0, Z.shape[0] - 1), xy[:, 0].astype(int).clip(0, Z.shape[1] - 1)]
    lr = np.log(z) - np.log(zt)
    med = float(np.median(lr))
    mad = 1.4826 * float(np.median(np.abs(lr - med)))
    err = med - math.log(f_ass / (f35 / 36 * max(W, H)))          # what a size computed from it would carry
    sd = math.sqrt(0.15 ** 2 + (mad / math.sqrt(min(len(lr), 10))) ** 2 + (0.3 * fsd) ** 2)
    out.update(n=int(st["n"]), angle=st["angle_med"], err=err, sd=sd, mad=mad, f_true=f_true)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--budget", type=float, default=540)
    a = ap.parse_args()
    t0 = time.time()
    df = labeled_dataset()
    ids = list(df[df.split == "train"].sample(a.n, random_state=1).image_id)
    done = {}
    if STATE.exists():
        for l in STATE.read_text(encoding="utf-8").splitlines():
            r = json.loads(l); done[r["key"]] = r["val"]
    with open(STATE, "a", encoding="utf-8") as fh:
        for k, iid in enumerate(ids):
            key = f"{iid}|{k}"
            if key in done:
                continue
            if time.time() - t0 > a.budget:
                print(f"PARTIAL {len(done)}/{len(ids)} - run again"); return
            done[key] = one(iid, k)
            fh.write(json.dumps({"key": key, "val": done[key]}) + "\n"); fh.flush()
    rows = [done[f"{iid}|{k}"] for k, iid in enumerate(ids)]
    ok = [r for r in rows if r.get("ok") and "err" in r]
    err = np.array([r["err"] for r in ok]); sd = np.array([r["sd"] for r in ok])
    cover = float(np.mean(np.abs(err) <= 1.645 * sd))
    lines = ["# Walk-along metric scale: simulated stereo pairs with known depth", "",
             "Made by `tools/stereo_eval.py` (simulation from train photos; field errors simulated: walked distance "
             "log-sd 0.15, focal log-sd 0.02 or 0.25).", "",
             f"- Pairs: {len(rows)}, scale found: {len(ok)} ({len(ok) / max(len(rows), 1):.0%})",
             f"- Scale error (log, what sizes carry): median |err| **{np.median(np.abs(err)):.3f}**, "
             f"RMS {np.sqrt(np.mean(err ** 2)):.3f}, 90th pct |err| {np.percentile(np.abs(err), 90):.3f}",
             f"- Stated sd: median **{np.median(sd):.3f}** (current object prior: 0.345)",
             f"- 90 % interval coverage: **{cover:.1%}** (target 85-95 %)", "",
             "| distance | steps | B/Z | pairs | found | median abs err | coverage |", "|---|---|---|---|---|---|---|"]
    for zm in (20, 50, 100, 200):
        for s in (2, 4, 8):
            sel = [r for r in rows if r.get("z_med") == zm and r.get("steps") == s]
            if not sel:
                continue
            g = [r for r in sel if "err" in r]
            if g:
                e = np.array([r["err"] for r in g]); d = np.array([r["sd"] for r in g])
                lines.append(f"| {zm} m | {s} | 1/{zm / (s * STRIDE):.0f} | {len(sel)} | {len(g)} | {np.median(np.abs(e)):.3f} | {np.mean(np.abs(e) <= 1.645 * d):.0%} |")
            else:
                lines.append(f"| {zm} m | {s} | 1/{zm / (s * STRIDE):.0f} | {len(sel)} | 0 | - | - |")
    (RESULTS / "stereo_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("DONE")


if __name__ == "__main__":
    main()
