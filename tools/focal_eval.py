"""Measure how well a turning set fixes the focal length (overlap.analyse, "joined photos" source).

Simulation (labelled as such): a photo is viewed by a camera turning on the spot with a random true focal
(35 mm-equivalent 22-30 mm on the source photo), small pitch/roll jitter, residual lens distortion k1, noise and
JPEG re-compression (tools/overlap_eval.py helpers). No EXIF is given, so the focal comes from the homographies.
err = log(focal_est / focal_true) of the views. The log-sd fitted on TRAIN photos (RMS, so bias is included)
becomes `focal_sd` for joined pictures (src/focal_check.json); VAL photos check the 90 % coverage of that sd.
Test photos are not read. Gate (OVERNIGHT_PLAN step 3): sd < 0.08 for sets of >= 3 photos.

Usage: python tools/focal_eval.py [--n-train 120] [--n-val 120] [--budget 540]
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

STATE = RESULTS / "focal_eval_state.jsonl"
OUT = Path(__file__).resolve().parent.parent / "src" / "focal_check.json"


def views_with_focal(img, f35_src, yaws, frac=0.55):
    a = np.asarray(img)
    H0, W0 = a.shape[:2]
    f = f35_src / 36 * max(W0, H0)
    w, h = int(W0 * frac), int(H0 * 0.8)
    Ks, Ko = E.K(f, W0, H0), E.K(f, w, h)
    k1 = E.rng.uniform(-0.08, 0.04)
    pitch = E.rng.normal(0, 4)
    out = []
    for y in yaws:
        R = E.rot(y, pitch + E.rng.normal(0, 0.7), E.rng.normal(0, 1.0))
        v = cv2.warpPerspective(a, Ks @ R @ np.linalg.inv(Ko), (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        out.append(Image.fromarray(E.degrade(E.distort(v, k1))))
    return out, f / max(w, h) * 36


def one(iid, k):
    E.seed(f"focal|{iid}|{k}")
    img = E.load(iid, 1600)
    W0, H0 = img.size
    f35 = float(E.rng.uniform(22, 30))
    f = f35 / 36 * max(W0, H0)
    ym = math.degrees(math.atan(W0 / 2 / f) - math.atan(W0 * 0.55 / 2 / f)) * 0.9   # stay inside the photo
    n = int(E.rng.choice([2, 3]))
    s = float(E.rng.uniform(0.5, 1.0)) * ym
    yaws = [-s / 2, s / 2] if n == 2 else [-s, 0, s]
    views, f_true = views_with_focal(img, f35, yaws)
    try:
        rep, _ = O.analyse(views, "turn", None, stitch=False)
    except Exception as e:  # noqa: BLE001 - a failed analysis is a result too
        return {"n": n, "ok": False, "why": str(e)[:80]}
    ok = rep.get("focal_source") == "joined photos"
    r = {"n": n, "ok": ok, "src": rep.get("focal_source"), "span": s}
    if ok:
        r["err"] = math.log(rep["focal_35mm"] / f_true)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--budget", type=float, default=540)
    a = ap.parse_args()
    t0 = time.time()
    df = labeled_dataset()
    items = [("train", i) for i in df[df.split == "train"].sample(a.n_train, random_state=3).image_id] + \
            [("val", i) for i in df[df.split == "val"].sample(a.n_val, random_state=3).image_id]
    done = {}
    if STATE.exists():
        for l in STATE.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done[r["key"]] = r["val"]
    with open(STATE, "a", encoding="utf-8") as fh:
        for k, (sp, iid) in enumerate(items):
            key = f"{sp}|{iid}|{k}"
            if key in done:
                continue
            if time.time() - t0 > a.budget:
                print(f"PARTIAL {len(done)}/{len(items)} - run again")
                return
            done[key] = one(iid, k)
            fh.write(json.dumps({"key": key, "val": done[key]}) + "\n")
            fh.flush()
    rows = {sp: [done[f"{s}|{i}|{k}"] for k, (s, i) in enumerate(items) if s == sp] for sp in ("train", "val")}

    def stats(rs):
        e = np.array([r["err"] for r in rs if r.get("ok")])
        if not len(e):
            return None
        return {"sets": len(rs), "focal_found": len(e) / len(rs), "bias": float(np.median(e)),
                "rms": float(np.sqrt(np.mean(e ** 2))), "mad_sd": float(1.4826 * np.median(np.abs(e - np.median(e)))),
                "p90_abs": float(np.percentile(np.abs(e), 90))}
    res = {sp: {n: stats([r for r in rows[sp] if r["n"] == n]) for n in (2, 3)} for sp in rows}
    sd = {str(n): max(0.02, round(res["train"][n]["rms"], 3)) if res["train"][n] else 0.25 for n in (2, 3)}
    cover = {}
    for n in (2, 3):
        e = np.array([r["err"] for r in rows["val"] if r["n"] == n and r.get("ok")])
        cover[str(n)] = float(np.mean(np.abs(e) <= 1.645 * sd[str(n)])) if len(e) else None
    OUT.write_text(json.dumps({"focal_sd": sd, "val_coverage_90": cover, "n_train": a.n_train, "n_val": a.n_val,
                               "split": "train fits sd, val checks coverage; simulated turning sets",
                               "made_by": "tools/focal_eval.py"}, indent=1) + "\n", encoding="utf-8")
    lines = ["# Focal length from turning sets (simulation from real photos)", "",
             "Made by `tools/focal_eval.py`; true focal 22-30 mm (35 mm eq.), lens distortion, noise, JPEG; no EXIF.", "",
             "| split | photos in set | sets | focal found | bias (log) | RMS | robust sd | 90th pct abs |",
             "|---|---|---|---|---|---|---|---|"]
    for sp in ("train", "val"):
        for n in (2, 3):
            s = res[sp][n]
            if s:
                lines.append(f"| {sp} | {n} | {s['sets']} | {s['focal_found']:.0%} | {s['bias']:+.3f} | {s['rms']:.3f} | "
                             f"{s['mad_sd']:.3f} | {s['p90_abs']:.3f} |")
    lines += ["", f"focal_sd (train RMS): {sd}; val 90 % coverage: "
              + ", ".join(f"{n} photos {c:.1%}" for n, c in cover.items() if c is not None),
              f"Gate (sd < 0.08 for 3-photo sets): {'PASS' if sd['3'] < 0.08 else 'FAIL'}"]
    (RESULTS / "focal_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("DONE")


if __name__ == "__main__":
    main()
