"""Experiment: refit the known-object scale prior (OVERNIGHT_PLAN step 4).

When a photo has no typed distance, no walk-along points and no known-size object of its own, sizes rely on
the depth ensemble corrected by a fixed factor learned from photos that DO show known-height objects
(models/dimension_object_prior.json, so far n = 4 photos, predictive log-sd 0.345).

Here: all usable train+val photos (test never read) are scanned with OWLv2. Photos with a tight-height object
(person, car, motorcycle, bus, door) go through the same depth ensemble and object_scale as the app. Each gives a
log factor l_i = log(true height / ensemble height) with its own measurement sd s_i.

Model: l_i = mu (+ b * log z_i) + u_i + e_i, where u_i ~ N(0, tau^2) between photos and e_i ~ N(0, s_i^2).
tau is estimated by DerSimonian-Laird, with mu weighted per site. Predictive sd for a new photo without objects
= sqrt(tau^2 + se(mu)^2). Leave-one-site-out CV gives the error and the coverage of a 90 % interval (the
held-out photo's own s_i is added, because the observed l_i contains it). The covariate (log of the median
scene depth) is kept only if it lowers the leave-site-out mean squared error.

Gate: leave-site-out predictive sd < 0.345 with coverage 85-95 %, else the old prior is kept.

Usage: python tools/object_prior_eval.py --budget 540   (repeat until DONE)
Output: results/object_prior_eval.md; models/dimension_object_prior.json if the gate passes (--write).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402,F401
from common import DATA, RESULTS, labeled_dataset  # noqa: E402

STATE = RESULTS / "object_prior_state.jsonl"
TIGHT = {"person", "car", "motorcycle", "bus", "door"}
OLD_SD = 0.345


def load_state():
    d = {}
    if STATE.exists():
        for l in STATE.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            d[r["id"]] = r
    return d


def tight_seen(dim, img):
    """Names of tight-height objects OWLv2 finds with the same query, threshold and box filters as
    dimension.object_scale (no depth needed)."""
    import torch
    from dimension import KNOWN_OBJECTS
    W, H = img.size
    inp = dim.owl_proc(text=[[t for t, _, _ in KNOWN_OBJECTS]], images=img, return_tensors="pt").to(dim.device)
    inp["pixel_values"] = inp["pixel_values"].to(dim.owl.dtype)
    with torch.inference_mode():
        out = dim.owl(**inp)
    Sd = max(W, H)
    res = dim.owl_proc.image_processor.post_process_object_detection(out, threshold=0.25, target_sizes=torch.tensor([[Sd, Sd]]))[0]
    names = set()
    for lb, bx in zip(res["labels"].cpu().numpy(), res["boxes"].float().cpu().numpy()):
        name = KNOWN_OBJECTS[int(lb)][0].replace("a photo of a ", "")
        x0, y0, x1, y1 = bx
        if name not in TIGHT or x0 < 3 or y0 < 3 or x1 > W - 3 or y1 > H - 3 or (y1 - y0) < 20:
            continue
        if name == "person" and (y1 - y0) < 1.3 * (x1 - x0):
            continue
        names.add(name)
    return names


def scan(df, budget):
    import torch
    from dimension import DEFAULT_F35, Dimensioner, focal_35mm
    from predict import load_image
    t0 = time.time()
    done = load_state()
    todo = [r for r in df.itertuples() if r.image_id not in done]
    if not todo:
        return True
    dim = Dimensioner()
    with open(STATE, "a", encoding="utf-8") as fh:
        for k, r in enumerate(todo):
            if time.time() - t0 > budget:
                print(f"PARTIAL {len(done)}/{len(df)} - run again")
                return False
            path = DATA / r.file
            img = load_image(path)
            W, H = img.size
            f35 = focal_35mm(str(path))
            f_exif = f35 / 36.0 * max(W, H) if f35 else None
            rec = {"id": r.image_id, "site": r.site_id, "label": r.label}
            try:
                # cheap pass first: does OWLv2 see a usable tight-height object at all?
                with dim.use(dim.owl):
                    seen = tight_seen(dim, img)
                rec["seen"] = sorted(seen)
                if seen:
                    with dim.use(dim.da, dim.pro):
                        depth, logvar, f_pro, per_model, _ = dim.depth(img, f_exif)
                    f_def = DEFAULT_F35 / 36.0 * max(W, H)
                    f = f_exif or (f_pro if f_pro and 0.5 * f_def < f_pro < 3 * f_def else f_def)
                    with dim.use(dim.owl):
                        obj = dim.object_scale(img, depth, f)
                    if obj and set(obj["objects"]) <= TIGHT:
                        rec.update(obj=obj["objects"], l=math.log(obj["factor"]), s=obj["log_sd"],
                                   z=float(np.median(depth)), exif=bool(f_exif))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("CUDA OOM - rest and run again")
                return False
            except Exception as e:  # noqa: BLE001 - record and move on
                rec["err"] = repr(e)[:100]
            done[r.image_id] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 50 == 0:
                print(f"  {len(done)}/{len(df)} ({time.time() - t0:.0f} s), with objects: "
                      f"{sum('l' in v for v in done.values())}", flush=True)
    return True


def fit(l, s, sites, X=None):
    """Random-effects fit with site-balanced weights. Returns (beta, tau, cov(beta))."""
    n = len(l)
    X = np.ones((n, 1)) if X is None else X
    cnt = {g: (sites == g).sum() for g in set(sites)}
    wsite = np.array([1.0 / cnt[g] for g in sites])

    def wls(w):
        Wm = w[:, None] * X
        A = X.T @ Wm
        beta = np.linalg.solve(A, Wm.T @ l)
        return beta, np.linalg.inv(A)
    w0 = wsite / s ** 2
    beta, _ = wls(w0)
    r = l - X @ beta
    Q = float(np.sum(w0 * r ** 2))
    c = float(np.sum(w0) - np.sum(w0 ** 2) / np.sum(w0))
    tau2 = max(0.0, (Q - (n - X.shape[1])) / c) if c > 0 else 0.0
    w = wsite / (s ** 2 + tau2)
    beta, cov = wls(w)
    # site-balanced weights shrink the effective n: scale the covariance by the design effect
    neff = w.sum() ** 2 / np.sum(w ** 2)
    cov = cov * (np.sum(w) / neff) if neff > 0 else cov
    return beta, math.sqrt(tau2), cov


def evaluate(rows, use_cov):
    l = np.array([r["l"] for r in rows]); s = np.array([r["s"] for r in rows])
    sites = np.array([r["site"] for r in rows]); z = np.log(np.array([r["z"] for r in rows]))
    zc = z - np.median(z)
    X = np.column_stack([np.ones(len(l)), zc]) if use_cov else None
    errs, cover, psd = [], [], []
    for g in sorted(set(sites)):
        te, tr = sites == g, sites != g
        beta, tau, cov = fit(l[tr], s[tr], sites[tr], None if X is None else X[tr])
        x = X[te] if X is not None else np.ones((te.sum(), 1))
        mu = x @ beta
        var_mu = np.einsum("ij,jk,ik->i", x, cov, x)
        sd_pred = np.sqrt(tau ** 2 + var_mu)                    # for a new photo without objects
        e = l[te] - mu
        errs += list(e); psd += list(sd_pred)
        cover += list(np.abs(e) <= 1.645 * np.sqrt(sd_pred ** 2 + s[te] ** 2))
    beta, tau, cov = fit(l, s, sites, X)
    return {"n": len(l), "sites": len(set(sites)), "beta": [float(b) for b in beta], "tau": tau,
            "sd_pred": float(math.sqrt(tau ** 2 + cov[0, 0])), "lso_mse": float(np.mean(np.square(errs))),
            "lso_med_abs": float(np.median(np.abs(errs))), "lso_cover": float(np.mean(cover)),
            "lso_sd_pred_med": float(np.median(psd)), "z_median": float(np.exp(np.median(z)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--write", action="store_true", help="write the prior if the gate passes")
    a = ap.parse_args()
    df = labeled_dataset()
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)      # test never read
    if not scan(df, a.budget):
        return
    rows = [r for r in load_state().values() if "l" in r]
    if len(rows) < 8:
        print(f"only {len(rows)} photos with tight objects - not enough"); print("DONE"); return
    base, cov = evaluate(rows, False), evaluate(rows, True)
    best = cov if cov["lso_mse"] < base["lso_mse"] else base
    # the old prior (x1.531, sd 0.345, fitted on 4 photos) on the same evidence
    l_ = np.array([r["l"] for r in rows]); s_ = np.array([r["s"] for r in rows]); e_ = l_ - math.log(1.531)
    old_cover = float(np.mean(np.abs(e_) <= 1.645 * np.sqrt(OLD_SD ** 2 + s_ ** 2)))
    # adopt when the new prior is calibrated AND either narrower or the old one under-covers (honest > narrow)
    gate = 0.85 <= best["lso_cover"] <= 0.95 and (best["lso_sd_pred_med"] < OLD_SD or old_cover < 0.85)
    lines = ["# Known-object scale prior refit (train+val photos, test not read)", "",
             f"Photos scanned: {len(load_state())}; with a tight-height object scale: {len(rows)} "
             f"({base['sites']} sites). Objects: " + ", ".join(sorted({o for r in rows for o in r['obj']})), "",
             "| model | factor (at median depth) | depth slope | tau | predictive sd | LSO median abs err | LSO MSE | LSO 90 % coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for nm, r in (("constant", base), ("log depth covariate", cov)):
        lines.append(f"| {nm} | x{math.exp(r['beta'][0]):.3f} | {r['beta'][1] if len(r['beta']) > 1 else 0:+.3f} | "
                     f"{r['tau']:.3f} | {r['lso_sd_pred_med']:.3f} | {r['lso_med_abs']:.3f} | {r['lso_mse']:.3f} | {r['lso_cover']:.1%} |")
    lines += ["", f"Old prior: x1.531, predictive sd {OLD_SD} (n = 4); on these {len(rows)} photos its 90 % interval "
              f"covers {old_cover:.1%} (median abs err {np.median(np.abs(e_)):.3f}).",
              f"Chosen: {'log depth covariate' if best is cov else 'constant'}; gate (coverage 85-95 % and narrower than "
              f"{OLD_SD} or the old prior under-covers): {'PASS' if gate else 'FAIL'}"]
    (RESULTS / "object_prior_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if gate and a.write:
        from dimension import OBJECT_PRIOR
        prior = {"factor": round(math.exp(best["beta"][0]), 3), "n": best["n"], "sites": best["sites"],
                 "tau": round(best["tau"], 3), "predictive_log_sd": round(best["lso_sd_pred_med"], 3),
                 "lso_coverage_90": round(best["lso_cover"], 3), "old_prior_coverage_90": round(old_cover, 3),
                 "source": "tools/object_prior_eval.py",
                 "note": "factor relative to the UNCORRECTED depth ensemble; train+val photos with person/car/"
                         "motorcycle/bus/door scales; sd and coverage from leave-one-site-out CV"}
        if best is cov:
            prior["log_depth_slope"] = round(best["beta"][1], 4)
            prior["depth_ref_m"] = round(best["z_median"], 2)
        OBJECT_PRIOR.write_text(json.dumps(prior, indent=1) + "\n")
        print(f"wrote {OBJECT_PRIOR}")
    print("DONE")


if __name__ == "__main__":
    main()
