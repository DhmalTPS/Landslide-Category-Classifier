"""Evaluate src/dimension.py without ground-truth sizes (none exist for these photos).

What can be checked from photos alone:
  1. availability   - how often a landslide outline is found, per class
  2. stability      - re-measure each photo under phone-like changes (JPEG, tone/brightness, noise,
                      downscale, small tilt, mirror). A trustworthy measurement barely moves: median |log ratio|
                      per measure (0.10 = about 10 % change)
  3. depth checks   - Depth Anything vs Depth Pro agreement, mirror stability, haze and grain-invariance
                      cues, and the trust verdict
  4. plausibility   - camera distance 2-300 m, grain D50 below the deposit width, slope 5-85 degrees
  5. speed + device - seconds per photo, and whether CUDA was used
Measured as the photo's labelled class (so classifier errors do not mix in). Nothing is tuned on the result.

  python tools/dimension_eval.py --per-class 10
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from common import IMAGES, RESULTS, labeled_dataset
from dimension import PROMPTS, Dimensioner, focal_35mm
from predict import load_image


def jpeg(img, q=35):
    import io
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


PERTURB = {
    "jpeg_q35": jpeg,
    "darker_tone": lambda im: ImageEnhance.Brightness(ImageEnhance.Contrast(im).enhance(0.8)).enhance(0.75),
    "brighter_warm": lambda im: Image.merge("RGB", [c.point(lambda p, k=k: min(255, int(p * k)))
                                                    for c, k in zip(ImageEnhance.Brightness(im).enhance(1.25).split(),
                                                                    (1.08, 1.0, 0.9))]),
    "noise": lambda im: Image.fromarray(np.clip(np.asarray(im, np.float32)
                                                + np.random.default_rng(0).normal(0, 8, (im.height, im.width, 3)),
                                                0, 255).astype(np.uint8)),
    "downscale_640": lambda im: im.resize((640, round(640 * im.height / im.width))) if im.width >= im.height
    else im.resize((round(640 * im.width / im.height), 640)),
    "tilt_3deg": lambda im: im.rotate(3, resample=Image.BICUBIC, expand=False).crop(
        (int(im.width * .03), int(im.height * .03), int(im.width * .97), int(im.height * .97))),
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(1.2)),
    "mirror": ImageOps.mirror,
}
KEYS = ["length", "width", "area", "volume", "drop"]
SUFFIX = ""          # set by --tag; results/dimension_eval{SUFFIX}.md


def flat(r):
    if not r.get("available"):
        return {}
    o = {m["key"]: m["value"] for m in r["measures"]}
    if "width" not in o and "median_width" in o:
        o["width"] = o["median_width"]
    for m in r["measures"]:      # relative 90 % interval width: log(P95 / P5)
        if m["range"][0] > 0:
            o[f"ci_{m['key']}"] = float(np.log(m["range"][1] / m["range"][0]))
    chk = {k: v for k, v in r["depth_checks"].items() if isinstance(v, (int, float)) or v is None}
    ratios = r["depth_checks"].get("model_ratios")
    if isinstance(ratios, dict) and ratios:
        chk["max_model_ratio"] = float(np.exp(max(abs(np.log(v)) for v in ratios.values())))
    elif "model_ratio_pro_over_da" in r["depth_checks"]:
        chk["max_model_ratio"] = float(np.exp(abs(np.log(r["depth_checks"]["model_ratio_pro_over_da"]))))
    o.update(outline=r.get("outline"), grain_method=r.get("grain_method"), scale_source=r["scale_source"],
             object_scale="in the photo" in r["scale_source"], grains_reliable=(r.get("grains") or {}).get("reliable"),
             distance=r["distance_m"], slope=r.get("slope_angle_deg"), trust=r["depth_trust"],
             prefailure=bool((r.get("prefailure_surface") or {}).get("available")),
             D50=(r.get("grains") or {}).get("D50_m"), **{f"chk_{k}": v for k, v in chk.items()})
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sites", type=int, default=12, help="sites with >= 2 photos for the same-site consistency test")
    ap.add_argument("--budget-min", type=float, default=0,
                    help="stop cleanly (exit code 3) after the current photo/site once this many minutes have passed; "
                         "rerun the same command to continue (for foreground chunks on low-RAM machines)")
    ap.add_argument("--tag", default="", help="suffix for the output files (keeps an earlier run for comparison)")
    args = ap.parse_args()
    t_start = time.time()

    def over_budget():
        if args.budget_min and (time.time() - t_start) / 60 > args.budget_min:
            print("PAUSED: time budget reached; rerun the same command to continue", flush=True)
            sys.exit(3)
    global SUFFIX
    SUFFIX = f"_{args.tag}" if args.tag else ""
    df = labeled_dataset()
    df = df[df.label.isin(list(PROMPTS))]
    pick = pd.concat(g.sample(min(len(g), args.per_class), random_state=args.seed) for _, g in df.groupby("label"))
    dim = Dimensioner()
    print(f"device: {dim.device} ({torch.cuda.get_device_name(0) if dim.device == 'cuda' else 'CPU'}), "
          f"{len(pick)} photos x {1 + len(PERTURB)} versions")
    # resumable: every finished photo is appended to a checkpoint file; a restart skips photos already done
    import json
    ck = RESULTS / f"dimension_eval{SUFFIX}.partial.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["row"]["image_id"]] = d
        print(f"resuming: {len(done)} photos already measured", flush=True)
    rows, stab, times = [], [], []
    for d in done.values():
        rows.append(d["row"])
        stab += d["stab"]
        times += d["time"]
    for i, r in enumerate(pick.itertuples(), 1):
        if r.image_id in done:
            continue
        n_stab = len(stab)
        path = next(IMAGES.glob(f"{r.image_id}.*"))
        img, f35 = load_image(path), focal_35mm(path)
        t0 = time.time()
        base = flat(dim(img, r.label, f35))
        times.append(time.time() - t0)
        rows.append({"image_id": r.image_id, "label": r.label, "available": bool(base), **base})
        if base:
            for name, fn in PERTURB.items():
                p = flat(dim(fn(img), r.label, f35))
                for k in KEYS:
                    if k in base and base[k] and p.get(k):
                        stab.append({"label": r.label, "perturb": name, "measure": k,
                                     "abs_log_ratio": abs(np.log(p[k] / base[k]))})
                    elif k in base and not p:
                        stab.append({"label": r.label, "perturb": name, "measure": k, "abs_log_ratio": np.nan})
        with open(ck, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"row": rows[-1], "stab": stab[n_stab:], "time": [times[-1]]}, default=float) + "\n")
        print(f"  {i}/{len(pick)} {r.label:20s} {'ok' if base else 'no outline'} ({times[-1]:.1f}s)", flush=True)
        over_budget()
        torch.cuda.empty_cache()

    # same-site consistency: different photos of one landslide must give similar sizes (no labels needed)
    multi = df.groupby("site_id").filter(lambda g: len(g) >= 2 and g.label.nunique() == 1)
    site_rows = []
    sck = RESULTS / f"dimension_eval{SUFFIX}.sites.jsonl"     # resumable, like the photos
    done_sites = {}
    if sck.exists():
        for line in sck.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done_sites[d["site_id"]] = d["rows"]
    for sid in list(multi.site_id.drop_duplicates().sample(frac=1, random_state=args.seed))[: args.sites]:
        if sid in done_sites:
            site_rows += done_sites[sid]
            continue
        n_before = len(site_rows)
        g = multi[multi.site_id == sid].head(4)
        vals = []
        for r in g.itertuples():
            path = next(IMAGES.glob(f"{r.image_id}.*"))
            v = flat(dim(load_image(path), r.label, focal_35mm(path)))
            if v:
                vals.append(v)
        if len(vals) >= 2:
            for k in ("length", "width", "area", "volume", "distance"):
                xs = [v[k] for v in vals if v.get(k)]
                if len(xs) >= 2:
                    site_rows.append({"site_id": sid, "label": g.label.iloc[0], "measure": k, "n": len(xs),
                                      "log_sd": float(np.std(np.log(xs), ddof=1))})
        with open(sck, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"site_id": sid, "rows": site_rows[n_before:]}, default=float) + "\n")
        print(f"  site {sid}: {len(vals)} photos measured", flush=True)
        over_budget()
    sites = pd.DataFrame(site_rows)
    res, st = pd.DataFrame(rows), pd.DataFrame(stab)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / f"dimension_eval{SUFFIX}.csv", index=False)
    st.to_csv(RESULTS / f"dimension_stability{SUFFIX}.csv", index=False)
    ok = res[res.available]
    lines = ["# Dimension estimation - evaluation", "",
             f"Device: **{dim.device}**, mean {np.mean(times):.1f} s per photo (all models, one photo). "
             f"{len(res)} photos, {args.per_class} per class, seed {args.seed}, measured as the labelled class.",
             "No ground-truth sizes exist, so this checks availability, stability, depth consistency and "
             "plausibility, not absolute accuracy.", "",
             "## 1. Availability (outline found)", "", res.groupby("label").available.mean().map("{:.0%}".format)
             .to_frame("available").to_markdown(), "",
             "## 2. Stability under phone-like changes (median |log ratio|; 0.10 = ~10 % change)", "",
             "Lost = outline not found after the change.", ""]
    if len(st):
        tab = st.pivot_table(index="perturb", columns="measure", values="abs_log_ratio", aggfunc="median").round(3)
        lost = st.groupby("perturb").abs_log_ratio.apply(lambda s: s.isna().mean()).map("{:.0%}".format)
        lines += [tab.assign(lost=lost).to_markdown(), "",
                  st.pivot_table(index="label", columns="measure", values="abs_log_ratio", aggfunc="median")
                  .round(3).to_markdown(), ""]
    if len(ok):
        chk = [c for c in ok.columns if c.startswith("chk_")]
        lines += ["## 3. Depth checks", "",
                  f"Trust verdict: {ok.trust.value_counts(normalize=True).map('{:.0%}'.format).to_dict()}", "",
                  ok[chk].describe().round(3).to_markdown(), "",
                  "Reading: model_ratio near 1 = Depth Pro and Depth Anything agree; flip_stability near 0 = "
                  "mirror-invariant; haze_rho < 0 = contrast falls with distance as it should; "
                  "grain slope near 0 = grain size independent of distance.", "",
                  f"Scale set by a person/car in the photo: {ok.object_scale.mean():.0%}. "
                  f"Grain sizes judged reliable (no size-distance trend): {ok.grains_reliable.mean():.0%}. "
                  f"Outline used SAM: {ok.outline.fillna('').str.contains('SAM').mean():.0%}. "
                  f"Outline used the trained head: {ok.outline.fillna('').str.contains('head').mean():.0%}.", "",
                  "Photos scaled by a person/car (check these by eye): "
                  + (", ".join(f"{a} ({b})" for a, b in ok[ok.object_scale][["image_id", "scale_source"]].values) or "none"), "",
                  "## 4. Plausibility", ""]
        checks = {"distance 2-300 m": ok.distance.between(2, 300),
                  "slope 5-85 deg": ok.slope.between(5, 85),
                  "D50 < width (when grains measured)": (ok.D50 < ok.width)[ok.D50.notna()]}
        lines += [f"- {k}: {v.mean():.0%} of {len(v)}" for k, v in checks.items()] + [""]
        ci = [c for c in ok.columns if c.startswith("ci_")]
        if ci:
            lines += ["## 4b. Uncertainty: median relative 90 % interval width, log(P95/P5); 0.7 = about x2 end to end", "",
                      ok.groupby("label")[ci].median().round(2).to_markdown(), ""]
        lines += [f"Pre-failure surface volume available: {ok.prefailure.mean():.0%}.", ""]
        if len(sites):
            lines += ["## 4c. Same-site consistency (different photos of one landslide)", "",
                      "Median log-sd across photos of the same site (0.10 = about 10 % spread). Viewpoints differ, so "
                      "some spread is real (different visible extent).", "",
                      sites.groupby("measure").log_sd.agg(["median", "count"]).round(3).to_markdown(), ""]
        lines += ["## 5. Median measures per class", "",
                  ok.groupby("label")[[k for k in KEYS + ["distance", "slope", "D50"] if k in ok]].median()
                  .round(2).to_markdown(), ""]
    if len(ok):
        # object evidence on the depth-ensemble scale: only photos whose scale came from tight-height objects ALONE
        import json, re
        from dimension import OBJECT_PRIOR
        tight = {"person", "car", "motorcycle", "bus", "door"}
        logs = []
        for src in ok.scale_source.dropna():
            m = re.match(r"(.+) in the photo \(x([0-9.]+)\)", src)
            if m and set(m.group(1).split("/")) <= tight:
                logs.append(np.log(float(m.group(2))))
        if logs:
            lg = np.array(logs)
            sd = float(np.std(lg, ddof=1)) if len(lg) > 1 else 0.5
            prior = {"factor": round(float(np.exp(np.median(lg))), 3), "n": len(lg),
                     "log_sd_between_photos": round(sd, 3),
                     "predictive_log_sd": round(float(np.sqrt(sd ** 2 * (1 + 1 / len(lg)) + 0.1 ** 2)), 3),
                     "factors": [round(float(np.exp(x)), 3) for x in lg], "source": f"dimension_eval{SUFFIX}",
                     "note": "factors relative to the UNCORRECTED depth ensemble; only when no prior was applied"}
            lines += [f"Known-size object evidence (tight heights only): factors {prior['factors']}, "
                      f"median x{prior['factor']} (n={prior['n']})", ""]
            if not any("corrected x" in s_ for s_ in ok.scale_source.dropna()) and len(lg) >= 4:
                OBJECT_PRIOR.write_text(json.dumps(prior, indent=1))
                lines += [f"Wrote {OBJECT_PRIOR.name}.", ""]
    (RESULTS / f"dimension_eval{SUFFIX}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / f'dimension_eval{SUFFIX}.md'}")


if __name__ == "__main__":
    main()
