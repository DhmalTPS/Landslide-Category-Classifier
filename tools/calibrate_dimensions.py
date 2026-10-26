"""Learn the dimension scale correction from field measurements, with leakage-free validation.

A single photo cannot know its own scale exactly, and monocular depth models are biased on landslide
scenes. With ground truth from the field (tape, laser rangefinder, GNSS, or Google Earth / DEM), this
fits one scale factor (plus per-class factors shrunk toward it). models/dimension_calibration.json is
then applied by src/dimension.py whenever no typed distance and no person/car scale is available.

Input: data/field/measurements.csv (a template is written if missing), one row per photo:
  image        path to the photo (relative to the project or absolute)
  site_id      the same landslide photographed several times shares one id
  label        landslide class (optional; the classifier is used if empty)
  length_m, width_m, distance_m   any of these that were measured (blank = not measured)

No leakage: the factor is a median of log(measured / estimated). It is validated by
LEAVE-ONE-SITE-OUT cross-validation: every site is predicted by a factor fitted only on the other
sites, so photos of one landslide never sit in both fit and validation. The reported CV error is
therefore what to expect on a new landslide. Fewer than 5 sites -> report only, no calibration written.

  python tools/calibrate_dimensions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd

from common import DATA, MODELS, RESULTS, ROOT
from dimension import CALIBRATION, Dimensioner, focal_35mm
from predict import load_image

FIELD = DATA / "field" / "measurements.csv"
KEYS = {"length_m": "length", "width_m": "width", "distance_m": "distance"}
SHRINK = 5          # per-class factor = weighted toward the global one until a class has >> 5 sites


def fit(df):
    """Global + per-class factors from log ratios (one median per site first, so big sites do not dominate)."""
    site = df.groupby(["site_id", "label"]).log_ratio.median().reset_index()
    g = float(site.log_ratio.median())
    per = {}
    for c, s in site.groupby("label"):
        w = len(s) / (len(s) + SHRINK)
        per[c] = float(np.exp(w * s.log_ratio.median() + (1 - w) * g))
    return float(np.exp(g)), per


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=FIELD)
    ap.add_argument("--dry-run", action="store_true", help="report only, never write the calibration")
    args = ap.parse_args()
    FIELD_ = args.csv
    if not FIELD_.exists() and FIELD_ != FIELD:
        raise SystemExit(f"{FIELD_} not found")
    if FIELD_ == FIELD and not FIELD.exists():
        FIELD.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["image", "site_id", "label", "length_m", "width_m", "distance_m"]).to_csv(FIELD, index=False)
        raise SystemExit(f"wrote an empty template {FIELD}; fill it with field measurements and run again")
    field = pd.read_csv(FIELD_)
    if field.empty:
        raise SystemExit(f"{FIELD} has no rows yet")
    dim = Dimensioner()
    dim.calib = {}                      # measure uncalibrated
    pred = None
    rows = []
    for r in field.itertuples():
        path = Path(r.image) if Path(r.image).is_absolute() else ROOT / r.image
        img = load_image(path)
        label = r.label if isinstance(r.label, str) and r.label else None
        if label is None:
            from predict import Predictor
            pred = pred or Predictor()
            label = pred([img])[0]["label"]
        res = dim(img, label, focal_35mm(path))
        if not res.get("available"):
            print(f"  {path.name}: no outline ({res.get('reason')})")
            continue
        est = {m["key"]: m["value"] for m in res["measures"]}
        est["distance"] = res["distance_m"]
        if res["scale_source"].startswith("a "):     # person/car scale: not the depth scale being calibrated
            continue
        for col, key in KEYS.items():
            v = getattr(r, col, np.nan)
            if pd.notna(v) and v > 0 and est.get(key):
                rows.append({"site_id": str(r.site_id), "label": label, "measure": key, "measured": float(v),
                             "estimated": float(est[key]), "log_ratio": float(np.log(v / est[key]))})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable measurement pairs")
    sites = df.site_id.unique()
    # leave-one-site-out CV
    cv = []
    for s in sites:
        tr, va = df[df.site_id != s], df[df.site_id == s]
        if tr.empty:
            continue
        g, per = fit(tr)
        for v in va.itertuples():
            f = per.get(v.label, g)
            cv.append({"site_id": s, "measure": v.measure,
                       "ape_raw": abs(v.estimated / v.measured - 1), "ape_cal": abs(v.estimated * f / v.measured - 1)})
    cv = pd.DataFrame(cv)
    g, per = fit(df)
    report = {"factor": round(g, 3), "per_class": {k: round(v, 3) for k, v in per.items()},
              "n_pairs": len(df), "n_sites": len(sites),
              "cv": "leave-one-site-out",
              "cv_median_abs_pct_error_uncalibrated": round(float(cv.ape_raw.median() * 100), 1) if len(cv) else None,
              "cv_median_abs_pct_error_calibrated": round(float(cv.ape_cal.median() * 100), 1) if len(cv) else None,
              "cv_by_measure": cv.groupby("measure")[["ape_raw", "ape_cal"]].median().mul(100).round(1).to_dict()
              if len(cv) else {}}
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "dimension_calibration_pairs.csv", index=False)
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    if len(sites) < 5:
        print(f"only {len(sites)} sites: not writing {CALIBRATION.name} (need >= 5 for a trustworthy factor)")
        return
    if report["cv_median_abs_pct_error_calibrated"] >= report["cv_median_abs_pct_error_uncalibrated"]:
        print("calibration does not beat the raw estimate in cross-validation: not writing it")
        return
    MODELS.mkdir(exist_ok=True)
    CALIBRATION.write_text(json.dumps(report, indent=2))
    print(f"wrote {CALIBRATION}")


if __name__ == "__main__":
    main()
