"""Data-driven thresholds for the app's instant on-phone photo check (no server round trip).

The phone computes the same six numbers as `metrics()` below (src/guide.py, function `metrics`) on the
photo scaled to 512 px on the long side:
  lum      mean brightness (0-255)
  dark     fraction of pixels < 16      (crushed shadows)
  bright   fraction of pixels > 239     (blown highlights)
  contrast standard deviation of brightness
  sharp    90th percentile of the per-tile Laplacian variance on an 8x8 grid
           (asks "is the sharpest part of the photo sharp?", so a smooth sky does not count as blur)
  sky      fraction of smooth (|Laplacian| < 6) pixels that are blue (B > R+8, B >= G, grey > 100) or very
           bright (grey > 220): "mostly sky" framing (high side only)
A photo is flagged when a number lies outside the range of the accepted TRAINING photos: below the
q-th percentile (lum, contrast, sharp) or above the (100-q)-th (lum, dark, bright, sky). q is the largest value in
Q_GRID whose combined flag rate on train photos is at most TARGET (5%): one nag in 20 good photos at most. The model has seldom seen
such photos, so its answer is less reliable. The check only warns; the user can still send the photo.
Calibrated on the train split only; the val split reports how often good photos get flagged; test is never read.

  python tools/calibrate_photo_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from common import IMAGES, labeled_dataset

OUT = Path(__file__).resolve().parent.parent / "src" / "photo_check.json"
Q_GRID, TARGET = (2, 1.5, 1, 0.75, 0.5, 0.25), 0.05


def metrics(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    s = 512 / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    a = np.asarray(img, np.float32)
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    h, w = g.shape
    tiles = [lap[1 + i * (h - 2) // 8: 1 + (i + 1) * (h - 2) // 8, 1 + j * (w - 2) // 8: 1 + (j + 1) * (w - 2) // 8].var()
             for i in range(8) for j in range(8)]
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    sky = (np.abs(lap) < 6) & (((B > R + 8) & (B >= G) & (g > 100)) | (g > 220))
    return {"sky": float(sky[1:-1, 1:-1].mean()), "lum": float(g.mean()), "dark": float((g < 16).mean()), "bright": float((g > 239).mean()),
            "contrast": float(g.std()), "sharp": float(np.percentile(tiles, 90))}


# phone-like failures applied to val photos: how many does the check catch? (sensitivity)
DEGRADE = {"blur (shake / focus miss)": lambda im: im.filter(ImageFilter.GaussianBlur(max(im.size) / 512 * 2.5)),
           "too dark (dusk, x0.35)": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
           "overexposed (x2.2)": lambda im: ImageEnhance.Brightness(im).enhance(2.2),
           "haze / fog (low contrast)": lambda im: ImageEnhance.Contrast(im).enhance(0.35),
           # top 60% replaced by a smooth sky-blue gradient: a photo aimed too high
           "mostly sky (aimed too high)": lambda im: sky_top(im)}


def sky_top(im, frac=0.6):
    a = np.asarray(im, np.float32).copy()
    h = int(a.shape[0] * frac)
    t = np.linspace(0, 1, h)[:, None]
    a[:h] = np.stack([150 + 40 * t, 190 + 25 * t, 235 + 10 * t], -1)
    return Image.fromarray(a.astype(np.uint8))


def table(rows, degrade=None):
    out = {}
    for r in rows.itertuples():
        try:
            im = ImageOps.exif_transpose(Image.open(next(IMAGES.glob(f"{r.image_id}.*")))).convert("RGB")
            m = metrics(degrade(im) if degrade else im)
        except (StopIteration, OSError):
            continue
        for k, v in m.items():
            out.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def main():
    df = labeled_dataset()
    tr, va = table(df[df.split == "train"]), table(df[df.split == "val"])
    def thresholds(q):
        th = {"lum_min": np.percentile(tr["lum"], q), "lum_max": np.percentile(tr["lum"], 100 - q),
              "dark_max": np.percentile(tr["dark"], 100 - q), "bright_max": np.percentile(tr["bright"], 100 - q),
              "contrast_min": np.percentile(tr["contrast"], q), "sharp_min": np.percentile(tr["sharp"], q),
              "sky_max": np.percentile(tr["sky"], 100 - q)}
        return {k: round(float(v), 4) for k, v in th.items()}
    flag = lambda m, th: ((m["lum"] < th["lum_min"]) | (m["lum"] > th["lum_max"]) | (m["dark"] > th["dark_max"]) |  # noqa: E731
                      (m["bright"] > th["bright_max"]) | (m["contrast"] < th["contrast_min"]) | (m["sharp"] < th["sharp_min"]) | (m["sky"] > th["sky_max"]))
    for q in Q_GRID:
        th = thresholds(q)
        print(f"  q={q}: train flag rate {flag(tr, th).mean():.3f}")
        if flag(tr, th).mean() <= TARGET:
            break
    res = {"thresholds": th, "n_train": int(len(tr["lum"])), "n_val": int(len(va["lum"])),
           "train_flag_rate": round(float(flag(tr, th).mean()), 3), "val_flag_rate": round(float(flag(va, th).mean()), 3),
           "percentile_q": q, "resize_long_side": 512,
           "val_detection_rate": {k: round(float(flag(table(df[df.split == "val"], f), th).mean()), 3)
                                  for k, f in DEGRADE.items()}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
