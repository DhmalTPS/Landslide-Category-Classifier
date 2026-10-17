"""Scenario A: import a labeled dataset that was given to us.

Supported layouts:
  folders:  <root>/<label>/*.jpg                     (labels from folder names)
            <root>/<train|val|test>/<label>/*.jpg    (dataset's own split is kept)
  csv:      --csv labels.csv --image-col file --label-col type [--group-col site] [--split-col split]

Labels are mapped onto our classes (common.CLASSES). If a label isn't a class name, it must
be in the --map JSON. When labels are unmapped, a template with keyword-based guesses is
written to data/label_map_<name>.json; check it, edit it, and pass it with --map.

Imported labels go to data/labels/<name>.csv (counted like a reviewer); images and metadata
are added the same way as other sources. Use a group/site column if the dataset has one:
otherwise every image is its own site and near-identical photos may straddle the splits.

Usage:
  python src/import_dataset.py D:/given_data --name given
  python src/import_dataset.py D:/given_data --name given --map data/label_map_given.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from collect_commons import dhash
from common import (CLASS_NAMES, DATA, IMAGES, LABELS_DIR, METADATA_CSV, REJECT, SPLITS_CSV, merge_metadata,
                    read_csv, write_csv)

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}
# keyword -> class, checked in order (first match wins)
GUESSES = [
    (("non", "no_", "no ", "normal", "stable", "negative", "background", "none"), "no_landslide"),
    (("debris", "lahar", "torrent"), "debris_flow"),
    (("mud", "earthflow", "earth_flow", "earth flow", "flow"), "earth_mud_flow"),
    (("rotation", "slump"), "rotational_slide"),
    (("fall", "topple", "rockfall"), "fall_topple"),
    (("translation", "planar", "wedge", "slide", "block"), "translational_slide"),
]


def guess(label):
    low = label.lower()
    for keys, cls in GUESSES:
        if any(k in low for k in keys):
            return cls
    return ""


def scan_folders(root):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        parts = p.relative_to(root).parts[:-1]
        split = SPLIT_NAMES.get(parts[0].lower()) if parts else None
        rest = parts[1:] if split else parts
        if not rest:
            raise SystemExit(f"cannot find a label folder for {p}")
        rows.append({"path": p, "label": rest[0], "group": "/".join(rest[1:]), "split": split or ""})
    return rows


def scan_csv(root, args):
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    rows = []
    for r in df.to_dict("records"):   # keeps column names with spaces (itertuples would rename them)
        split = SPLIT_NAMES.get(r[args.split_col].lower(), "") if args.split_col else ""
        rows.append({"path": root / r[args.image_col], "label": r[args.label_col],
                     "group": r[args.group_col] if args.group_col else "", "split": split})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--name", required=True, help="short dataset name, e.g. given")
    ap.add_argument("--map", help="JSON {their_label: our_class or 'reject'}")
    ap.add_argument("--csv")
    ap.add_argument("--image-col", default="file")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col")
    ap.add_argument("--split-col")
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    root = Path(args.root)
    items = scan_csv(root, args) if args.csv else scan_folders(root)
    theirs = sorted({i["label"] for i in items})
    mapping = {l: l for l in theirs if l in CLASS_NAMES}
    if args.map:
        mapping |= json.loads(Path(args.map).read_text(encoding="utf-8"))
    unmapped = [l for l in theirs if l not in mapping]
    if unmapped:
        tpl = DATA / f"label_map_{args.name}.json"
        tpl.parent.mkdir(parents=True, exist_ok=True)
        tpl.write_text(json.dumps({l: mapping.get(l) or guess(l) for l in theirs}, indent=2), encoding="utf-8")
        counts = pd.Series([i["label"] for i in items]).value_counts()
        print("labels without a mapping:\n" + counts[unmapped].to_string())
        raise SystemExit(f"\nwrote a guessed mapping to {tpl}. Check every line (valid values: "
                         f"{', '.join(CLASS_NAMES)}, {REJECT}), then rerun with --map {tpl}")
    bad = {v for v in mapping.values() if v not in CLASS_NAMES + [REJECT]}
    if bad:
        raise SystemExit(f"mapping has invalid targets: {bad}")

    IMAGES.mkdir(parents=True, exist_ok=True)
    meta = read_csv(METADATA_CSV)
    known = set(meta.title) if meta is not None else set()
    rows = []
    labels, split_rows = [], []
    for it in items:
        rel = it["path"].relative_to(root).as_posix() if it["path"].is_relative_to(root) else str(it["path"])
        title = f"{args.name}:{rel}"
        image_id = f"{args.name}_" + hashlib.sha1(title.encode()).hexdigest()[:12]
        site = f"{args.name}:" + (it["group"] or rel)
        if title not in known:
            img = ImageOps.exif_transpose(Image.open(it["path"])).convert("RGB")
            img.thumbnail((args.max_side, args.max_side))
            img.save(IMAGES / f"{image_id}.jpg", quality=90)
            rows.append({"image_id": image_id, "file": f"raw/images/{image_id}.jpg", "source": args.name,
                         "title": title, "url": "", "license": "provided dataset", "artist": "",
                         "description": it["label"], "root_category": "", "category": it["label"],
                         "site_id": site, "hint_label": "", "dhash": dhash(img)})
        labels.append({"image_id": image_id, "label": mapping[it["label"]], "reviewer": args.name,
                       "vlm_shown": "0", "time": ""})
        if it["split"]:
            split_rows.append({"image_id": image_id, "site_id": site, "split": it["split"]})

    merge_metadata(rows)
    write_csv(pd.DataFrame(labels), LABELS_DIR / f"{args.name}.csv")
    if split_rows:   # keep the dataset's own split; split.py never reassigns known sites
        old = read_csv(SPLITS_CSV)
        new = pd.DataFrame(split_rows)
        merged = new if old is None else pd.concat([old[~old.image_id.isin(new.image_id)], new])
        write_csv(merged, SPLITS_CSV)
        print("kept the dataset's own split:", new.split.value_counts().to_dict())
    elif not any(i["group"] for i in items):
        print("warning: no site/group info; each image is its own site, so photos of the same "
              "landslide may land in both train and test")
    print(f"imported {len(labels)} images; label counts:",
          pd.Series([l["label"] for l in labels]).value_counts().to_dict())


if __name__ == "__main__":
    main()
