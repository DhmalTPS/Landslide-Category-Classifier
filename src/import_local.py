"""Import your own photos (field trips, phone) into the dataset.

Folder layout (one folder per landslide site, so all its photos stay in one split):
    field_photos/<site_name>/*.jpg
    field_photos/<class_name>/<site_name>/*.jpg      # optional: class known already -> used as hint

Images are resized so the long side is <= 1024 px; EXIF rotation is applied.
Usage:  python src/import_local.py field_photos [--source field]
"""
import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from collect_commons import dhash, hamming
from common import CLASS_NAMES, IMAGES, METADATA_CSV, merge_metadata, read_csv

EXTS = {".jpg", ".jpeg", ".png", ".webp"}   # convert iPhone HEIC to JPG first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--source", default="field")
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    root = Path(args.folder)
    IMAGES.mkdir(parents=True, exist_ok=True)
    meta = read_csv(METADATA_CSV)
    old = meta.to_dict("records") if meta is not None else []
    hashes = [r["dhash"] for r in old if r.get("dhash")]
    known = {r["title"] for r in old}
    rows = []

    added = dup = 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(root)
        title = f"{args.source}:{rel.as_posix()}"
        if title in known:
            continue
        parts = rel.parts[:-1]
        hint = parts[0] if parts and parts[0] in CLASS_NAMES else ""
        site_parts = parts[1:] if hint else parts
        site = "/".join(site_parts) or p.stem   # no folder -> each photo its own site
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        h = dhash(img)
        if any(hamming(h, o) <= 4 for o in hashes):
            dup += 1
            continue
        hashes.append(h)
        image_id = f"{args.source}_" + hashlib.sha1(title.encode()).hexdigest()[:12]
        img.save(IMAGES / f"{image_id}.jpg", quality=90)
        rows.append({"image_id": image_id, "file": f"raw/images/{image_id}.jpg", "source": args.source,
                     "title": title, "url": "", "license": "own work", "artist": "", "description": "",
                     "root_category": "", "category": "", "site_id": f"{args.source}:{site}",
                     "hint_label": hint, "dhash": h})
        added += 1
    total = len(merge_metadata(rows))
    print(f"added {added}, skipped {dup} near-duplicates, {total} total")


if __name__ == "__main__":
    main()
