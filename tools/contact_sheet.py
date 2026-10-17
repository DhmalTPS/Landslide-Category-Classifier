"""Tile photos into numbered contact sheets for quick visual checks (viewpoint, content).

Usage:  python tools/contact_sheet.py OUT_DIR [--ids ids.txt] [--cols 5] [--rows 4] [--tile 256]
Writes OUT_DIR/sheet_###.jpg and OUT_DIR/index.csv (sheet, cell, image_id).
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import DATA, METADATA_CSV, read_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ids", help="text file with one image_id per line (default: all)")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--skip-labeled", action="store_true", help="only images nobody has labeled")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    meta = read_csv(METADATA_CSV)
    if args.skip_labeled:
        done = set()
        for p in (DATA / "labels").glob("*.csv"):
            done |= set(read_csv(p).image_id)
        meta = meta[~meta.image_id.isin(done)]
    if args.limit:
        meta = meta.head(args.limit)
    if args.ids:
        keep = Path(args.ids).read_text().split()
        meta = meta.set_index("image_id").loc[keep].reset_index()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per, t = args.cols * args.rows, args.tile
    index = []
    for s in range(0, len(meta), per):
        sheet = Image.new("RGB", (args.cols * t, args.rows * t), "white")
        draw = ImageDraw.Draw(sheet)
        for k, r in enumerate(meta.iloc[s:s + per].itertuples()):
            img = ImageOps.contain(Image.open(DATA / r.file).convert("RGB"), (t - 4, t - 4))
            x, y = (k % args.cols) * t, (k // args.cols) * t
            sheet.paste(img, (x + 2, y + 2))
            draw.rectangle([x + 2, y + 2, x + 30, y + 20], fill="yellow")
            draw.text((x + 5, y + 5), str(k), fill="black")
            index.append({"sheet": s // per, "cell": k, "image_id": r.image_id})
        sheet.save(out / f"sheet_{s // per:03d}.jpg", quality=85)
        (out / f"sheet_{s // per:03d}.txt").write_text("\n".join(
            f"{k}: {r.image_id} | {r.title.removeprefix('File:')[:90]}"
            for k, r in enumerate(meta.iloc[s:s + per].itertuples())), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(index).to_csv(out / "index.csv", index=False)
    print(f"{len(index)} images -> {out} ({(len(index) + per - 1) // per} sheets)")


if __name__ == "__main__":
    main()
