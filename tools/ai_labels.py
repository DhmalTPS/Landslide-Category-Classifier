"""Record labels made by an AI annotator from contact sheets (tools/contact_sheet.py).

These labels are NOT human ground truth. They are stored under their own reviewer name
(default `claude_ai`) so eval.py can say who made the test labels, and humans can check them
with `review.py --relabel-others` (agreement is reported by eval.py).

Codes file: one line per sheet, `<sheet> <cell><code>[v] ...`, e.g. `003 0f 1x 2tv 3n`
  f fall_topple, r rotational_slide, t translational_slide, d debris_flow,
  e earth_mud_flow, n no_landslide, x reject;  suffix v = ground photo looking down (from above)

Usage:  python tools/ai_labels.py SHEET_DIR CODES_FILE [--reviewer claude_ai]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import LABELS_DIR, REJECT, read_csv, write_csv  # noqa: E402

CODES = {"f": "fall_topple", "r": "rotational_slide", "t": "translational_slide", "d": "debris_flow",
         "e": "earth_mud_flow", "n": "no_landslide", "x": REJECT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_dir")
    ap.add_argument("codes")
    ap.add_argument("--reviewer", default="claude_ai")
    args = ap.parse_args()

    index = pd.read_csv(Path(args.sheet_dir) / "index.csv", dtype=str)
    cell = {(int(r.sheet), int(r.cell)): r.image_id for r in index.itertuples()}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    new = {}
    for line in Path(args.codes).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        try:
            sheet = int(parts[0].rstrip(":"))
        except ValueError:
            raise SystemExit(f"bad sheet number in line: {line!r}")
        for tok in parts[1:]:
            num = "".join(ch for ch in tok if ch.isdigit())
            code = tok[len(num):]
            if code[:1] not in CODES or code[1:] not in ("", "v"):
                raise SystemExit(f"bad code {tok!r} on sheet {sheet}")
            if not num or (sheet, int(num)) not in cell:
                raise SystemExit(f"cell {tok!r} on sheet {sheet} does not exist in index.csv")
            label = CODES[code[0]]
            view = "" if label == REJECT else ("down" if code.endswith("v") else "up")
            new[cell[(sheet, int(num))]] = {"label": label, "viewpoint": view}
    missing = [k for k in cell if cell[k] not in new]
    if missing:
        print(f"warning: {len(missing)} cells without a code, e.g. sheet/cell {missing[:5]}")

    path = LABELS_DIR / f"{args.reviewer}.csv"
    old = read_csv(path)
    rows = [] if old is None else [r for r in old.to_dict("records") if r["image_id"] not in new]
    rows += [{"image_id": i, "label": v["label"], "reviewer": args.reviewer, "vlm_shown": "0",
              "time": now, "viewpoint": v["viewpoint"]} for i, v in new.items()]
    write_csv(pd.DataFrame(rows), path)
    df = pd.DataFrame(rows)
    print(f"{len(new)} labels recorded; {path} now has {len(df)}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
