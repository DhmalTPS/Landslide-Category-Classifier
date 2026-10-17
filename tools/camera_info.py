"""Add camera make/model (from EXIF, via the Commons API) to metadata.csv and flag phone photos.

The app receives phone photos, so the test split's phone-captured subset is the closest thing
we have to real deployment input; eval.py reports it separately (column `is_phone`).

Usage:  python tools/camera_info.py
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from collect_commons import api  # noqa: E402
from common import merge_metadata  # noqa: E402

PHONE_MAKES = ("apple", "samsung", "xiaomi", "redmi", "poco", "oneplus", "oppo", "vivo", "realme",
               "huawei", "honor", "motorola", "google", "nokia", "lg electronics", "htc", "sony mobile",
               "asus", "lenovo", "zte", "tecno", "infinix", "micromax", "lava", "nothing", "fairphone")
PHONE_MODEL = re.compile(r"iphone|galaxy|\bsm-[a-z]|pixel|redmi|\bmi \d|oneplus|nexus|moto|lumia|xperia", re.I)


def is_phone(make, model):
    mk = make.lower()
    if mk.startswith("sony") and not PHONE_MODEL.search(model):
        return False                       # Sony cameras vs Sony Xperia phones
    return any(mk.startswith(m) for m in PHONE_MAKES) or bool(PHONE_MODEL.search(model))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            epilog='See the module docstring for details.').parse_args()
    rows = merge_metadata().to_dict("records")
    todo = [r["title"] for r in rows if r.get("source") == "wikimedia_commons" and not r.get("camera_checked")]
    found = {}
    for i in range(0, len(todo), 50):
        d = api({"action": "query", "titles": "|".join(todo[i:i + 50]), "prop": "imageinfo",
                 "iiprop": "commonmetadata"})
        for p in d.get("query", {}).get("pages", {}).values():
            md = {m["name"]: m["value"] for m in (p.get("imageinfo") or [{}])[0].get("commonmetadata", [])
                  if isinstance(m.get("value"), str)}
            make, model = md.get("Make", "").strip(), md.get("Model", "").strip()
            found[p["title"]] = (make, model)
        print(f"{min(i + 50, len(todo))}/{len(todo)}", flush=True)

    def update(r):
        if r.get("title") in found:
            make, model = found[r["title"]]
            r.update(camera_make=make, camera_model=model, is_phone="1" if is_phone(make, model) else "0",
                     camera_checked="1")

    if not found:   # nothing new: do not rewrite metadata.csv
        print("no photos without camera info")
        return
    df = merge_metadata(update=update)
    print(f"phone photos: {(df.get('is_phone') == '1').sum()} of {len(df)}")


if __name__ == "__main__":
    main()
