"""Download Creative-Commons photos from Openverse (mostly Flickr) for the deployment region.

Wikimedia Commons has few ground photos of South Asian landslides; Flickr has many roadside
phone/camera photos from Indian and Nepali hill roads. Openverse indexes them with licence data.
Photos already on Commons are skipped (source=wikimedia), near-duplicates are skipped by dhash.
`site_id` = photographer, because one person's photos are usually of the same slide/trip;
keeping them in one split avoids leakage between train and test.

Openverse anonymous limits: 20 requests/min, ~200/day. Images are fetched from the original
host (Flickr), which does not count against that limit.

Usage:  python src/collect_openverse.py [--pages 5]
"""
import argparse
import hashlib
import time
from io import BytesIO

import requests
from PIL import Image

from collect_commons import dhash, hamming
from common import IMAGES, merge_metadata

API = "https://api.openverse.org/v1/images/"
UA = "IC202P-landslide-research/0.1 (student course project)"
QUERIES = [
    "landslide India", "landslide Uttarakhand", "landslide Himachal", "landslide Kerala",
    "landslide Sikkim", "landslide Darjeeling", "landslide Meghalaya", "landslide Munnar",
    "landslide Kodagu", "landslide Nepal", "landslide Bhutan", "landslide Karakoram",
    "landslide Leh Manali", "landslide Kashmir", "landslide Sri Lanka", "landslide road",
    "landslide highway", "road blocked landslide", "mudslide road", "rockfall road",
    "rock fall road", "debris flow", "landslip road", "ghat road", "hill road India",
    "mountain road India", "road cutting India", "Himalayan road",
]
session = requests.Session()
session.headers["User-Agent"] = UA


def search(q, page):
    for attempt in range(5):
        try:
            r = session.get(API, params={"q": q, "page": page, "page_size": 20, "license_type": "all-cc",
                                         "excluded_source": "wikimedia"}, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(60)
            continue
        if r.status_code >= 400:   # past the last page, or a bad query
            return [], False
        time.sleep(3.5)            # stay under 20 requests/min
        j = r.json()
        return j.get("results", []), page < j.get("page_count", 0)
    return [], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=5, help="pages of 20 results per query")
    ap.add_argument("--min-side", type=int, default=400)
    args = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)

    rows = merge_metadata().to_dict("records")
    have = {r["image_id"] for r in rows}
    hashes = [r["dhash"] for r in rows if r.get("dhash")]
    done = {r.get("root_category") for r in rows}
    for q in QUERIES:
        if "openverse:" + q in done:   # finished in an earlier run (saves the daily request quota)
            continue
        added = skipped = 0
        new_rows = []
        for page in range(1, args.pages + 1):
            results, more = search(q, page)
            for x in results:
                image_id = "ov_" + hashlib.sha1(x["id"].encode()).hexdigest()[:12]
                if image_id in have:
                    continue
                have.add(image_id)
                if min(x.get("width") or 9999, x.get("height") or 9999) < args.min_side:
                    skipped += 1
                    continue
                try:
                    r = session.get(x["url"], timeout=60)
                    r.raise_for_status()
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                except Exception:
                    skipped += 1
                    continue
                if min(img.size) < args.min_side:
                    skipped += 1
                    continue
                img.thumbnail((1024, 1024))
                h = dhash(img)
                if any(hamming(h, o) <= 4 for o in hashes):
                    skipped += 1
                    continue
                hashes.append(h)
                img.save(IMAGES / f"{image_id}.jpg", quality=90)
                lic = f"CC {x.get('license', '').upper()} {x.get('license_version', '')}".strip()
                if x.get("license") in ("pdm", "cc0"):
                    lic = x["license"].upper()
                new_rows.append({
                    "image_id": image_id, "file": f"raw/images/{image_id}.jpg", "source": f"openverse_{x['source']}",
                    "title": x.get("title") or "", "url": x.get("foreign_landing_url", ""), "license": lic,
                    "artist": (x.get("creator") or "")[:200],
                    "description": " ".join(t["name"] for t in (x.get("tags") or [])[:20])[:500],
                    "root_category": "openverse:" + q, "category": q,
                    "site_id": f"openverse:{x['source']}:{x.get('creator') or x['id']}", "hint_label": "",
                    "dhash": h,
                })
                added += 1
            if not more:
                break
        merge_metadata(new_rows)
        print(f"{q:30s} {added} added, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
