"""Download candidate landslide photos from Wikimedia Commons with license metadata.

Each category's `hint` is only a prior for the human reviewer; nothing is trusted until
verified in review.py. `site_id` is the category a file was found in and is used to keep
photos of the same landslide/event together when splitting train/val/test.

Usage:  python src/collect_commons.py [--max-per-category 300] [--max-per-search 120]
                                      [--search-only] [--dry-run]
"""
import argparse
import hashlib
import json
import re
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageOps

from common import DATA, IMAGES, METADATA_CSV, merge_metadata, read_csv

WALK_CACHE = DATA / "cache" / "commons_walk"

API = "https://commons.wikimedia.org/w/api.php"
UA = "IC202P-landslide-research/0.1 (student course project)"
THUMB_WIDTH = 1024
SMALL_WIDTH = 500    # standard thumbnail step, used when the original is narrower than THUMB_WIDTH
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

# category -> (hint label or None, recursion depth)
SOURCES = {
    "Category:Rockfalls": ("fall_topple", 3),
    "Category:Rock slides": ("translational_slide", 2),
    "Category:Blockslides": ("translational_slide", 2),
    "Category:Debris flow": ("debris_flow", 3),
    "Category:Lahars": ("debris_flow", 2),
    "Category:Mudflows": ("earth_mud_flow", 2),
    "Category:Mudslides": (None, 2),            # journalistic term; type varies
    "Category:Landslide scars": (None, 2),
    "Category:Landslide deposits": (None, 2),
    "Category:Active landslides": (None, 1),
    "Category:Landslides in India": (None, 3),
    "Category:Landslides in Nepal": (None, 3),
    "Category:Landslides in Himachal Pradesh": (None, 3),
    "Category:Landslides in Uttarakhand": (None, 3),
    # deployment region: more South Asian hill-road scenes (round 3)
    "Category:Landslides in Kerala": (None, 3),
    "Category:2024 Wayanad landslides": (None, 2),
    "Category:Landslides in Sikkim": (None, 3),
    "Category:Landslides in Bhutan": (None, 3),
    "Category:Landslides in Pakistan": (None, 3),
    "Category:Landslides in Sri Lanka": (None, 3),
    "Category:Landslides in Bangladesh": (None, 3),
    "Category:Mass movements (geology) in Nepal": (None, 3),
    "Category:Roads in Uttarakhand": ("no_landslide", 1),
    "Category:Roads in Himachal Pradesh": ("no_landslide", 1),
    "Category:Ghat roads in India": ("no_landslide", 1),
    # negatives: slopes and cuts without failure
    "Category:Road cuts": ("no_landslide", 1),
    "Category:Mountain roads in India": ("no_landslide", 1),
    "Category:Cliffs in India": ("no_landslide", 1),
}

# Full-text searches (File namespace). Category trees are full of aerial shots, glaciers and art;
# title searches return mostly ground photos taken by people on foot or on the road (many are
# Geograph field photos), which is what the deployed app receives from engineers' phones.
SEARCHES = {
    'intitle:rockfall': "fall_topple",
    'intitle:"rock fall"': "fall_topple",
    'intitle:"fallen rocks"': "fall_topple",
    'intitle:"fallen boulder"': "fall_topple",
    'intitle:topple': "fall_topple",
    'intitle:slump': "rotational_slide",
    'intitle:"rotational landslide"': "rotational_slide",
    'intitle:"rotational slide"': "rotational_slide",
    'intitle:"translational landslide"': "translational_slide",
    'intitle:"planar slide"': "translational_slide",
    'intitle:"shallow landslide"': "translational_slide",
    'intitle:"debris flow"': "debris_flow",
    'intitle:lahar': "debris_flow",
    'intitle:"debris avalanche"': "debris_flow",
    'intitle:"debris torrent"': "debris_flow",
    'intitle:earthflow': "earth_mud_flow",
    'intitle:"earth flow"': "earth_mud_flow",
    'intitle:mudflow': "earth_mud_flow",
    'intitle:"mud flow"': "earth_mud_flow",
    'intitle:mudslide': None,
    'intitle:landslip': None,
    'intitle:landslide intitle:road': None,
    'intitle:landslide India': None,
    'intitle:landslide Nepal': None,
    'intitle:landslide Uttarakhand': None,
    'intitle:landslide Himachal': None,
    'intitle:landslide Kerala': None,
    'intitle:landslide Sikkim': None,
    'intitle:landslide Darjeeling': None,
    'intitle:landslide': None,
    # second round: classes that were still scarce after the first round
    'intitle:"land slip"': None,
    'intitle:"slope failure"': None,
    'intitle:"earth slide"': None,
    'intitle:"soil slip"': None,
    'intitle:"mud slide"': "earth_mud_flow",
    'intitle:"thaw slump"': "earth_mud_flow",
    'intitle:"retrogressive thaw"': "earth_mud_flow",
    'intitle:"rotational slip"': "rotational_slide",
    'intitle:slip intitle:road': None,
    'intitle:"debris flow" -lahar': "debris_flow",
    'intitle:"flash flood" debris': "debris_flow",
    'intitle:"landslide" intitle:Japan': None,
    'intitle:"landslide" intitle:Taiwan': None,
    'intitle:"landslide" intitle:Philippines': None,
    'intitle:"landslide" intitle:Bhutan': None,
    'intitle:"road cutting"': "no_landslide",
    'intitle:"cut slope"': "no_landslide",
    'intitle:"road cut"': "no_landslide",
    'intitle:"ghat road"': "no_landslide",
    'intitle:"hill road"': "no_landslide",
    'intitle:"mountain road" India': "no_landslide",
    'intitle:"rock face"': "no_landslide",
    'intitle:"rock cutting"': "no_landslide",
}

# Subcategory rules, matched against the category a file was found in (case-insensitive).
# Protection/mitigation structures are useful hard negatives; art, maps, diagrams are not photos.
NEGATIVE_CATS = ("protection", "mitigation", "prevention", "barrier", "gallery", "galleries", "net",
                 "netting", "shed", "fence", "retaining", "anchor", "bolt", "warning")
EXCLUDE_CATS = ("in art", "map", "diagram", "icon", "sign", "simulation", "logo", "stamp",
                "painting", "drawing", "illustration", "video", "glacier", "moraine", "satellite",
                "aerial", "museum", "poster")
EXCLUDE_TITLE = ("painting", "drawing", "map", "diagram", "sign", "satellite", "aerial", "drone",
                 "illustration", "lithograph", "engraving", "poster", "chart", "graph", "plan",
                 "museum", "glacier", "moraine", "stamp", "logo", "screenshot", "model")


def has_word(text, vocab):
    """Whole-word match allowing simple plurals/forms ('nets', 'signs', 'maps', 'anchored'),
    so 'Netherlands', 'Anchorage' or 'design' do not match 'net', 'anchor', 'sign'."""
    words = set(re.findall("[a-z]+", text.lower()))
    forms = lambda v: {v, v + "s", v + "es", v + "ed", v + "ing"}  # noqa: E731
    return (any(words & forms(v) for v in vocab if " " not in v)
            or any(re.search(rf"\b{re.escape(v)}\b", text.lower()) for v in vocab if " " in v))


def category_hint(cat, root_hint):
    """Prior for the reviewer only (never used as a training label)."""
    if has_word(cat, EXCLUDE_CATS):
        return None
    if has_word(cat, NEGATIVE_CATS):
        return "no_landslide"
    return root_hint or ""


def title_ok(title):
    return not has_word(title.removeprefix("File:"), EXCLUDE_TITLE)


session = requests.Session()
session.headers["User-Agent"] = UA


def api(params):
    for attempt in range(8):
        try:
            # POST: 50 long file titles can exceed the URL length limit (HTTP 414) of a GET
            r = session.post(API, data={**params, "format": "json", "maxlag": 5}, timeout=30)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500 or "maxlag" in r.text[:200]:
            time.sleep(int(r.headers.get("Retry-After", 5 * (attempt + 1))))
            continue
        r.raise_for_status()
        time.sleep(0.2)
        return r.json()
    raise RuntimeError(f"Commons API kept failing for {params}")


def members(cat):
    """(files, subcategories) of a category, in one paginated query."""
    files, subs, cont = [], [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                 "cmtype": "file|subcat", "cmprop": "title|type", "cmlimit": 500, **cont})
        for m in d.get("query", {}).get("categorymembers", []):
            (files if m["type"] == "file" else subs).append(m["title"])
        if "continue" not in d:
            return files, subs
        cont = d["continue"]


def walk(root, depth, max_cats):
    """Breadth-first: (file_title, category_it_was_found_in). Stops after max_cats categories."""
    seen, queue, pairs = {root}, [(root, 0)], []
    while queue and len(seen) <= max_cats:
        cat, d = queue.pop(0)
        files, subs = members(cat)
        pairs += [(f, cat) for f in files if f.lower().endswith(EXTS)]
        if d < depth:
            for sub in subs:
                if sub not in seen:
                    seen.add(sub)
                    queue.append((sub, d + 1))
        if len(seen) % 25 == 0:
            print(f"    ...{len(seen)} categories queued, {len(pairs)} files", flush=True)
    return pairs


def walk_cached(root, depth, max_cats):
    """Category walks are slow (rate-limited); cache them so reruns are instant."""
    path = WALK_CACHE / (hashlib.sha1(f"{root}|{depth}".encode()).hexdigest()[:12] + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"walking {root} (depth {depth})", flush=True)
    pairs = walk(root, depth, max_cats)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs), encoding="utf-8")
    return pairs


def search(query, limit):
    """(file_title, site_category) for a full-text search. The site is the file's first visible
    category (usually the place, e.g. "Black Ven"), so photos of one landslide share a site."""
    titles, offset = [], 0
    while len(titles) < limit:
        d = api({"action": "query", "list": "search", "srsearch": query, "srnamespace": 6,
                 "srlimit": 500, "sroffset": offset})
        titles += [x["title"] for x in d.get("query", {}).get("search", [])
                   if x["title"].lower().endswith(EXTS)]
        if "continue" not in d:
            break
        offset = d["continue"]["sroffset"]
    titles = titles[:limit]
    site = {}
    for i in range(0, len(titles), 50):
        cont = {}
        while True:
            d = api({"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "categories",
                     "clshow": "!hidden", "cllimit": "max", **cont})
            for p in d.get("query", {}).get("pages", {}).values():
                cats = [c["title"] for c in p.get("categories", [])]
                if cats and p["title"] not in site:
                    site[p["title"]] = cats[0]
            if "continue" not in d:
                break
            cont = d["continue"]
    return [(t, site.get(t, "Category:search " + query)) for t in titles]


def search_cached(query, limit):
    path = WALK_CACHE / (hashlib.sha1(f"search|{query}|{limit}".encode()).hexdigest()[:12] + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"searching {query}", flush=True)
    pairs = search(query, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs), encoding="utf-8")
    return pairs


def round_robin(pairs, n, exclude):
    """Pick up to n files, cycling over categories so many different sites are represented."""
    by_cat = {}
    for title, cat in pairs:
        if title not in exclude and title_ok(title) and category_hint(cat, "") is not None:
            by_cat.setdefault(cat, []).append(title)
    picked, lists = [], list(by_cat.items())
    while len(picked) < n and lists:
        nxt = []
        for cat, titles in lists:
            if len(picked) == n:
                break
            picked.append((titles.pop(0), cat))
            if titles:
                nxt.append((cat, titles))
        lists = nxt
    return picked


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def image_info(titles, width=THUMB_WIDTH):
    """Metadata + a download URL. Original files are rate-limited hard by the media servers (429,
    Retry-After 600), scaled thumbnails are not. When the original is narrower than the requested
    width the API returns the original, so those files are re-requested at a smaller standard
    thumbnail width (Geograph photos are ~640 px wide)."""
    d = api({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
             "iiprop": "url|size|extmetadata", "iiurlwidth": width})
    info = {}
    for p in d.get("query", {}).get("pages", {}).values():
        if "imageinfo" not in p:
            continue
        ii = p["imageinfo"][0]
        em = ii.get("extmetadata", {})
        info[p["title"]] = {
            "download_url": ii.get("thumburl") or ii["url"],
            "url": ii.get("descriptionurl", ""),
            "width": ii.get("width", 0),
            "height": ii.get("height", 0),
            "license": strip_html(em.get("LicenseShortName", {}).get("value")),
            "artist": strip_html(em.get("Artist", {}).get("value"))[:200],
            "description": strip_html(em.get("ImageDescription", {}).get("value"))[:500],
        }
    unscaled = [t for t, v in info.items() if "/thumb/" not in v["download_url"]]
    if unscaled and width > SMALL_WIDTH:
        for t, v in image_info(unscaled, SMALL_WIDTH).items():
            if "/thumb/" in v["download_url"]:
                info[t] = v
    return info


class RateLimited(Exception):
    pass


def download(url):
    """Fetch one image. A 429 from the media servers carries Retry-After: 600, so instead of
    stalling we raise RateLimited and let the caller retry the file later."""
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            raise RateLimited(url)
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return ImageOps.exif_transpose(Image.open(BytesIO(r.content))).convert("RGB")
    raise RuntimeError("download kept failing")


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, :-1] > g[:, 1:]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-category", type=int, default=300, help="files per root category")
    ap.add_argument("--max-subcats", type=int, default=400, help="subcategories visited per root")
    ap.add_argument("--max-per-search", type=int, default=120, help="files per search query")
    ap.add_argument("--search-only", action="store_true", help="skip the category trees")
    ap.add_argument("--match", help="only sources whose name matches this regex (to run collectors in parallel)")
    ap.add_argument("--min-side", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true", help="list and count only, no downloads")
    args = ap.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)

    hint_of = {**{root: hint for root, (hint, _) in SOURCES.items()},
               **{"search:" + q: hint for q, hint in SEARCHES.items()}}

    def reapply_rules(r):   # category rules may have changed since the file was downloaded
        if r.get("source") == "wikimedia_commons":
            r["hint_label"] = category_hint(r["category"], hint_of.get(r["root_category"])) or ""

    rows = (read_csv(METADATA_CSV) if args.dry_run else merge_metadata(update=reapply_rules))
    rows = rows.to_dict("records") if rows is not None else []
    hashes = [r["dhash"] for r in rows if r.get("dhash")]
    seen_titles = {r["title"] for r in rows if r.get("title")}

    sources = [(root, hint, lambda r=root, d=depth: walk_cached(r, d, args.max_subcats))
               for root, (hint, depth) in SOURCES.items() if not args.search_only]
    sources += [("search:" + q, hint, lambda q=q: search_cached(q, args.max_per_search * 3))
                for q, hint in SEARCHES.items()]
    if args.match:
        sources = [s for s in sources if re.search(args.match, s[0], re.I)]
    for root, hint, get_pairs in sources:
        # re-read: another collector process may have added files since we started
        rows = merge_metadata().to_dict("records")
        hashes = [r["dhash"] for r in rows if r.get("dhash")]
        seen_titles = {r["title"] for r in rows if r.get("title")}
        pairs = get_pairs()
        already = sum(1 for r in rows if r.get("root_category") == root)
        cap = args.max_per_search if root.startswith("search:") else args.max_per_category
        picked = round_robin(pairs, max(0, cap - already), seen_titles)
        print(f"{root:50s} {len(pairs):5d} files found, {already} already have, {len(picked)} to fetch", flush=True)
        if args.dry_run:
            continue
        cat_of = dict(picked)
        seen_titles |= set(cat_of)
        added = skipped = 0
        pending = list(cat_of)
        for attempt in range(2):
            limited = []
            for i in range(0, len(pending), 50):
                new_rows = []
                for title, inf in image_info(pending[i:i + 50]).items():
                    if min(inf["width"], inf["height"]) < args.min_side:
                        skipped += 1
                        continue
                    try:
                        img = download(inf["download_url"])
                    except RateLimited:
                        limited.append(title)
                        time.sleep(2)
                        continue
                    except Exception as e:
                        print(f"  skip {title}: {e}")
                        skipped += 1
                        continue
                    h = dhash(img)
                    if any(hamming(h, o) <= 4 for o in hashes):
                        skipped += 1
                        continue
                    hashes.append(h)
                    image_id = "commons_" + hashlib.sha1(title.encode()).hexdigest()[:12]
                    img.save(IMAGES / f"{image_id}.jpg", quality=90)
                    cat = cat_of[title]
                    new_rows.append({
                        "image_id": image_id, "file": f"raw/images/{image_id}.jpg", "source": "wikimedia_commons",
                        "title": title, "url": inf["url"], "license": inf["license"], "artist": inf["artist"],
                        "description": inf["description"], "root_category": root, "category": cat,
                        "site_id": "commons:" + cat.removeprefix("Category:"), "hint_label": category_hint(cat, hint),
                        "dhash": h,
                    })
                    added += 1
                    time.sleep(0.5)
                rows += new_rows
                merge_metadata(new_rows)
                print(f"    {min(i + 50, len(pending))}/{len(pending)} checked, {added} added", flush=True)
            if not limited:
                break
            print(f"  {len(limited)} rate-limited; retrying once after a pause", flush=True)
            pending = limited
            time.sleep(120)
        skipped += len(limited)
        print(f"  {root}: {added} added, {skipped} skipped (small/duplicate/failed)", flush=True)
    print(f"done: {len(rows)} images in {METADATA_CSV}")


if __name__ == "__main__":
    main()
