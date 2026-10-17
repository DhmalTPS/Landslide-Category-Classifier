"""Print the Wikimedia Commons category tree under a root, with direct file counts."""
import sys, time, requests

API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "IC202P-landslide-research/0.1 (student project; contact via repo)"


def get(params):
    for attempt in range(5):
        r = S.get(API, params={**params, "format": "json"}, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.5)
        return r.json()
    raise RuntimeError("rate limited")


def subcats(cat):
    d = get({"action": "query", "list": "categorymembers", "cmtitle": cat,
             "cmtype": "subcat", "cmlimit": 500})
    return [m["title"] for m in d["query"]["categorymembers"]]


def nfiles(cat):
    d = get({"action": "query", "prop": "categoryinfo", "titles": cat})
    return next(iter(d["query"]["pages"].values())).get("categoryinfo", {}).get("files", 0)


def walk(cat, depth, maxdepth, seen):
    if cat in seen:
        return
    seen.add(cat)
    print(f"{'  ' * depth}{cat.removeprefix('Category:')}  [{nfiles(cat)}]", flush=True)
    if depth < maxdepth:
        for c in subcats(cat):
            walk(c, depth + 1, maxdepth, seen)


if __name__ == "__main__":
    seen = set()
    for root in sys.argv[2:]:
        walk(root, 0, int(sys.argv[1]), seen)
