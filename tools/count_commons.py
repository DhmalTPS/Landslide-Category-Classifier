"""Recursive unique image-file counts for Commons categories (deduplicated across subcategories)."""
import sys
from explore_commons import get, subcats


def files_in(cat):
    out, cont = [], {}
    while True:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "file",
                 "cmlimit": 500, **cont})
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


def recursive_files(cat, maxdepth, depth=0, seen=None):
    seen = set() if seen is None else seen
    if cat in seen:
        return set()
    seen.add(cat)
    fs = {f for f in files_in(cat) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
    if depth < maxdepth:
        for c in subcats(cat):
            fs |= recursive_files(c, maxdepth, depth + 1, seen)
    return fs


if __name__ == "__main__":
    depth = int(sys.argv[1])
    for cat in sys.argv[2:]:
        print(f"{cat.removeprefix('Category:'):40s} {len(recursive_files(cat, depth)):6d}", flush=True)
