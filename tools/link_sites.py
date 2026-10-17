"""Merge photos of the same place/event into one site, then fix splits so no site spans two.

`site_id` from the collectors is a proxy (first Commons category, or the Flickr photographer).
The same landslide can appear under different categories (e.g. "Earthflow deposits in the UK"
vs "Active landslides" for Black Ven), which leaks near-identical scenes across train/test.
Photos are linked (union-find) when they share:
  - the original site_id;
  - a specific title stem (title without numbers, extension, geograph suffix), if that stem is
    used at <= 3 original sites (generic titles like "Rock fall" are not specific);
  - a Starr photo series day (File:Starr-YYMMDD-...);
  - near-identical image content: DINOv2 cosine similarity >= --visual (default 0.90).
The merged id is written to metadata `site_id` (original kept in `site_id_raw`), and every
merged site that spans splits is moved entirely to the split holding most of its photos
(ties: train > val > test, so test only keeps groups it clearly owns). Models must be
retrained afterwards.

Usage:  python tools/link_sites.py [--visual 0.90] [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import DATA, SPLITS_CSV, merge_metadata, read_csv, write_csv  # noqa: E402

PRIORITY = {"train": 0, "val": 1, "test": 2}


def stem(t):
    t = t.lower()
    t = re.sub(r"^file:", "", t)
    t = re.sub(r"\.(jpe?g|png|tiff?|webp)$", "", t)
    t = re.sub(r"- geograph\.org\.uk - \d+", "", t)
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"\d+", "", t)
    return re.sub(r"[^a-z]+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = read_csv(METADATA := DATA / "metadata.csv")
    raw = (meta.site_id_raw.where(meta.site_id_raw != "", meta.site_id) if "site_id_raw" in meta
           else meta.site_id)
    d = meta.assign(raw=raw.values).reset_index(drop=True)
    n = len(d)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    stems = d.title.map(stem)
    sites_per_stem = pd.Series(d.raw.values, index=stems.values).groupby(level=0).nunique()
    first = {}
    for i in range(n):
        st = stems[i]
        sm = re.match(r"file:starr-(\d{6})", d.title[i].lower())
        keys = ["site:" + d.raw[i]]
        if len(st) >= 12 and len(st.split()) >= 2 and sites_per_stem.get(st, 99) <= 3:
            keys.append("stem:" + st)
        if sm:
            keys.append("starr:" + sm.group(1))
        for k in keys:
            if k in first:
                union(i, first[k])
            else:
                first[k] = i

    cache = np.load(DATA / "cache" / "emb_facebook_dinov2-base.npz")
    have = [i for i in range(n) if d.image_id[i] in cache.files]
    X = np.stack([cache[d.image_id[i]] for i in have]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(0, len(have), 500):
        S = X[s:s + 500] @ X.T
        for a, b in zip(*np.where(S >= args.visual)):
            if s + a < b:
                union(have[s + a], have[b])

    root = [find(i) for i in range(n)]
    gid = {r: "group:" + d.image_id[r] for r in set(root)}
    d["site_new"] = [gid[r] for r in root]
    # keep the original id for groups that did not merge anything
    sizes = d.groupby("site_new").raw.nunique()
    d.loc[d.site_new.map(sizes) == 1, "site_new"] = d.raw
    print(f"original sites {d.raw.nunique()} -> merged sites {d.site_new.nunique()}")

    splits = read_csv(SPLITS_CSV)
    sp = d[["image_id", "site_new"]].merge(splits[["image_id", "split"]], on="image_id", how="left")
    moves = []
    for site, g in sp.groupby("site_new"):
        counts = g.split.value_counts()
        if len(counts) > 1:
            target = min(counts.index, key=lambda s: (-counts[s], PRIORITY[s]))
            for _, r in g[g.split != target].iterrows():
                moves.append((r.image_id, r.split, target, site))
    mv = pd.DataFrame(moves, columns=["image_id", "from", "to", "site"])
    print(f"{len(mv)} photos change split ({mv.groupby(['from', 'to']).size().to_dict() if len(mv) else {}})")
    if args.dry_run:
        return
    out = DATA / "site_merge_moves.csv"
    mv.to_csv(out, index=False)
    new_site = dict(zip(d.image_id, d.site_new))

    def update(r):
        if not r.get("site_id_raw"):
            r["site_id_raw"] = r["site_id"]
        r["site_id"] = new_site.get(r["image_id"], r["site_id"])
    merge_metadata(update=update)
    target = dict(zip(mv.image_id, mv.to))
    splits["split"] = [target.get(i, s) for i, s in zip(splits.image_id, splits.split)]
    splits["site_id"] = splits.image_id.map(new_site).fillna(splits.get("site_id", ""))
    write_csv(splits, SPLITS_CSV)
    print(f"updated metadata site_id and {SPLITS_CSV}; moves logged in {out}")


if __name__ == "__main__":
    main()
