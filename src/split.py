"""Assign train/val/test by site (group), before labeling.

All photos of one site/event go to the same split, so the test set measures generalization
to unseen landslides rather than recognition of the same scene. Splitting before labeling
lets review.py label the test split blind (VLM suggestion hidden).

Existing photo assignments are kept; only new photos are assigned. Greedy: largest groups first,
each into the split furthest below its target share.

Usage:  python src/split.py [--val 0.15 --test 0.2 --seed 0]
"""
import argparse
import random

import pandas as pd

from common import METADATA_CSV, SPLITS_CSV, read_csv, write_csv


def assign_splits(val=0.15, test=0.2, seed=0, verbose=True):
    """Give a split to every photo that has none yet; existing photo assignments never change.
    A new photo of a site that already has a split joins that split (majority if mixed);
    new sites are assigned greedily, largest first, to the split furthest below its share."""
    target = {"train": 1 - val - test, "val": val, "test": test}

    meta = read_csv(METADATA_CSV)
    old = read_csv(SPLITS_CSV)
    have = dict(zip(old.image_id, old.split)) if old is not None else {}
    known = meta[meta.image_id.isin(have)]
    site_split = {}
    if len(known):
        k = known.assign(split=known.image_id.map(have))
        site_split = k.groupby("site_id").split.agg(lambda s: s.value_counts().index[0]).to_dict()

    new = meta[~meta.image_id.isin(have)]
    count = pd.Series(list(have.values())).value_counts().reindex(list(target), fill_value=0).to_dict()
    sizes = new.groupby("site_id").size()
    new_sites = [s for s in sizes.index if s not in site_split]
    random.Random(seed).shuffle(new_sites)
    new_sites.sort(key=lambda s: -sizes[s])   # stable: ties stay shuffled
    for site in new_sites:
        total = sum(count.values()) + sizes[site]
        split = max(target, key=lambda s: target[s] * total - count[s])
        site_split[site] = split
        count[split] += sizes[site]

    out = meta[["image_id", "site_id"]].copy()
    out["split"] = [have.get(i) or site_split[s] for i, s in zip(out.image_id, out.site_id)]
    write_csv(out, SPLITS_CSV)
    if not verbose:
        return out
    summary = out.groupby("split").agg(images=("image_id", "size"), sites=("site_id", "nunique"))
    print(summary.to_string())
    mixed = out.groupby("site_id").split.nunique()
    if (mixed > 1).any():
        print(f"warning: {(mixed > 1).sum()} sites span several splits (run tools/link_sites.py to fix)")
    big = meta.groupby("site_id").size()
    big = big[big > 0.1 * len(meta)]
    if len(big):
        print("warning: sites holding >10% of images:", ", ".join(big.index))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assign_splits(args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
