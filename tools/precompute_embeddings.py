"""Fill the probe's embedding cache for every photo (labeled or not), so probe.py only fits.

Embeds variant 0 (the photo) for all images and phone-style variants 1..aug for train/val
images. Resumable: the cache is saved every 40 batches by probe.embed_all.

Usage:  python tools/precompute_embeddings.py [--backbone facebook/dinov2-base] [--aug 3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402,F401
from common import METADATA_CSV, SPLITS_CSV, read_csv  # noqa: E402
from probe import embed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--aug", type=int, default=3)
    args = ap.parse_args()
    df = read_csv(METADATA_CSV).merge(read_csv(SPLITS_CSV)[["image_id", "split"]], on="image_id")
    embed_all(df, args.backbone)
    fit = df[df.split.isin(["train", "val"])]
    for k in range(1, args.aug + 1):
        embed_all(fit, args.backbone, k)
    test = df[df.split == "test"]
    embed_all(test, args.backbone, 1)   # phone-degraded test copies (probe_phonesim)
    print("done", args.backbone)


if __name__ == "__main__":
    main()
