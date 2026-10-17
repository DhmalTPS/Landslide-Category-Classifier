"""Shared config: class definitions, paths and CSV helpers used by every script.

Import this before transformers/timm: it moves the Hugging Face model cache to the project
drive (C: is nearly full; Qwen2.5-VL-7B alone is ~16 GB).
"""
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

# Photo titles contain any Unicode; a Windows console/pipe (cp1252) must not crash a long run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import pandas as pd  # noqa: E402

# IC202P_WORKDIR redirects data/, results/ and models/ (used for tests on throwaway data).
WORK = Path(os.environ.get("IC202P_WORKDIR", ROOT))
DATA = WORK / "data"
IMAGES = DATA / "raw" / "images"
METADATA_CSV = DATA / "metadata.csv"          # every candidate image (from collectors)
VLM_CSV = DATA / "vlm_predictions.csv"        # Qwen-VL suggestions / zero-shot predictions
LABELS_DIR = DATA / "labels"                  # human labels, one CSV per reviewer
SPLITS_CSV = DATA / "splits.csv"              # train/val/test assignment
RESULTS = WORK / "results"
MODELS = WORK / "models"

# Classes visible in a ground photo taken from downhill, mapped to Hungr et al. (2014) types.
CLASSES = {
    "fall_topple": (
        "Rock fall or topple: steep rock face or cliff with detached blocks; boulders, talus or "
        "scattered rock fragments at the base or on a road. (Hungr 2014: rock/ice fall, "
        "boulder/debris fall, rock block topple, flexural topple)"),
    "rotational_slide": (
        "Rotational slide / slump: curved, spoon-shaped concave main scarp; displaced block "
        "tilted back toward the slope; back-tilted trees; bulging or hummocky toe. "
        "(Hungr 2014: rock/clay/silt rotational slide)"),
    "translational_slide": (
        "Translational (planar) slide: shallow, elongated scar on a slope with a flat sliding "
        "surface, often along bedding or joint planes; a sheet of soil, debris or rock slabs "
        "moved down without rotating. (Hungr 2014: rock planar/wedge slide, "
        "gravel/sand/debris slide)"),
    "debris_flow": (
        "Debris flow: material that flowed down a gully or channel; unsorted mix of boulders, "
        "gravel and mud, levees along the channel, a fan-shaped deposit at the outlet. "
        "(Hungr 2014: debris flow, debris flood, debris avalanche)"),
    "earth_mud_flow": (
        "Earthflow or mudflow: fine-grained, wet soil or mud moving as a tongue or lobe; "
        "smooth or hummocky surface, few large boulders. (Hungr 2014: earthflow, mud flow)"),
    "no_landslide": (
        "No landslide: a stable slope, cliff, road cut, quarry, construction site or "
        "landscape with no visible slope failure."),
}
CLASS_NAMES = list(CLASSES)
REJECT = "reject"   # image unusable: aerial/satellite view, map, diagram, too close-up, not a slope


def read_csv(path, **kw):
    path = Path(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw) if path.exists() else None


def write_csv(df, path):
    """Atomic write (tmp file + replace). On Windows the replace fails while another program
    (Excel, a reader) has the file open, so retry for a while before giving up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    df.to_csv(tmp, index=False)
    for attempt in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (1 + attempt % 4))
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"could not replace {path} (open in another program?)")


class _Lock:
    """Cross-process lock via an exclusively created file (collectors may run concurrently).
    A lock file older than `stale` seconds is treated as left over from a crashed process;
    the owner token makes sure a process only ever removes its own lock."""
    def __init__(self, path, stale=300):
        self.path, self.stale = Path(str(path) + ".lock"), stale
        self.token = f"{os.getpid()}-{uuid.uuid4().hex}"

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, self.token.encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            if self.path.read_text() == self.token:
                self.path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def merge_metadata(new_rows=(), update=None):
    """Add rows to metadata.csv without clobbering rows written by other processes.

    Re-reads the file under a lock, appends rows whose image_id is new, optionally applies
    `update(row_dict)` to every row, and writes atomically. Returns the merged DataFrame."""
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _Lock(METADATA_CSV):
        cur = read_csv(METADATA_CSV)
        if cur is None and not new_rows:
            return pd.DataFrame(columns=["image_id", "file", "title", "site_id", "dhash"])
        rows = cur.to_dict("records") if cur is not None else []
        have = {r["image_id"] for r in rows}
        added = [r for r in new_rows if r["image_id"] not in have]
        rows += added
        changed = False
        if update:
            for r in rows:
                before = dict(r)
                update(r)
                changed = changed or r != before
        df = pd.DataFrame(rows)
        if added or changed or cur is None:   # unchanged metadata is never rewritten
            write_csv(df, METADATA_CSV)
    return df


def all_labels():
    """Every reviewer's labels: image_id, label, reviewer, vlm_shown, time."""
    frames = [read_csv(p) for p in sorted(LABELS_DIR.glob("*.csv"))]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=["image_id", "label", "reviewer", "vlm_shown", "time", "viewpoint"])
    return pd.concat(frames, ignore_index=True).fillna("")


def is_ai(reviewer):
    """Reviewers named *_ai are AI annotators (tools/ai_labels.py), not human ground truth."""
    return str(reviewer).endswith("_ai")


def voting_rows(g):
    """The label rows that vote for one image: human labels take precedence; AI-annotator
    votes count only where no human labeled the image."""
    human = g[~g.reviewer.map(is_ai)]
    return human if len(human) else g


def tied_images():
    """image_ids whose (human-precedence) votes are tied; review.py --conflicts shows these."""
    out = []
    for image_id, g in all_labels().groupby("image_id"):
        votes = voting_rows(g).label.value_counts()
        if len(votes) > 1 and votes.iloc[0] == votes.iloc[1]:
            out.append(image_id)
    return out


def final_labels(verbose=True):
    """Ground truth: majority vote across reviewers; ties are left out until resolved.
    Human labels take precedence: AI-annotator votes count only where no human labeled the image.
    Viewpoint (up/down) comes from the voting reviewers, else from any reviewer who recorded it."""
    lab = all_labels()
    rows = []
    for image_id, g_all in lab.groupby("image_id"):
        g = voting_rows(g_all)
        votes = g.label.value_counts()
        if len(votes) > 1 and votes.iloc[0] == votes.iloc[1]:
            continue
        own = [v for v in g.get("viewpoint", pd.Series(dtype=str)).fillna("") if v]
        views = own or [v for v in g_all.get("viewpoint", pd.Series(dtype=str)).fillna("") if v]
        rows.append({"image_id": image_id, "label": votes.index[0], "n_reviewers": len(g),
                     "vlm_shown": "1" if (g.vlm_shown == "1").any() else "0",
                     "reviewers": "+".join(sorted(g.reviewer.unique())),
                     # majority; ties prefer "up" (deterministic across runs)
                     "viewpoint": max(sorted(set(views)), key=lambda v: (views.count(v), v == "up")) if views else ""})
    out = pd.DataFrame(rows, columns=["image_id", "label", "n_reviewers", "vlm_shown", "reviewers", "viewpoint"])
    ties = lab.image_id.nunique() - len(out)
    if verbose and ties:
        print(f"note: {ties} images have tied reviewer votes and are excluded (resolve in review.py --conflicts)")
    return out


def labeled_dataset(include_reject=False, val_frac=0.15, seed=0):
    """metadata + split + final label for every usable labeled image.

    If no labeled image is in val (e.g. a provided dataset with only train/test), a val set
    is carved out of train by site, in memory only (splits.csv is not changed)."""
    meta, splits = read_csv(METADATA_CSV), read_csv(SPLITS_CSV)
    df = meta.merge(splits[["image_id", "split"]], on="image_id").merge(final_labels(), on="image_id")
    if not (df.split == "val").any() and (df.split == "train").any():
        # chosen from ALL labeled train sites (rejects included), so every model gets the same val
        sites = sorted(df[df.split == "train"].site_id.unique())
        rng = random.Random(seed)
        rng.shuffle(sites)
        val_sites = set(sites[:max(1, round(val_frac * len(sites)))])
        df.loc[df.site_id.isin(val_sites) & (df.split == "train"), "split"] = "val"
        print(f"note: no val split; using {len(val_sites)} train sites as val")
    if not include_reject:
        df = df[df.label != REJECT]
    return df.reset_index(drop=True)


# Deployment region, used by eval.py and tools/robustness_eval.py (one definition for both).
SOUTH_ASIA = re.compile(
    r"\b(?:india\b|indian\b|nepal|bhutan|pakistan|sri lanka|bangladesh|himalaya|himachal|uttarakhand|"
    r"kerala|wayanad|sikkim|darjeeling|kashmir|ladakh|assam\b|meghalaya|mizoram|arunachal|nagaland|"
    r"manipur|karnataka|tamil nadu|nilgiri|munnar|idukki|kodagu|coorg|goa\b|konkan|ghats?\b|shimla|"
    r"manali|kullu|kinnaur|chamoli|kedarnath|joshimath|rishikesh|gangtok|kathmandu|pokhara|annapurna|"
    r"gilgit|hunza|kandy\b|madikeri|attabad|karakoram|guptakashi|zanskar|spiti)", re.I)


def is_south_asia(row):
    """True if a photo's category/title/description names a South Asian place.
    `root_category` counts only when it is a real Commons category: for search results it holds
    the search query (e.g. "intitle:landslide Uttarakhand"), which says nothing about the photo."""
    root, cat = str(row.get("root_category", "")), str(row.get("category", ""))
    parts = [str(row.get(c, "")) for c in ("title", "description")]
    # Openverse rows and Commons search placeholders store the search query in `category`
    if not root.startswith("openverse:") and not cat.startswith("Category:search "):
        parts.append(cat)
    if root.startswith("Category:"):
        parts.append(root)
    return bool(SOUTH_ASIA.search(" ".join(parts).replace("_", " ")))


def data_fingerprint():
    """sha1 of the (image_id, split, site, final label) table: what a trained model depends on.
    Stored with every model; check_integrity.py compares it instead of file timestamps."""
    import hashlib
    df = labeled_dataset(include_reject=True)[["image_id", "split", "site_id", "label"]].sort_values("image_id")
    return hashlib.sha1(df.to_csv(index=False).encode()).hexdigest()[:16]
