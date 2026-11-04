"""Assemble the Hugging Face Space (Docker) for the web app and optionally push it.

    python tools/build_hf_space.py                          # stage into .cache/hf_space and list it
    python tools/build_hf_space.py --push USER/landslide    # stage, create the Space if needed, upload

The Space gets: deploy/hf_space/* (Dockerfile, README header, requirements, prefetch), src/*.py and src/*.json,
and the trained heads from models/ (the CNN checkpoint is not used by the app and stays out). Pushing needs a
Hugging Face write token: `hf auth login` once, or HF_TOKEN in the environment.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".cache" / "hf_space"
MODEL_FILES = ["selected.json", "probe_base.joblib", "probe_large.joblib", "probe_siglip.joblib",
               "outline_head.joblib", "dimension_object_prior.json"]


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "src").mkdir(parents=True)
    (STAGE / "models").mkdir()
    for f in (ROOT / "deploy" / "hf_space").iterdir():
        if f.is_file():
            shutil.copy2(f, STAGE / f.name)
    for f in sorted((ROOT / "src").glob("*")):
        if f.suffix in (".py", ".json"):
            shutil.copy2(f, STAGE / "src" / f.name)
    missing = [m for m in MODEL_FILES if not (ROOT / "models" / m).exists()]
    if missing:
        sys.exit(f"missing in models/: {missing}")
    for m in MODEL_FILES:
        shutil.copy2(ROOT / "models" / m, STAGE / "models" / m)
    files = sorted(p for p in STAGE.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files)
    print(f"staged {len(files)} files, {size / 1e6:.1f} MB in {STAGE}")
    return files


def push(repo):
    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()["name"]
    if "/" not in repo:
        repo = f"{user}/{repo}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo, repo_type="space",
                      commit_message="Update app", delete_patterns=["src/*", "models/*"])
    print(f"pushed: https://huggingface.co/spaces/{repo}")
    print(f"app:    https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="REPO", help="Space id, e.g. USER/landslide-classifier (or just the name)")
    a = ap.parse_args()
    for f in stage():
        print("  " + f.relative_to(STAGE).as_posix())
    if a.push:
        push(a.push)


if __name__ == "__main__":
    main()
