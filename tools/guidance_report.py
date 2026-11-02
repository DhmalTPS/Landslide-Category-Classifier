"""Does the in-app guidance help? Evidence from field use (data/app_uploads/guidance_log.csv).

The app logs, with every uploaded photo: which instant photo-check items were flagged, whether the user
sent it anyway ("override"), how many retakes came before it, whether the live camera guide was used,
the mode and survey shot, and the interface language, next to the server's answer (status, confidence,
usable score).

The questions this answers (with Wilson 95% intervals; differences are reported, not over-claimed):
  1. Does the photo check predict bad photos?  P(server says bad_photo or uncertain | flagged) vs | clean.
  2. Do overrides pay off?                     outcomes of flagged photos the user sent anyway.
  3. Do retakes help?                          outcomes after >= 1 retake vs first attempts.
  4. Does the live camera guide help?          outcomes with vs without the live guide.
  5. Which check items are flagged most often (where users need the most guidance).

  python tools/guidance_report.py [--csv data/app_uploads/guidance_log.csv]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd

from common import DATA

OUT = Path(__file__).resolve().parent.parent / "results" / "guidance_report.md"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(df, mask, col="weak"):
    sub = df[mask]
    k, n = int(sub[col].sum()), len(sub)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}-{hi:.0%})" if n else "no photos yet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA / "app_uploads" / "guidance_log.csv"))
    a = ap.parse_args()
    if not Path(a.csv).exists():
        print(f"no guidance log yet ({a.csv}). Run the app (without --no-save) and take photos with the phone.")
        return
    df = pd.read_csv(a.csv, dtype=str).fillna("")
    df["weak"] = df.status.isin(["bad_photo", "uncertain"])
    df["bad"] = df.status.eq("bad_photo")
    df["flagged"] = df.flags.ne("")
    df["retook"] = pd.to_numeric(df.retakes, errors="coerce").fillna(0) > 0
    df["live_on"] = df.live.str.lower().eq("true")
    df["over"] = df.override.str.lower().eq("true")
    lines = [f"# Guidance report ({len(df)} photos)", "",
             "Weak answer = the server said *bad photo* or *uncertain*.", "",
             "| Question | Group | Weak answers |", "|---|---|---|",
             f"| 1. Photo check predicts weak answers | flagged | {rate(df, df.flagged)} |",
             f"| | clean | {rate(df, ~df.flagged)} |",
             f"| 2. Overrides | flagged, sent anyway | {rate(df, df.flagged & df.over)} |",
             f"| 3. Retakes | after >= 1 retake | {rate(df, df.retook)} |",
             f"| | first attempt | {rate(df, ~df.retook)} |",
             f"| 4. Live camera guide | used | {rate(df, df.live_on)} |",
             f"| | not used | {rate(df, ~df.live_on)} |", "",
             "Flagged items:", ""]
    items = df.flags[df.flagged].str.split("|").explode().value_counts()
    lines += [f"- {k}: {v}" for k, v in items.items()] or ["- none"]
    lines += ["", f"Modes: {df['mode'].value_counts().to_dict()}; languages: {df['lang'].value_counts().to_dict()}",
              "", "Read with care: field users are not randomised to guidance on/off, so these are associations."]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
