"""How does the app behave on hard, ambiguous landslide photos? (complements system_test.py)

The labeling rule "unsure between two types -> reject" removed the hardest landslide photos
from the scored test set. A second AI pass over all test rejects
(data/labels/ai_annotation/hard_cases/hard_cases.csv) marked:
  - hard: usable ground photo of a real slope failure whose type is ambiguous (+ best guess)
  - wrongly_rejected: clearly usable, type clear
This script runs the deployed Predictor on those photos and reports what the engineer would
see: rejected as a bad photo / "uncertain" (incl. family answers) / a confident type, and how
often confident answers and family answers agree with the best guess. For truly ambiguous
photos, saying "uncertain" is the desired behaviour; best guesses are themselves uncertain.

Usage:  python tools/hard_case_eval.py
Output: results/hard_cases.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402,F401
from common import DATA, METADATA_CSV, RESULTS, read_csv  # noqa: E402
from predict import FAMILIES, Predictor, load_image  # noqa: E402

HARD = DATA / "labels" / "ai_annotation" / "hard_cases" / "hard_cases.csv"


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            epilog='See the module docstring for details.').parse_args()
    hard = read_csv(HARD)
    if hard is None or hard.empty:
        raise SystemExit(f"{HARD} missing or empty")
    meta = read_csv(METADATA_CSV).set_index("image_id")
    pred = Predictor("auto")
    rows = []
    for i in range(0, len(hard), 16):
        chunk = hard.iloc[i:i + 16]
        for (_, h), r in zip(chunk.iterrows(), pred([load_image(DATA / meta.loc[iid, "file"]) for iid in chunk.image_id])):
            rows.append({**h.to_dict(), "status": r["status"], "pred": r["label"], "conf": round(r["confidence"], 3),
                         "family": r["family"] or "", "second": r["ranked"][1][0]})
    d = pd.DataFrame(rows)
    lines = [f"# Hard-case behaviour (model: {pred.model_name}, confident ≥ {pred.confident_min})\n",
             "Photos the first labeling pass rejected; a second AI pass found these usable "
             "(labels are AI best guesses, see data/labels/ai_annotation/hard_cases/).\n"]
    for kind, g in d.groupby("kind"):
        ok = g[g.status == "ok"]
        fam = g[(g.status == "uncertain") & (g.family != "")]
        lines += [f"## {kind} (n = {len(g)})\n", "| Outcome | Share | Agrees with best guess |", "|---|---|---|",
                  f"| rejected as unusable photo | {(g.status == 'bad_photo').mean():.2f} | – |",
                  f"| uncertain (top two shown) | {(g.status == 'uncertain').mean():.2f} | "
                  f"top-2 contains best guess: {((g.pred == g.best_guess) | (g.second == g.best_guess))[g.status == 'uncertain'].mean():.2f} |",
                  f"| … of which family answer | {len(fam) / len(g):.2f} | "
                  f"{pd.Series([b in FAMILIES[f] for b, f in zip(fam.best_guess, fam.family)]).mean() if len(fam) else float('nan'):.2f} |",
                  f"| confident type | {len(ok) / len(g):.2f} | {(ok.pred == ok.best_guess).mean() if len(ok) else float('nan'):.2f} |", ""]
    d.to_csv(RESULTS / "hard_cases.csv", index=False)
    (RESULTS / "hard_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
