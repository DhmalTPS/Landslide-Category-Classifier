# IC202P: Landslide type classification from field phone photos

Status: 2026-10-16. Final numbers after two full audits, retrained from clean data (results reproduce
exactly on re-run; the pipeline also ran end to end on a fresh copy of the data). All scores
measure agreement with **AI-made labels**; see "Label caveat".

## Summary

- **Task:** a junior engineer takes a phone photo of a slope from downhill. The app (`src/app.py`)
  first checks that the photo is usable, then gives the landslide type (Hungr et al. 2014
  classes), its confidence, an on-site checklist and a safety note.
- **App model:** an ensemble of three frozen-feature linear probes: DINOv2-base, DINOv2-large and
  SigLIP2-base. It was chosen **on train+val only**, by site-balanced cross-validation plus three
  distribution-shift tests. When confidence is below 0.6 the app says "uncertain" and shows the
  top two types.
- **Untouched test split** (263 usable photos, 152 sites, no site shared with train/val):

  | Model | Accuracy | Macro-F1 [95% CI] |
  |---|---|---|
  | App model | 0.806 | 0.784 [0.688, 0.857] |
  | Fine-tuned ConvNeXt-Tiny | 0.722 | 0.665 |
  | Qwen2.5-VL zero-shot | 0.555 | 0.327 |

- **End to end** (`tools/system_test.py`, all 1,008 test photos including unusable ones):
  - **Usable photos:** the app answers 69% confidently, and those answers are right **86%** of
    the time (88% on phone-degraded copies). 14% are marked uncertain, and 17% are wrongly
    rejected as bad photos.
  - **All photos:** 30% of confident answers were on photos that should have been rejected.
    Precision over all photos is therefore **0.60**. The test split has about 3 unusable photos
    per usable one; in the field, where most uploads are slope photos, precision will be closer
    to the usable-photo figure.
- **Still to do by the team:**
  1. A person re-labels the test split blind; all labels are currently AI-made.
  2. Test the app on 30–50 local phone photos.

## 1. Problem and classes

| Class | Hungr 2014 types (as in `src/common.py`) |
|---|---|
| `fall_topple` | rock/ice fall, boulder/debris fall, rock block topple, flexural topple |
| `rotational_slide` | rock/clay/silt rotational slide (slump) |
| `translational_slide` | rock planar/wedge slide, gravel/sand/debris slide |
| `debris_flow` | debris flow, debris flood, debris avalanche |
| `earth_mud_flow` | earthflow, mud flow |
| `no_landslide` | stable slope, cut or cliff |
| `reject` (photo check) | aerial, satellite, map, art, archive scan, close-up, slope not visible |

## 2. Data

No public ground-level landslide-type dataset exists; the benchmarks (LandslideBench, Bijie,
Landslide4Sense…) use overhead imagery.

**Sources (4,954 photos):**
- **Wikimedia Commons, 3,684 photos:**
  - 2,426 from title searches, including about 1,130 Geograph field photos;
  - 879 from South-Asian landslide/road categories;
  - 379 from the first Rockfalls / Rock slides / Blockslides category walks.
- **Openverse, 1,270 photos:** 1,267 from Flickr, 3 from other hosts.
- **Phone photos:** 636 are phone photos according to their EXIF data, which exists for Commons
  photos only.

**Usable photos:** 1,325, of which 322 are from South Asia and 237 are phone photos. The rest were
rejected as unusable. Reject rates by source: title searches 62%, category trees 82%,
Openverse 86%.

| | fall_topple | rotational | translational | debris_flow | earth_mud | no_landslide | reject |
|---|---|---|---|---|---|---|---|
| train | 194 | 67 | 99 | 180 | 39 | 329 | 2,406 |
| val | 25 | 6 | 22 | 13 | 4 | 84 | 478 |
| test | 58 | 11 | 32 | 29 | 8 | 125 | 745 |

- **Split by site, fixed before labeling.** `tools/link_sites.py` merges photos of the same
  place or event into one site, because the same place can sit under different Commons
  categories (e.g. Black Ven, the Maui debris flow, Attabad Lake). Photos are linked by:
  - specific title stems;
  - photo series;
  - DINOv2 similarity ≥ 0.90.
  This gives 2,062 sites. No site spans two splits, and no usable test photo has a
  near-identical train/val photo (checked by `tools/check_integrity.py`).
- **One large event:** 132 photos of a single debris flow (Nedumpoil, Kerala 2022) form one
  site, all in train. The robustness metrics are site-balanced so it cannot dominate.
- **Licences:** stored per photo in `data/metadata.csv`, with credits in `data/credits.csv`.
  - 753 photos are NC (non-commercial) and 411 are ND (no derivatives). This is fine for a
    course project; see `results/licences.md` for what is allowed.
  - Credit every photo you show.

### Label caveat

- **Who labeled:** every label was made by an AI annotator (Claude, reviewer `claude_ai`)
  following `labeling_guide.md` and `tools/ai_labeling_prompt.md`. The full audit trail is in
  `data/labels/ai_annotation/`.
- **Spot checks:** two manual checks (48 + 24 photos; samples in `data/labels/ai_annotation/qa*`)
  judged about 80–90% of labels consistent with the guide (a judgement, not a recorded score; the
  2 clear errors found were corrected, `qa2/fix.txt`).
- **Noise detection on train/val only** (`tools/find_label_issues.py`):
  - 299 suspicious labels were re-labeled blind; the second look agreed with the first on 63%.
  - After the site merge, 9 of those photos moved into test. They got their first-look labels
    back, so the test split carries no noise correction.
- **The "unsure → reject" rule:** when the annotator was unsure between two types, the photo
  was rejected. So the hardest landslide photos are not in the scored test set. Section 4.4
  measures the app on them separately.
- **What this means for the scores:** they measure agreement with these labels. Before quoting
  them as final, a teammate or faculty member must re-label the test split blind:

  ```
  .venv\Scripts\python src/review.py --reviewer <name> --split test --relabel-others
  ```

  Human labels override AI labels automatically, and `src/eval.py` then reports Cohen's kappa
  between people and the AI labels.

## 3. Models

| Model | What it is |
|---|---|
| Qwen2.5-VL-7B zero-shot | VLM baseline, 4-bit; class read from next-token probabilities of option letters |
| `probe_base` / `probe_large` / `probe_siglip` | frozen DINOv2-base / DINOv2-large / SigLIP2-base features + logistic regression; photo-check head + type head |
| `cnn` | ConvNeXt-Tiny fine-tuned end to end (25 epochs, best val epoch 14) |
| **selected (app)** | **average of the three probes' type probabilities; photo check = average of their check heads** |

**Shared training details:**
- **Phone-style copies.** Probes train on each photo plus 3 phone-style copies
  (`src/phone_aug.py`: framing, tilt, exposure, blur, noise, JPEG recompression). The CNN uses the
  same augmentation on the fly.
- **Choosing C.** The regularisation strength is chosen by site-grouped, class-stratified CV on
  train+val. The same rule runs in `src/probe.py` and `tools/robustness_eval.py`, so the
  evaluated models are the ones deployed.
- **No test leakage.** Each model stores the ids of the photos it was trained on. `eval.py`,
  `system_test.py` and `check_integrity.py` refuse any model that saw a test photo.

**How the app's model was chosen** (`tools/robustness_eval.py`, train+val only; the rule is
recorded in `models/selected.json` with a hash of the selection code):
1. Compare all seven systems: the three probes and every ensemble of them.
2. Score each by the **mean site-balanced macro-F1** of four tests:
   - site-grouped, stratified 5-fold CV;
   - Commons → Openverse;
   - outside → South Asia;
   - known non-phone cameras → phone photos.
   Test sites never appear in training.
3. The highest score wins; ties at 3 decimals go to the system with fewer members.
4. The "uncertain" threshold is the first confidence level that reaches 85% CV accuracy at
   ≥ 50% coverage. That is 0.6.

## 4. Results

### 4.1 Robustness (train+val, cross-validated; `results/robustness.md`)

| System | Robust score | CV macro-F1 (site-bal.) | →Openverse | →South Asia F1 (acc) | →phone camera | ECE | Δ CV F1 vs selected (P ≤ 0) |
|---|---|---|---|---|---|---|---|
| **base+large+siglip** | **0.676** | 0.714 | 0.743 | 0.578 (0.741) | 0.669 | 0.040 | – |
| large+siglip | 0.670 | 0.701 | 0.689 | 0.617 (0.741) | 0.675 | 0.036 | +0.013 (0.08) |
| base+large | 0.665 | 0.712 | 0.695 | 0.587 (0.734) | 0.668 | 0.029 | +0.004 (0.38) |
| base+siglip | 0.654 | 0.704 | 0.703 | 0.548 (0.737) | 0.662 | 0.063 | +0.007 (0.32) |
| large | 0.651 | 0.684 | 0.661 | 0.587 (0.712) | 0.669 | 0.091 | +0.025 (0.04) |
| base | 0.639 | 0.680 | 0.719 | 0.558 (0.723) | 0.598 | 0.040 | +0.032 (0.00) |
| siglip | 0.583 | 0.637 | 0.549 | 0.536 (0.690) | 0.611 | 0.028 | +0.065 (0.00) |

- **Δ column:** a paired site bootstrap of CV macro-F1 (selected minus the other system). The
  selected ensemble is clearly better than every single backbone. Against the two-model
  ensembles the gaps are small (P(≤ 0) between 0.08 and 0.38).
- **Selective prediction** (CV, type head only, before the photo check): answering only at
  confidence ≥ 0.6 covers 73% of photos at 88.1% accuracy. At ≥ 0.8 it covers 46% at 94.4%.

### 4.2 Untouched test split (263 usable photos, 152 sites; `results/summary.md`)

| Model | Accuracy | Macro-F1 [95% CI, site bootstrap] | Phone-degraded acc / F1 |
|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | 0.555 | 0.327 [0.268, 0.399] | – |
| ConvNeXt-Tiny fine-tuned | 0.722 | 0.665 [0.548, 0.745] | 0.684 / 0.582 |
| probe_siglip | 0.726 | 0.642 [0.536, 0.725] | 0.722 / 0.639 |
| probe_base | 0.745 | 0.649 [0.537, 0.738] | 0.741 / 0.661 |
| probe_large | 0.776 | 0.736 [0.643, 0.810] | 0.787 / 0.745 |
| **selected (app)** | **0.806** | **0.784 [0.688, 0.857]** | 0.768 / 0.697 |

The test split agrees with the train+val choice: the selected ensemble is the best system here
too.

**Subgroups for the selected model** (macro-F1 over the classes present in each subgroup):

| Subgroup | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Uphill/level views (deployment viewpoint) | 254 | 0.815 | 0.800 |
| South Asia | 48 | 0.854 | 0.807 |
| Real phone photos (EXIF) | 20 | 0.600 | 0.526 |

The phone subgroup has n = 20, so that number is very uncertain.

**Per class** (test, selected):

| Class | Recall | Precision |
|---|---|---|
| fall_topple | 0.67 | 0.66 |
| rotational_slide (n = 11) | 0.91 | 0.83 |
| translational_slide | 0.53 | 0.65 |
| debris_flow | 0.97 | 0.93 |
| earth_mud_flow (n = 8) | 0.75 | 0.75 |
| no_landslide | 0.90 | 0.88 |

### 4.3 End-to-end system test (`results/system_test.md`)

The real app pipeline (photo check, then type, then threshold) run on all 1,008 test photos.

| | Original | Phone-degraded |
|---|---|---|
| Photo check accuracy (usable vs reject) | 0.836 | 0.832 |
| Usable photos accepted / unusable rejected | 0.833 / 0.838 | 0.825 / 0.835 |
| Confident answers given (all photos) | 258 | 254 |
| … of which on photos that should have been rejected | 29.5% | 31.5% |
| **End-to-end precision, all photos** | **0.605** | **0.602** |
| Usable photos answered confidently / marked uncertain | 0.692 / 0.141 | 0.662 / 0.163 |
| **Type accuracy when answered, usable photos** | **0.857** | **0.879** |

- **Calibration transfers to test:** CV predicted 88.1% accuracy at ≥ 0.6 before the photo
  check, and test gives 85.7%.
- **The photo check is the weakest link.** It wrongly rejects about 17% of usable photos and
  lets through about 16% of unusable ones.

### 4.4 Hard, ambiguous photos (`results/hard_cases.md`)

A second AI pass over the 745 rejected test photos found **58 usable photos of real but
ambiguous landslides** (plus 6 clearly usable photos that had been wrongly rejected). On the 58
hard photos the app:
- rejects 29% as bad photos;
- says "uncertain" on 34%; in 65% of those, the best guess is among its top two;
- answers confidently on 36%, and only 57% of those confident answers match the best guess.

So on genuinely ambiguous photos the app hesitates or refuses about two-thirds of the time,
which is the desired behaviour. When it does answer them confidently, it is right much less
often than on clear photos (57% vs 86%).

### 4.5 Findings

1. **Frozen foundation-model features with linear heads beat both alternatives.** Single probes
   score 0.64–0.74 test macro-F1 and the ensemble 0.78. The fine-tuned CNN reaches 0.67, because
   about 900 training photos are too few for full fine-tuning. The zero-shot VLM reaches 0.33; it
   rarely recognises debris flows and earth/mud flows. This matches LandslideBench (2026,
   overhead imagery): Qwen3-VL-8B zero-shot 19% vs fine-tuned ConvNeXt 79%.
2. **The three-backbone ensemble has the best combined robustness score** (and is best on →Openverse
   and on test). It does not win every shift test: on →South Asia, large+siglip (0.617), base+large
   and DINOv2-large alone (both 0.587) beat its 0.578, and on →phone camera large+siglip is higher
   (0.675 vs 0.669). Not every ensemble beats every single model either: base+siglip is weaker on
   →South Asia than DINOv2-large alone.
3. **Region shift is the main risk.** Trained without any South-Asian photo and tested on them,
   site-balanced macro-F1 drops from 0.71 (CV) to 0.58; accuracy from 0.77 to 0.74. With South-Asian
   photos in training (the app), the 48 South-Asian test photos score 0.854 accuracy / 0.807 macro-F1,
   so the collected South-Asian data matters.
4. **Phone-style degradation costs some macro-F1** (test 0.784 → 0.697), and its cost in the app
   shows up as **coverage**: usable photos answered confidently fall from 0.692 to 0.662 and
   "uncertain" rises from 0.141 to 0.163, while accuracy on confident answers stays high
   (0.857 → 0.879). The confidence threshold absorbs the loss.
   - **Caveat:** the degradation recipe is the one used in training, so this test is
     favourable.
   - Real phone photos (n = 20) score lower (0.60 accuracy).
5. **The weakest areas:**
   - **`translational_slide`:** often predicted as rock fall (13 test photos); and 40 true debris
     flows were predicted as translational slides in CV.
   - **`earth_mud_flow`:** 51 photos in total, 43 in train+val; CV recall 0.40.
   - **`no_landslide` ↔ `fall_topple`:** a stable cliff with old talus vs a rock fall.

### 4.6 Cheap accuracy levers, out-of-fold (`tools/accuracy_eval.py`, added 2026-11-03)

These runs use usable train+val photos only (1062 photos, 586 sites), with site-grouped 5-fold CV and the
same splitter as 4.1. C is chosen inside the training folds and the test split is not read. The difference
column is a paired site bootstrap with a 90 % interval.

| variant | macro-F1 | site-weighted F1 | accuracy | answered / acc there | diff vs shipped |
|---|---|---|---|---|---|
| shipped (mean of 3 probes) | 0.704 | 0.714 | 77.3 % | 73 % / 88.1 % | - |
| log opinion pool | 0.694 | 0.707 | 77.4 % | 80 % / 85.8 % | -0.009 [-0.022, +0.005] |
| stacked weights (nested) | 0.659 | 0.663 | 73.7 % | 68 % / 86.0 % | -0.045 [-0.068, -0.023] |
| concatenated features | 0.707 | 0.713 | 78.8 % | 85 % / 82.8 % | +0.002 [-0.013, +0.017] |
| multi-crop TTA, test only | 0.717 | 0.712 | 78.2 % | 74 % / 87.0 % | +0.012 [-0.004, +0.027] |
| multi-crop TTA, train+test | 0.715 | 0.727 | 78.2 % | 74 % / 86.8 % | +0.012 [-0.004, +0.027] |
| TTA over phone-style copies | 0.706 | 0.701 | 77.6 % | 70 % / 89.0 % | +0.001 [-0.018, +0.020] |
| multi-crop + copies | 0.713 | 0.708 | 78.0 % | 71 % / 88.0 % | +0.008 [-0.007, +0.024] |
| per-class offsets (nested) | 0.713 | 0.705 | 78.4 % | 72 % / 88.7 % | +0.007 [-0.018, +0.030] |
| multi-crop + copies + offsets | 0.699 | 0.701 | 77.0 % | 70 % / 89.5 % | -0.007 [-0.039, +0.023] |

Multi-crop averaging (the full photo plus 5 crops at 0.85 scale) adds about one point. Its interval still
includes 0, and accuracy among confident answers drops slightly, so the shipped models were not changed.
With about 590 sites, a 1-point gain is within noise; more labelled sites would settle it.
Nine cheap levers tried on the frozen features all land within ±1.2 points of the shipped system. On
these features, accuracy has plateaued, and the next gain has to come from more labelled sites (above
all translational_slide and earth_mud_flow) or human-checked labels (§2).

## 5. Field app

```
.venv\Scripts\python src/app.py [--token SECRET] [--no-save]
```

**To open it:** use the printed address on a phone on the same Wi-Fi or hotspot. If it doesn't
load, allow Python through Windows Firewall for private networks.

**Flow:**
1. "Take photo" opens the camera. The browser fixes the rotation and shrinks the photo before
   upload.
2. The **photo check** runs. Below 50% usable the result is still shown in full, marked "not
   reliable", with sizes on one tap (see *Photo quality* below).
3. The **type** is shown with all probabilities.
4. Below 0.6 confidence the app says **uncertain** and shows the top two. If both are flows (or
   both slides) and together reach 0.6, it gives the family.
5. **Check on site** and **Safety** notes for that type.

**Hardening:**
- uploads are capped at 25 MB and 40 megapixels;
- requests time out;
- one photo is processed at a time;
- `--token` optionally restricts access;
- uploads are saved to `data/app_uploads/` (capped by `--max-saved`) so they can be labeled
  later.

**Status:** a laptop prototype, not a public website (README "Status").

### 5.1 User guidance (`src/guide.py`, added 2026-11-02)

The guidance follows the user's journey (stepper: Safe spot → Frame → Check → Result), with one aid at each stage.

| Stage | Guidance |
|---|---|
| First use | 5-card illustrated tour (what the app does, where to stand, framing, light/level, reading results); skippable, replayable from Help |
| Before capture | Per-session safety checklist (4 items, all must be ticked) before the camera opens. Single-photo or **site-survey** mode with a 4-shot list (whole slope, scarp, deposit close-up, scale object). Distance helper (pace count × stride, rangefinder, Maps) with range validation |
| Capture | Live viewfinder (needs `--https` or localhost): scarp/toe framing bands per shot, rule-of-thirds grid, level line and aim angle from the phone sensors, live pills coloured by severity (green / amber minor / red serious), a **photo-quality meter** (Good / Fair / Poor / Very poor), one instruction at a time with a direction arrow for aim and framing, a ring that is green only when nothing is wrong, haptics, optional voice. Level and instruction change only after two readings in a row agree, so the hints do not flicker. The shutter always works. Falls back to the phone camera app |
| Check | **Instant on-phone photo check** before upload: light, shadows, highlights, haze, blur, resolution, level, aim, each graded minor or serious. Good and Fair photos go straight to analysis; Poor and Very poor ones show the graded list with "Retake" and "Analyse anyway" (one tap each) |
| Waiting | Upload %, then 5 stages paced by the learned server time (moving average of past runs), field tips, cancel, 4-minute timeout |
| Result | **Next steps** card (retake; add the distance and re-measure the same photo; scale shot when depth trust is low; deposit close-up when grain size is unreadable; next survey shot). Tickable on-site signs with a tally; when uncertain, the top-2 types' signs are compared side by side. A "what do these numbers mean" glossary, plus a spoken summary |
| Photo quality | A band above every result: Good / Fair / Poor / Very poor with the reasons, a Retake button from Poor on, and the line "nothing is adjusted to look better". The result itself is never hidden: a photo that fails the server's photo check still shows its type, all probabilities, the field questions and signs, marked "not reliable", and sizes on one tap with a caveat that the 90% ranges were checked on usable photos only. History, reports and the shared text carry the quality |
| Always | Help sheet (photo guide, safety, glossary, limits, settings), recent observations kept on the phone (8, with thumbnails), offline banner, retry, sun mode (high contrast), keyboard/Esc and screen-reader labels |

**How the photo quality is graded.** Each check is *minor* when the photo is beyond the 0.5th / 99.5th
percentile of the training photos (the calibrated thresholds in `src/photo_check.json`) and *serious* when it is far outside
(fixed margins in `guide.data()["severe"]`, e.g. light below 0.7x the minimum or sharpness below 0.4x; these
margins are UI steps, not fitted to labelled data). Good = no issue; Fair = 1-2 minor; Poor = 1 serious or 3+
minor; Very poor = 2+ serious, or the server's photo check below 50% usable. Results are shown for every
level because the server's photo check wrongly refuses about 17% of usable test photos (system test above):
hiding those answers would lose good results, while the band keeps the warning in front of the user.
Headless browser tests (Edge, English and Hindi, 28 checks each, mocked server) cover the four levels,
the hysteresis, the result and sizes for a failed photo check, history tags and the live meter, arrow and
shutter ring.

**Photo-check thresholds are data-driven** (`tools/calibrate_photo_check.py` → `src/photo_check.json`):
- six pixel statistics on the photo scaled to 512 px: brightness, crushed shadows, blown highlights, contrast, sharpness, and the smooth blue/bright **sky fraction**;
- a photo is flagged when a statistic is outside the q-th / (100−q)-th percentile of accepted **train** photos, with q the largest value giving at most a 5% combined false-flag rate on train: q = 0.5;
- the val split is held out and the test split is never read;
- the phone recomputes the statistics with the same Pillow-style resampling and float32 rounding, because canvas scaling made sharpness about 40% higher.

| Check | Result |
|---|---|
| False flags on good photos | train 3.6%, val 3.9% |
| Detection of simulated failures (val) | blur 100%, dusk-dark 100%, mostly sky 100%, haze 96.1%, overexposure 86.4% |
| Phone vs Python decisions (Edge) | 36/36 agree (5 statistics); 10/10 with sky; median differences 0.00% (sky 0.02%) |

### 5.2 Guidance upgrades (2026-11-02)

| # | Upgrade | How it works |
|---|---|---|
| 1 | **Cut-off warning** | The server reports which photo borders the landslide outline covers by more than 15% (same rule as `cut_off`). The app names the edge and the fix, e.g. "top: tilt up or step back" |
| 2 | **Mostly-sky check** | The sky fraction (above) is shown live in the viewfinder and in the instant check |
| 3 | **Site result for surveys** | `POST /fuse`: the type is the mean of the shots' class probabilities. Sizes use `fuse_site` over the whole-slope shots: the median, with the between-photo spread added to the range. Grains come from the deposit close-up. A combined card and a site report are shown |
| 4 | **Decision questions** when unsure (or the top-2 margin is below 0.2) | Up to 5 yes/no field questions. The model's probabilities are updated by Bayes' rule with likelihoods P(yes \| type), and each next question has the largest expected information gain. Answers can be undone. The likelihoods are expert rules of thumb from Hungr 2014 (`src/guide_data.py`), kept within 0.05–0.92, and the screen says so |
| 5 | **Type first, sizes after** | `stage=type` returns the type in about 3–5 s and keeps the photo on the server (last 6). `POST /measure?job=` adds the sizes (about 25 s); while waiting, the user can tick signs and answer questions. Re-measuring with a typed distance reuses the job, so there is no re-upload; if the job has expired, the photo is sent again automatically |
| 6 | **Observation record** | Notes, weather, GPS (needs https) and the shot are stored with each result. "Save report" writes a standalone HTML page (photo, outline, type, answers, ticked signs, sizes) that prints to PDF. "Share" sends a text summary with the photo |
| 7 | **Offline queue** | Without a connection, photos wait in IndexedDB on the phone and are sent automatically when it returns (on reconnect, or every 60 s). Results go to history and survey fusion |
| 8 | **Practice quiz** | 5 picture pairs (close-up, mostly sky, aerial, tilted, dark vs a good photo), with feedback. Shown after the tour, replayable from Help |
| 9 | **Guidance evidence** | Each upload logs the check flags, overrides, retakes, live-guide use, mode, shot and language to `data/app_uploads/guidance_log.csv`. `tools/guidance_report.py` reports weak-answer rates (bad photo or uncertain) per group, with Wilson 95% CIs |
| 10 | **Hindi** | Language button: text nodes are translated as they appear (dictionary of about 300 strings plus patterns for dynamic text); speech uses hi-IN |

History now keeps the full last 20 results (with photo) on the phone; tapping one reopens it.

**Limitations:**
- The live steadiness limit is a heuristic: no labelled video exists to calibrate it. It only delays the green ring, never the shutter.
- Field usability has not yet been tested with real junior engineers. `guidance_report.py` will measure it once photos come in; it shows associations only, because users are not randomised.
- The Hindi text was machine-drafted and needs a native speaker's review. Some mixed-format sentences stay partly in English.
- The question likelihoods are not fitted to data; the dataset has no field answers.

### 5.3 Overlapping photo sets, walk-along scale and video (features 2-3, added 2026-11-03)

A landslide too big for one photo can be sent as a **set** of 2-8 overlapping photos. There are two ways
to take one:
- **turn on the spot**: the photos are joined into one picture with a homography mosaic;
- **walk along** the slope: the photos are analysed together.

The pipeline:
- **Matching and checks** (`src/overlap.py`): RootSIFT with MAGSAC++ estimates both a homography and a
  fundamental matrix. A parallax test tells turning from walking. It then checks overlap bands, order,
  duplicates, tilt, zoom and light.
- **Calibration** (`tools/overlap_eval.py`): simulated sets from train photos (simulation) give a link
  threshold of 30 inliers, 0.0 % false links, an overlap error of 0.2 points, 100 % walk detection and a
  weak band below 20 %.
- **Type for a set:** the joined picture gets weight w = 0.25 and the single photos the rest. In
  out-of-fold tests this gives macro-F1 0.710, against 0.674 for the middle photo alone and 0.712 for
  the whole photo.
- **Walk-along metric scale:** the engineer types the normal steps walked between photos (0.75 m per
  step). The central photo is triangulated against its neighbours, using an essential matrix, recoverPose
  and points with reprojection error < 2 px and ray angle > 1°. This gives metric depths, which then fix
  the size scale. `tools/stereo_eval.py` tests it on simulated pairs with known depth, with the step
  length (log-sd 0.15) and the focal also simulated as uncertain:
  - median |log error| 0.120 and 90th percentile 0.272;
  - stated sd 0.150, against 0.345 for the known-object prior;
  - 90 % coverage 87.6 %;
  - a scale was found for 71 % of pairs; pairs that are too far away for the baseline are refused.

  On the real server (step 7 of the night plan) a simulated walk set with 4 steps measured 1548 points
  on the landslide. Its 90 % length range narrowed from 5.7x (no steps) to 3.3x (max/min).
- **Focal length of a joined picture** (`tools/focal_eval.py`, simulated turns without EXIF):
  - The focal estimated from the homographies has a log error RMS of 0.145 (2 photos) and 0.161
    (3 photos), with a bias of about -0.07.
  - Val coverage of that sd is 93-98 %, so the measured value is used (`src/focal_check.json`).
  - Before this change, a joined picture's focal was wrongly treated as EXIF, with sd 0.02.
  - The 0.08 target was not reached; residual lens distortion and the bias dominate.
- **Video:** in set mode a gallery video can be picked. The phone samples frames, scores sharpness
  (variance of the Laplacian) and the estimated overlap with the last keyframe. It keeps the sharpest frame
  once the overlap reaches the ideal band, then sends those frames as a normal photo set; no video is
  uploaded. For walk videos the total steps are split over the photo pairs by time. The headless test on a
  synthetic pan picked 4 keyframes, none of them blurred, with overlaps of 31-72 % and the end of the
  sweep covered.

- **Live auto sweep:** with "Auto" on in the set viewfinder, the shutter fires by itself after the
  first photo, at most once every 1.2 s. Three things must hold: the overlap with the last photo is in
  the ideal band, the frame is sharp (Laplacian variance at least 0.6 × the median of recent frames), and
  the camera is not moving too fast. "Slow down" appears when two readings in a row show a shift of
  more than 0.35 frame widths per second. In the headless test on frames with known overlap, the
  shutter fired only on the sharp in-band frame: not at 95 % overlap, not on a blurred frame, and not
  twice within 1.2 s.
- **Known-object scale prior refit** (`tools/object_prior_eval.py`):
  - The old prior (×1.531, sd 0.345) was fitted on 4 photos. On 83 train+val photos from 43 sites
    with a person, car, motorcycle, bus or door, its 90 % interval covered only **62.7 %**: too narrow.
  - The refit is a random-effects model, site-balanced, with leave-one-site-out CV. It gives ×1.83 at
    the median scene depth of 8.3 m, slope −0.198 per log depth (farther scenes need less correction),
    and a predictive sd of **0.536** with **94.0 %** coverage.
  - Sizes that rely on this prior therefore get wider ranges. They are now honest. The narrow routes
    remain the walk-along steps, a typed distance, or a known object in the photo itself.

**Limitations:** all set numbers come from simulation built from real photos, not from field sets. The
step length is assumed to be 0.75 m. The auto sweep and video selection have been tested on synthetic
frames only; real phone videos have not been tested.

## 6. Audit (2026-10-16): what a full scan found and fixed

Three independent reviews covered the code, the methodology and the documents, followed by a
data audit. Everything was fixed and then the whole pipeline was re-run.

**Leakage**
- **Site grouping by first Commons category leaked:** about 40–50 usable test photos had a twin
  site in train/val (estimated before the fix). Sites are now merged by title, photo series and visual similarity
  (`tools/link_sites.py`); 230 photos changed split. All models were retrained and all results
  regenerated.
- **Noise correction reached test:** after the merge, 9 noise-corrected labels were in test.
  Their first-look labels were restored.

**Selection and calibration**
- The model was chosen on cross-validation that included the test split. Selection is now
  train+val only.
- The deployed probes used a different C rule than the evaluated ones. There is now one shared
  rule.
- The CV folds were unstratified. They are now stratified and shuffled.
- Shift tests were dominated by one site. Metrics are now site-balanced.

**Metrics and labels**
- The South-Asia tag read search-query text, which tagged UK photos as South Asian. Fixed.
- Subgroup macro-F1 counted absent classes as 0. Fixed.
- The headline accuracy ignored confident answers on unusable photos. End-to-end precision is
  now reported.
- The hardest photos were missing from scoring. They are now measured separately (section 4.4).

**Reproducibility**
- `pred_selected` is now written by `eval.py`.
- The obsolete second selector (`tools/select_model.py`) was removed.
- The CNN replaces the deployed model only when a run finishes.
- Feature caches and CSVs are written atomically, and the file lock is safe.

**Robustness**
- **review.py:** the default queue was empty; the tie view disagreed with the ground-truth vote;
  human re-labels lost the viewpoint; concurrent saves could clash and fast keypresses could skip
  photos; there was no CSRF guard.
- **app.py:** no pixel cap or timeouts; saves happened outside the lock.
- **probe.py:** crashed on a dataset without rejects.
- **split.py:** could rewrite existing splits.
- **Collector:** hint rules were too loose ("Netherlands" matched "net"), `--dry-run` wrote files,
  and EXIF orientation was ignored.
- **All scripts:** non-ASCII titles could crash runs on a Windows console.

**Second audit (final):** --help ran 5 scripts for real (now parsers everywhere); staleness checks
  used file times (now a content fingerprint of split+labels stored in every model); `link_sites.py` would
  have merged all future photos into one site; Openverse query text still tagged South Asia; Pillow's
  pixel cap was really 80 MP; non-deterministic viewpoint ties; stale ensemble predictions could pass; and
  several crash paths (empty thresholds, missing checkpoints, non-object JSON, model errors in the app).
  All fixed; probes, selection and all evaluations re-run (identical test results).

**Documentation and licences**
- Licences and attribution are now documented (`tools/credits.py`).
- README, labeling guide and docstrings were brought in line with the code. design.md is marked
  as the original plan.

## 6b. Dimensions after classification (added 2026-10-26, upgraded the same day)

After the type is predicted, `src/dimension.py` (deep-learning models and orchestration) and
`src/dim_stats.py` (pixel statistics and robust estimators) estimate the size of the landslide. The
approach is hybrid throughout: networks propose, statistics fuse, check and measure. No single
network sets a number on its own.

| Stage | Deep learning | Statistics / geometry |
|---|---|---|
| Outline | CLIPSeg (class prompts); self-trained DINOv2 head; SAM with 6 jittered prompt sets | Excess-Green + sky prior; graph cut (max-flow) with colour and depth-edge terms; lambda by stability selection |
| Depth | Depth Anything V2 and Depth Pro, each with test-time augmentation (TTA); ZoeDepth was tested and dropped (see below) | Shape from both models, scale = consensus of the models; per-pixel inverse-variance fusion; guided filter |
| Scale | OWLv2 known-height objects (person, car, ..., trees weak) | Weighted median, 3-MAD outlier removal; known-object prior (x1.53, n=4); field calibration (leave-one-site-out) |
| Geometry | - | RANSAC plane; percentile extents; view-angle-corrected area; contour perimeter |
| Volume | - | Area-volume laws combined with a thin-plate-spline pre-failure surface (CV-chosen smoothing), weighted by inverse variance |
| Grains | SAM instances at zoom 1 and zoom 2 | Convex-hull b-axis; Kellerhals-Bray grid-by-number; truncated log-normal MLE; bootstrap CI; zoom agreement |
| Uncertainty | - | Monte Carlo over outline variant x scale x volume priors: P5-P95 and size-class probabilities |
| Checks | - | TTA stability; model agreement; haze rho; texture-frequency slope; plane inliers; object-vs-depth; Corominas H/L |
| Class doubt | Classifier's top two classes | Measured under both; probability-weighted geometric mean |
| Several photos of one site | - | `--site`: median across photos; range combined with the between-photo spread (`fuse_site`) |

**Synthetic recovery tests (known answers, CPU):**
- truncated log-normal D50: 0.095 against a true 0.100;
- graph cut IoU 1.00, against 0.55 for a plain threshold on the same noisy probabilities;
- pre-failure pit volume: 18.4 against about 15.7 m³;
- texture-depth slope: 0.98 against an expected 1 (for textures above about 0.06 cycles/px; lower
  frequencies are biased low, so the check only flags slopes outside 0.3-2.0).

**Evaluation history (40 photos x 9 versions, plus 12 multi-photo sites; `results/dimension_eval_v1..v4.md`).**
Stability is the median |log change| under each photo edit (0.10 is about a 10 % change); lower is better.

| | v1 (CLIPSeg+SAM) | v2 (3 depth models, median scale) | v3 (ZoeDepth dropped) | v4 (final) |
|---|---|---|---|---|
| Area under blur / JPEG / noise / tilt | 0.37 / 0.48 / 0.24 / 0.27 | 0.49 / 0.40 / 0.27 / 0.31 | 0.32 / 0.42 / 0.25 / 0.23 | 0.31 / 0.42 / 0.25 / 0.25 |
| Area under mirror / darker / downscale | 0.20 / 0.15 / 0.25 | 0.05 / 0.09 / 0.15 | 0.06 / 0.09 / 0.17 | 0.06 / 0.09 / 0.16 |
| Volume under blur / noise / downscale / tilt | 0.44 / 0.30 / 0.29 / 0.32 | 0.60 / 0.74 / 0.31 / 0.36 | 0.38 / 0.46 / 0.29 / 0.33 | 0.35 / 0.44 / 0.22 / 0.28 |
| Trust high / medium / low | 12 / 75 / 12 % | 0 / 72 / 28 % | 30 / 42 / 28 % | 30 / 42 / 28 % |
| Camera distance 2-300 m | 100 % | 80 % | 100 % | 100 % |
| Same-site log-sd: length / width / area | - | 0.50 / 0.32 / 0.77 | 0.40 / 0.52 / 0.86 | 0.40 / 0.45 / 0.80 |
| 90 % interval width, log(P95/P5), length | - | - | 1.3-3.1 | 1.37-1.44 |
| Seconds per photo (RTX 4060) | ~20 | 30 | 23 | 25 |

**Decisions, each backed by evidence:**
- **ZoeDepth dropped (v3).** It read 4-6x closer than the other two models (median largest-pair ratio 6.25 on 40 photos), was contradicted by the known-size objects, and the empirical "depth error" it produced (log-sd 1.9) was bias, not noise.
- **Known-object scale prior x1.53 (v4).** All 9 photos where OWLv2 found an object of known height said the ensemble reads too close (factors 1.04-4.36; p about 1/512 if unbiased). Only the 4 tight-height objects set the factor (person x1.27 and x1.85, motorcycle x1.92, car x1.04); its predictive log-sd is 0.345. It is applied only when a photo has no typed distance, no object of its own and no field calibration (`models/dimension_object_prior.json`). Caveat: it is estimated on the same evaluation photos, but it is one global number and does not affect the stability metrics, which do not depend on the overall scale.
- **On/off switches replaced by smooth weights (v3-v4).** These are the pre-failure surface weight (relief-to-noise ratio) and the grain-size floor on thickness, which was removed. Volume stability improved, e.g. blur 0.60 -> 0.35 and downscale 0.31 -> 0.22.
- **Remaining weak spots.** Heavy JPEG and noise still move the area by 25-42 %; volume under noise (0.44) is worse than v1 (0.30), because v1 had no surface-based volume. Same-site spread (width about 1.6x, area about 2.2x) is partly real: each photo sees a different part from a different distance. **Report ranges, and fuse several photos of an important site (`--site`).**

**Assumed values (literature-based, not fitted).** These cannot be fitted without measured sizes;
`tools/calibrate_dimensions.py` replaces the scale-related ones once field data exist.

| Value | Used | Source |
|---|---|---|
| Single-model metric depth error | log-sd 0.25, plus the per-photo disagreement between the models | zero-shot AbsRel of 0.2-0.3 outdoors |
| Focal-length error | EXIF 0.02, Depth Pro 0.15, default 0.25 (log-sd) | - |
| Object heights | person 1.70, car 1.50, motorbike 1.10, truck 3.0, bus 3.2, door 2.1, pole 9, tree 15 (log-sd 0.06-0.5) | typical dimensions |
| Guzzetti law | V = 0.074 A^1.45, log-sd 0.6 | Guzzetti et al. 2009 |
| Deposit thickness | debris flow 1 m (0.3-3), earthflow 2 m (0.5-5), talus 0.5 m (0.2-2) | typical ranges |
| Mobility | log10(H/L) = -0.047 - 0.085 log10 V, scatter about 0.2 | Corominas 1996 |
| Size classes | Fell 1994 | - |

Data-driven, not hand-set:
- graph-cut lambda (stability selection, per photo);
- thin-plate-spline smoothing (5-fold CV, per photo);
- head regularisation C and self-training round (val split);
- the head's fusion weight (from val AUC);
- the known-object scale prior (OWLv2 evidence);
- the calibration factor (leave-one-site-out CV).

**Leakage.** The outline head is fitted on the train split, and C and the round are chosen on val.
The test split is never read. Splits are separated by site, and the train/val site overlap is
asserted to be zero. The calibration is validated leave-one-site-out. No other part of the dimension
pipeline is fitted on these photos.

**Evaluation (`results/dimension_eval*.md`, no ground-truth sizes):**
- outline availability;
- stability under 8 phone-like edits;
- the depth checks;
- plausibility;
- interval widths;
- same-site consistency: different photos of the same landslide must agree.

**Limitations**
- Absolute scale is unverified until field measurements exist.
- Volumes and thickness priors carry a factor of about 2-3x.
- Grains are resolvable mainly in close-up deposit photos.
- The phone is assumed upright unless the tilt sensor is used.
- Heavy JPEG compression or blur changes sizes by 20-40 %.

## 7. Limitations and next steps

1. **Human labels for the test split:** see "Label caveat" in section 2. This is the largest open
   item.
2. **Real local phone photos:** only 20 phone photos are in the test split. Taking 30–50 local
   photos (`src/import_local.py` + `review.py`) is the truest test of the app.
3. **The photo check:** about 17% of good photos are refused. More rejected and accepted
   examples taken by engineers would help.
4. **The weakest classes:** `translational_slide` and `earth_mud_flow` need more data.
5. **Decision support only:** the app sees one photo, so cracks or seepage outside the frame are
   invisible to it. A geologist must confirm.

## Reproduce

Follow the README workflow table, steps 1–8. Each script's top docstring documents its options.
After any interruption, run `tools/check_integrity.py`; it must print ALL OK.
