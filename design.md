# Design: ground-level landslide type classifier

> **Original plan (2026-10-15), kept for the record.** Several items changed during the build:
> labels are coarse only (no fine Hungr label per photo), flows were not merged (the app gives a
> 'flow' family answer instead), the deployed model is an ensemble of frozen-feature probes chosen
> by `tools/robustness_eval.py` (ConvNeXt is a comparison), the VLM reads option-letter
> probabilities instead of JSON, sources are Wikimedia Commons + Openverse/Flickr (no USGS), and
> the layout is `data/raw/images/` + `data/metadata.csv`. The implemented system is described in
> `README.md` and `report.md`.

Confirmed (2026-10-15):
- Task: landslide type classification.
- Input: ground photos taken from downhill.
- Dataset: probably none provided; design for both cases.
- Hardware: RTX 4060 Laptop, 8 GB VRAM.

## Label set (both scenarios)

Classes chosen to be visible from a downhill ground view. Based on Hungr et al. 2014 (see `research_notes.md`).

| Class | Hungr 2014 types it covers | Visual cues from downhill |
|---|---|---|
| `fall_topple` | rock/ice fall, block/flexural topple | steep rock face, detached blocks, talus/boulders at toe or on road |
| `rotational_slide` | rock/clay/silt rotational slide (slump) | curved concave main scarp, back-tilted blocks, tilted trees, bulging toe |
| `translational_slide` | rock planar/wedge slide, gravel/sand/debris slide | planar or elongated shallow scar, exposed bedding/joint plane, sheet of displaced material |
| `debris_flow` | debris flow, debris flood, debris avalanche | gully/channel, levees, unsorted boulders in mud matrix, fan deposit |
| `earth_mud_flow` | earthflow, mud flow | fine-grained lobe/tongue, wet, smooth or hummocky surface |
| `no_landslide` | (negative class) | stable slopes, road cuts, quarries, construction |

Every image is stored with its fine label (the Hungr type) as well as the coarse class.
If data is too thin, `debris_flow` and `earth_mud_flow` merge into `flow`.

## Scenario A: a labeled dataset is provided

1. Map the provided labels to the table above and audit them (class counts, duplicates, leakage).
2. **Main model:** fine-tune ConvNeXt-Tiny (alt: EfficientNetV2-S), ImageNet weights, 224-384 px, AMP, batch 16-32.
3. **Baseline:** Qwen2.5-VL-7B zero-shot in 4-bit (~6 GB), constrained JSON output.
4. Same test split and metrics for both (`eval.py`).

## Scenario B: we build the dataset (expected)

1. **Collect:**
   - Wikimedia Commons (CC-licensed). Direct file counts under `Category:Landslides by type`: Debris flow 37, Mudslides 64, Blockslides 27, Rock slides 21, Mudflows 7, Solifluctions 27, Soil creep 23, Peat slides 8. Subcategories add more.
   - USGS Denver Library photo collection (public domain).
   - Your own field photos (best match for the downhill viewpoint).
   - Keep `source`, `license`, `url`, `event/site id` for every image in `data/metadata.csv`.
2. **Pre-label with the VLM:** run Qwen-VL on every image and ask for JSON `{class, cues, confidence}`. Humans then verify every label. This is where the VLM earns its place.
3. **Model, small data (~30-150 images per class):** frozen DINOv2 embeddings + logistic regression / kNN. Robust with little data and trains in seconds.
4. **Model, as data grows (>150 per class):** fine-tune ConvNeXt-Tiny as in Scenario A.
5. **Split by event/site, not by image.** Photos of the same landslide must not appear in both train and test.

## Evaluation (both scenarios)

- Metrics: accuracy, macro-F1, per-class recall, confusion matrix.
- Comparison: VLM zero-shot vs DINOv2 linear probe vs fine-tuned CNN.
- Error analysis: which classes get confused (expect rotational vs translational).

## Environment

- Python 3.11 venv in `.venv/`. CUDA build of PyTorch (none installed yet).
- Packages: timm, transformers, accelerate, bitsandbytes, scikit-learn.

## Planned layout

```
papers/            research papers
data/raw/<class>/  images
data/metadata.csv  source, license, url, site_id, fine_label, coarse_label
src/collect.py     Commons/USGS downloader with license capture
src/vlm_label.py   Qwen-VL zero-shot / pre-labeling (4-bit)
src/probe.py       DINOv2 linear probe
src/train_cnn.py   ConvNeXt fine-tune
src/eval.py        shared metrics + confusion matrix
```
