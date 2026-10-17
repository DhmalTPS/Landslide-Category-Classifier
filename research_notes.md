# Research notes: landslide type classification

Status (2026-10-15): objective confirmed. Landslide type classifier, ground photos taken from
downhill, dataset probably self-built, RTX 4060 8 GB. Design for both dataset scenarios: `design.md`.
Pipeline built and smoke-tested end to end (see `README.md`). Final state and findings: section
"Findings from the full build" below and `report.md`.

## Classification terminology

Standard reference: Hungr, Leroueil & Picarelli (2014), *The Varnes classification of
landslide types, an update*, Landslides 11:167-194, doi:10.1007/s10346-013-0436-y.
PDF: `papers/Hungr2014_Varnes_classification_update.pdf`; tables: `papers/Hungr2014_tables_extracted.txt`.

- Two axes: **movement type** x **material**.
- Movement types (6): Fall, Topple, Slide, Spread, Flow, Slope deformation.
- Material: Rock vs Soil (clay, mud, silt/sand/gravel, debris, peat, ice).
- 32 types in total. Many depend on properties not visible in images (clay sensitivity,
  plasticity, saturation, velocity), so the full set is not realistic for an image classifier.

Practical label set used by LandslideBench (7 types + non-landslide):
rock fall, rock slide, earth slide, debris flow, earth flow, mud flow, mudslide.

Our classes (ground view, downhill): fall_topple, rotational_slide, translational_slide,
debris_flow, earth_mud_flow, no_landslide (definitions in `src/common.py`).

## VLM zero-shot vs trained model

From LandslideBench (`papers/LandslideAgent2026_LandslideBench.pdf`, arXiv 2606.18661),
Google Earth RGB, 2.4-4.8 m/px, 2,130 image-mask-text triplets:

| Approach | Fine-grained type accuracy |
|---|---|
| Qwen3-VL-8B zero-shot | 19.18% (chance ~14%) |
| LandslideVLM (LoRA fine-tuned VLM) | 52.05% |
| ResNet-50 | 62.21% |
| CSPDarkNet | 73.06% |
| Swin Transformer | 73.97% |
| EfficientNet | 76.71% |
| ConvNeXt | 79.00% |

## Datasets

| Dataset | Labels | Type labels? |
|---|---|---|
| LandslideBench | 7 types + negatives, masks, text | Yes, but overhead imagery (not our viewpoint) |
| Bijie (Guizhou) | 770 landslide / 2,003 non-landslide | No |
| Landslide4Sense | pixel masks | No |
| LMHLD | pixel masks, 5 sensors, 7 areas | No |
| Debrisflow23 | debris flow vs normal | Binary only |
| ESSD 2024 coseismic | mapping, 9 earthquake events | No |

No public ground-level landslide-type dataset found -> Scenario B (build our own).

## Observations from our own runs (2026-10-15)

- **Qwen2.5-VL-7B, 4-bit, RTX 4060 8 GB:** 5.5 GB after load, 6.0 GB peak; ~1.3 s per
  image (letter-probability classification), ~6 s with a generated description.
- **Leading prompts cause hallucination:** asking for "scarp shape, material, channel" made the
  model claim those features in every photo, and its description contradicted its own label
  (classified rotational_slide, described a debris flow). The description prompt is now neutral.
- **VLM over-predicts no_landslide:** 90 of the first 156 photos (mostly from Rockfall
  categories). Some are correct (rock sheds, protection nets: the category hint was wrong);
  some are misses: a rockfall debris cone at the foot of West Bay cliff, Dorset, was called
  no_landslide at 74%. It misses small deposits in wide shots.
- **Commons category hints are noisy:** many photos in "Rockfalls" are mitigation structures.
  Protection/mitigation subcategories now map to no_landslide; art/maps/diagrams are excluded.
- **Wikimedia rate limits:** the media servers return 429 with Retry-After: 600 for some image
  requests (a shared institute IP may add to this). The collector skips them and retries once.

## Findings from the full build (2026-10-16; details in report.md)

- **Data sources:** share rejected as unusable: Commons category trees 82%, Openverse/Flickr 86%,
  Commons title searches 62% (best yield of ground photos). Wikimedia rate-limits *original* files hard (429, Retry-After 600)
  but not scaled thumbnails: request a thumbnail width below the original.
- **Frozen features win at this data size** (~1,300 usable photos, noisy AI labels): single linear probes
  0.64-0.74 test macro-F1, their ensemble 0.78, fine-tuned ConvNeXt-Tiny 0.67, Qwen2.5-VL zero-shot 0.33
  (final, after the site-merge audit; report.md).
- **The three-backbone ensemble generalised best** (DINOv2-base + DINOv2-large + SigLIP2, chosen by
  site-balanced CV + shift tests on train+val); single models and weaker pairs do not always beat each other.
- **Region shift is the main risk:** trained without South Asia -> 0.58 site-balanced macro-F1 there
  (vs 0.71 CV), accuracy 0.74 vs 0.77.
- **Site grouping by Commons category was leaky** (same place under different categories);
  `tools/link_sites.py` merges sites by title stems, photo series and DINOv2 similarity.
- **Selective prediction:** see report.md for CV vs test accuracy at the 0.6 threshold.
- **Selection hygiene:** choose models on train+val only; an early run that used test photos in CV was redone.

## Papers in `papers/`

| File | What it is |
|---|---|
| Hungr2014_Varnes_classification_update.pdf | The terminology standard (32 types) |
| LandslideAgent2026_LandslideBench.pdf | Type-classification benchmark; VLM vs CNN numbers |
| LMHLD2025_multisource_landslide_dataset.pdf | Large multi-sensor detection dataset |
| Ensemble2023_global_landslide_detection.pdf | Global detection with deep ensembles |
| Multisource2025_landslide_detection_mapping.pdf | Detection across sensors/regions |
| ESSD2024_global_coseismic_landslide_dataset.pdf | Earthquake-triggered landslide dataset (21 MB) |
| SciRep2025_debris_flow_detection.pdf | Debris-flow image recognition (Debrisflow23) |
| SciRep2025_DeepLabv3plus_landslide_detection.pdf | Segmentation with improved DeepLabv3+ |
| VLM_RemoteSensing_review_2023.pdf | Survey of VLMs in remote sensing |
