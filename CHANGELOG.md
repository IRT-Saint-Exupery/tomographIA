# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — Unreleased

First public release.

### 3-D pipeline

- `tomographia.segmentation` — `run_inference` (3-D UNet on a TIF stack:
  overlapping patches averaged as probabilities, validation threshold read
  from the checkpoint, optional small-component removal, strict checkpoint
  loading via `load_segmentation_checkpoint`); `SegmentationTrainer`
  (BCE / class-balanced focal + Dice, F1-optimal threshold selected on
  validation, best epoch by validation loss, single final test evaluation,
  optional TTA and AMP, `on_log` callback); `PatchGenerator` /
  `SegmentationDataset` (HDF5 patch cache keyed on its settings, per-volume or
  per-patch splits); `build_pos_sampler`, `estimate_pos_weight`,
  `AugmentedDataset`, `make_synthetic_volume`; volume-level normalisations.
- `tomographia.components` — `build_cc_mask` (closing / dilation used only to
  group fragments; labels cover the predicted voxels) and
  `compute_descriptors`: descriptors in physical units, principal axes from
  the covariance of the voxel boxes, IBSI elongation and flatness, mesh-based
  sphericity (≤ 1), bounding-box ratios, log volume, intensity statistics;
  `CANONICAL_FEATURES`.  `close_image` / `cc_report` (cc3d) for a size /
  bounding-box summary.
- `tomographia.classification` — `fit_classifier` (imputer + scaler +
  class-balanced logistic regression, threshold from out-of-fold training
  predictions, random or volume-level hold-out), `classify_defects_logistic`,
  `build_classified_volume`, `cluster_defects` / `make_cluster_volume`
  (KMeans), `classify_defects` dispatch.
- `tomographia.reporting` — `compute_yield_report` (intensity-window defect
  density) and `generate_pdf_report` (takes the outputs of the functions
  above directly).

### 2-D pipeline

- `tomographia.patches` — `build_patch_index`, `split_patches` (per folder or
  grouped by specimen), `PatchDataset`, `PatchTransform`.
- `tomographia.classification` — `ResNetPatchClassifier` (single-channel
  ResNet-18…152 with a differentiable `features()` embedding) and
  `PatchClassifierTrainer`.
- `tomographia.prototypes` — prototype head and network on the CNN embedding,
  expert-anchor medoid initialisation, composite prototype loss, training
  loop, diagnostics and plots.
- `tomographia.evaluation` — F1-optimal threshold, point metrics, ROC / PR-AUC,
  ECE, Brier, plain and cluster bootstrap, temperature scaling and the
  calibrated evaluation protocol.
- `tomographia.slice_inference` — whole-slice / whole-volume prediction,
  stacking, sparse-voxel removal, isosurface mesh, slice overlay.
- `tomographia.labelling` — patch-labelling session, terminal CLI
  (`tomographia-label`) and Streamlit app (`tomographia-label-app`).

### Infrastructure

- Shared natural-order slice reading for every module.
- Optional extras: `torch`, `train`, `vision`, `report`, `cc3d`, `labelling`, `dev`.
- Test suite (runs with and without torch), GitHub Actions CI, `CITATION.cff`.
