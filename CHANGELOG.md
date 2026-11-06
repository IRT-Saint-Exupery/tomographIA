# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-06

First public release.

### Packaging

- Versions are git tags (`vX.Y.Z`). Pushing a tag runs `release.yml`: build
  the wheel and sdist from the tag, test the installed wheel in clean
  environments, then attach both to a GitHub Release (PyPI publishing ready
  but off by default).
- The version is defined once, in `tomographia.__version__`.
- `pyproject.toml` declares the licence as an SPDX expression (`MIT`), the
  form setuptools requires from 2027-02-18.

### 3-D pipeline

- `tomographia.segmentation`: `run_inference` (3-D UNet on a TIF stack:
  overlapping patches averaged as probabilities; threshold, preprocessing,
  `min_size` and flip TTA read from the checkpoint; strict checkpoint loading
  via `load_segmentation_checkpoint`); `SegmentationTrainer` (BCE /
  class-balanced focal + Dice, F1-optimal threshold selected on validation,
  best epoch by validation loss, single final test evaluation, probability
  TTA, AMP, `on_log` callback, checkpoints with their preprocessing
  metadata); `PatchGenerator` / `SegmentationDataset` (HDF5 patch cache keyed
  on its settings and input files, per-volume or per-patch splits, voxels of
  overlapping patches counted once, labels matched on the slice number);
  `build_pos_sampler`, `estimate_pos_weight`, `AugmentedDataset`,
  `make_synthetic_volume`; volume-level normalisations.
- `tomographia.components`: `build_cc_mask` (closing / dilation used only to
  group fragments; labels cover the predicted voxels) and
  `compute_descriptors`: voxel size given with its unit, all descriptors in
  mm, principal axes from the covariance of the voxel boxes, IBSI-style
  elongation and flatness, mesh-based sphericity (≤ 1), bounding-box ratios,
  log volume, intensity statistics, `touches_border` flag; `CANONICAL_FEATURES`.  `close_image` / `cc_report` (cc3d) for a size /
  bounding-box summary.
- `tomographia.classification`: `fit_classifier` (imputer + scaler +
  class-balanced logistic regression, threshold from out-of-fold training
  predictions grouped by volume when holding out volumes, random or
  volume-level hold-out, feature ranges and missing values recorded),
  `classify_defects_logistic` (warns on features outside the training range),
  `build_classified_volume`, `cluster_defects` / `make_cluster_volume`
  (KMeans), `classify_defects` dispatch.
- `tomographia.reporting`: `compute_yield_report` (intensity-window defect
  fraction) and `generate_pdf_report` (takes the outputs of the functions
  above directly).

### 2-D pipeline

- `tomographia.patches`: `build_patch_index`, `split_patches` (per folder or
  grouped by specimen), `PatchDataset`, `PatchTransform`.
- `tomographia.classification`: `ResNetPatchClassifier` (single-channel
  ResNet-18…152 with a differentiable `features()` embedding) and
  `PatchClassifierTrainer`.
- `tomographia.prototypes`: prototype head and network on the CNN embedding,
  expert-anchor medoid initialisation, composite prototype loss, training
  loop, diagnostics and plots.
- `tomographia.evaluation`: F1-optimal threshold, point metrics, ROC / PR-AUC,
  ECE, Brier, plain and cluster bootstrap (guarded for few groups), bounded
  temperature scaling (skipped on separable validation) and the calibrated
  evaluation protocol (threshold and rankings on scaled logits).
- `tomographia.slice_inference`: whole-slice / whole-volume prediction with
  the calibration temperature,
  stacking, sparse-voxel removal, isosurface mesh, slice overlay.
- `tomographia.labelling`: patch-labelling session, terminal CLI
  (`tomographia-label`) and Streamlit app (`tomographia-label-app`).

### Infrastructure

- Shared natural-order slice reading for every module.
- Optional extras: `torch`, `train`, `vision`, `report`, `cc3d`, `labelling`, `dev`.
- Test suite (runs with and without torch), GitHub Actions CI, `CITATION.cff`.
