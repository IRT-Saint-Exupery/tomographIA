# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2024-12-01

Initial open-source release.  Extracted and cleaned from the internal
`tomographIA` research codebase.

### Added

**Segmentation** (`tomographia.segmentation`)
- `run_inference()` — importable 3-D UNet inference function; previously only
  ran as a `__main__` script.  Supports CPU and CUDA, optional heatmap saving,
  per-batch progress callback, and big-endian TIF auto-conversion.
- `PatchedVolumeBig` / `SingleVolumeBig` — lazy overlapping-patch datasets for
  large volumes; in-memory path only (disk-backed h5py path dropped).
- I/O utilities: `get_tif_files`, `get_volume_as_numpy_array`,
  `apply_normalization_by_name`, and four normalization strategies.

**Components** (`tomographia.components`)
- Pipeline 2 (default, skimage): `build_cc_mask`, `compute_descriptors`,
  `load_grayscale_volume`.  `compute_descriptors` produces the 12 canonical
  classifier features: `sphericity`, `anisotropy`, `flatness`, `elongation`,
  `fill_frac`, `r1`, `r2`, `ar_zx`, `ar_zy`, `ar_yx`, `int_std_rel`,
  `log_voxels`.
- Pipeline 1 (legacy, cc3d + scipy): `close_image`, `cc_report`, `dict_to_df`.

**Classification** (`tomographia.classification`)
- `fit_classifier()` — trains a logistic-regression bundle from descriptor
  CSVs; handles train/test split, optimal-threshold selection on training
  predictions only (no leakage), and `feats_avail` tracking.
- `classify_defects_logistic()` — inference from a saved bundle.
- `build_classified_volume()` — maps CC label arrays to uint8 class volumes.
- `classify_defects()` — dispatch function supporting `"logistic"` and
  `"kmeans"` methods.
- `cluster_defects` / `make_cluster_volume` — unsupervised KMeans path.
- `SimpleNN`, `MulticlassNN`, `HeatmapEmbedding` — neural-network
  architectures for heatmap-embedding-based classification (inference only;
  see Notes).

**Reporting** (`tomographia.reporting`)
- `compute_yield_report()` — per-slice and overall pore/crack/total yield
  from a classified volume and raw TIF directory.
- `generate_pdf_report()` — PDF defect report with orthogonal projections and
  bounding-box overlays.

**Models** (`tomographia.models`)
- `Unet3D`, `Unet3d_classifier` — 3-D U-Net for binary segmentation and
  patch classification.
- `select_model()` — factory supporting `unet3d`, `unet3d_classifier`,
  `swin_unetr` (monai), `unetr` (monai), and resnet variants.

### Fixed

- `utils.py` self-import: original `tomographIA.data_processing.utils`
  imported itself on line 11, causing a circular import.  Fixed by calling
  normalisation functions directly.
- `report_generation()` `opt` global: function body referenced `opt.data_path`
  (a CLI argparse namespace), causing `NameError` when called
  programmatically.  Replaced with an explicit `data_path` parameter.
- `check_mode.py` bit-depth conversion: missing `mkdir` before saving
  converted TIFs, and a per-file flag that only reflected the last file
  processed.  Fixed with `converted_dir.mkdir()` and a single
  `needs_conversion` accumulator.
- `neural_network_dict()` key casing: was `"multiclassNN"` (lowercase m),
  inconsistent with `"SimpleNN"`.  Changed to `"MulticlassNN"`.
- `find_heatmap_box()` triplication: was duplicated across three modules.
  Consolidated in `classification/embeddings_nn.py`.
- Optimal-threshold sentinel off-by-one: `precision_recall_curve` returns
  `n_thresholds+1` values for precision/recall.  The sentinel (P=1, R=0)
  is now trimmed before argmax to avoid a latent `IndexError` on degenerate
  data.  This does not change threshold selection on real data.
- `assert` replaced with `ValueError` in `SingleVolumeBig._load()`.
- Missing `Optional`/`Callable` type annotations in `compute_descriptors`.

### Removed

- `mae/` directory — excluded; CC-BY-NC 4.0 license is incompatible with MIT.
- MLflow experiment tracking — deployment infrastructure, not part of the
  library API.
- Disk-backed h5py patch dataset (`PatchedVolume` / `SingleVolume`).
- CLI `__main__` blocks converted to importable functions.
- Module-level side-effects (unconditional `main_segmentation_evaluation()`
  call on import).

### Notes

- **NN classification (SimpleNN / MulticlassNN)**: these architectures
  operate on 512-d heatmap embeddings, not on tabular descriptor features.
  No training code existed in the original codebase for these models; they
  are included for inference compatibility only.  Use `fit_classifier()` for
  a fully trainable classification pipeline.
- `HeatmapEmbedding` has a lazy `fc1` layer.  For reproducible inference,
  save and reload both the `HeatmapEmbedding` and `SimpleNN`/`MulticlassNN`
  state dicts together — the original scripts did not do this.
- Pipeline 1 (`close_image`) requires the optional
  `connected-components-3d` package (`pip install tomographia[cc3d]`).
