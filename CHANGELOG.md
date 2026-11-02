# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

**2-D patch pipeline** — ported from the internal `fdefects` development line.

- `tomographia.patches` — `build_patch_index` (sorted, multi-folder),
  `split_patches` (per-folder 80/10/10, or **grouped** by
  specimen/volume so no group spans two splits), `PatchDataset` (optional
  global index for anchors), `PatchTransform` (flips, ±15° rotation with
  reflected borders, `(x/255 − 0.5)/0.5` normalisation; pure torch, replaces
  albumentations).
- `tomographia.classification.ResNetPatchClassifier` — single-channel
  ResNet-18/34/50/101/152 with differentiable `features()`; state dicts of the
  original `ResNet*_classifier` classes load unchanged.
  `PatchClassifierTrainer` / `PatchTrainConfig` — class-weighted BCE, AUC model
  selection, F1-optimal threshold, revert-to-best + LR decay.
  `models.select_model("resnet…")` now returns this class.
- `tomographia.prototypes` — `PrototypeHead`, `PrototypeNetwork`,
  `build_prototype_network`; expert-anchor medoid initialisation
  (`extract_embeddings`, `fit_embedding_scaler`, `prototypes_from_anchors`);
  composite `prototype_loss` with `ProtoLossWeights`; `train_prototype_network`
  with `ProtoTrainConfig`; diagnostics
  (coalescence, drift, anchor assignment accuracy, nearest examples /
  learned-anchor purity, per-prototype confusion) and plotting helpers.
- `tomographia.evaluation` — F1-optimal threshold, point metrics, ROC/PR-AUC,
  ECE, Brier, plain and cluster bootstrap CIs, `fit_temperature` and the
  `evaluate_calibrated` protocol.
- `tomographia.slice_inference` — tiling, batched slice/volume prediction for
  classifiers and prototype networks, stacking, sparse-voxel removal,
  isosurface mesh, slice overlay plot.
- `tomographia.labelling` — patch-labelling session, terminal CLI
  (`tomographia-label`) and Streamlit app (`tomographia-label-app`).
- Extras `vision` (torchvision, tqdm) and `labelling` (streamlit, matplotlib).

**Segmentation training** (`tomographia.segmentation`)
- `SegmentationTrainer` — self-contained training loop for binary 3-D
  segmentation; no external base-class dependency.  Features: combined
  BCE/Focal + Dice loss, per-epoch F1-maximising threshold search, optional
  test-time augmentation (flip mode), per-volume micro-metrics, and an
  `on_log` callback for MLflow / W&B integration (no hard MLflow dependency).
- `SoftDiceLoss`, `FocalLoss` — self-contained loss modules.
- `build_pos_sampler`, `estimate_pos_weight` — class-imbalance utilities.
- `AugmentedDataset` — wraps any segmentation dataset with TorchIO
  transforms (lazy import; no torchio required unless a transform is passed).
- `PatchGenerator`, `SegmentationDataset` — HDF5-backed patch extraction and
  dataset for training (h5py imported lazily; no h5py required for inference).

### Fixed

**2-D pipeline (relative to the internal source)**

- **Prototype head fed the classifier logit** (`PrototypeNetwork`): the source
  `ProtoResNet50.forward` called the full `ResNet50_classifier`, including the
  trained `fc` layer, so the head received a `[B, 1]` logit that broadcast
  silently against the 2048-d scaler, while the scaler, prototypes and anchor
  medoids were computed from the 2048-d `extract_features` embedding.  The
  head now receives `backbone.features(x)` and rejects embeddings of the
  wrong dimension; a regression test covers the fix.
- **F1-optimal threshold off by one**: `precision_recall_curve` outputs were
  aligned with `prec[1:], rec[1:]`, pairing the F1 of threshold `i+1` with
  `thresholds[i]`; the selected threshold was one score step below the
  optimum (all trainers and evaluation scripts).  Now `prec[:-1], rec[:-1]`.
- **Temperature scaling** optimised `T` directly (could turn negative and was
  silently reset to 1); now optimises `log T`.
- **Non-reproducible splits**: patch files were listed in `os.listdir` order.
- **ResNet trainer**: the `ReduceLROnPlateau` scheduler was never stepped
  (gated on an attribute the base trainer never set) — removed;
  `patience_stop` counted all epochs instead of epochs without improvement —
  replaced by `early_stop_patience`; `--augmentation type=bool` enabled
  augmentation for any value; `train_fraction < 1` mis-aligned prediction ids.
- **ResNet-18/101/152** ignored `pretrained` and always loaded ImageNet
  weights; deprecated torchvision `pretrained=` API replaced by `weights=`.
- **Prototype training**: `raise("…")` raised a `str` (itself a `TypeError`);
  `collect_val_logits` read a non-existent `"logits"` key (removed; use
  `predict_prototype_network(...)["logit"]`).
- **Labelling**: CLI read `scale`/`offset` from the argument parser instead
  of the parsed arguments, and never passed them to the session; model
  assistance called `select_model` with wrong keyword arguments; the patch and
  context panels were displayed upside down.
- **Isosurface**: vertex scaling mixed axis order for anisotropic factors;
  empty volumes now return an empty mesh instead of raising.

**3-D pipeline**

- **AMP/GradScaler dead code** (`SegmentationTrainer`): original
  `train_complex_oversample.py` created a `GradScaler` at line 757 but never
  passed it to `training_step`, so AMP silently never activated.  AMP is now
  functional via `use_amp=True`; the default is `False` (full FP32).
- **`opt.min_size` global reference** (`SegmentationTrainer`): original
  `validation_epoch_end` referenced the module-level argparse namespace `opt`,
  causing `NameError` when called outside `__main__`.  Promoted to constructor
  parameter `min_size`.
- **Patch loop bounds** (`PatchGenerator._Patcher`): port initially iterated
  over the stride-padded volume shape, which for volumes whose dimensions are
  exactly divisible by the stride generates one extra patch per axis from the
  padding region — a patch with the correct shape that passes the size guard
  and is spuriously saved.  Fixed to match the source: iterate over the
  hard-padded shape, consistent with `Patcher.constant_padded_volume_shape` in
  the original `patchers.py` L105.
- **`best_treshold` metric key** (`SegmentationTrainer`): original source
  (`train_complex_oversample.py` L497) had a deliberate spelling error
  `"best_treshold"` preserved for backward compatibility with internal MLflow
  dashboards.  No such dashboards exist in `tomo_open`; corrected to
  `"best_threshold"`.
- **`gradient_accumulation_steps` silent no-op**: parameter was accepted by
  the original `simple.Trainer` constructor but the accumulation loop was
  never implemented — every step called `optimizer.step()` unconditionally.
  Parameter removed to avoid false confidence.
- **Test-set logit buffer leak** (`SegmentationTrainer`): original
  `evaluation_epoch_end` (via `simple.Trainer`) only averaged batch losses
  without processing or clearing `_val_logits`/`_val_labels`/`_val_volids`,
  contaminating the next epoch's validation metrics.  `evaluation_epoch_end`
  now delegates to `validation_epoch_end()` which processes and clears all
  buffers.
- **Per-patch RNG global state** (`PatchGenerator`): replaced
  `np.random.seed(seed)` (global mutation) with
  `np.random.RandomState(seed)` (instance-local) for reproducible,
  side-effect-free splits.
- **`h5py`/`cv2`/`mlflow` crash-on-import chains**: all three were
  module-level imports in the original that would crash on import if the
  optional package was absent.  Made lazy (imported inside the methods that
  need them) or replaced with an injection point (`on_log` callback replaces
  mlflow).

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
