
<p align="center">
  <img src="viz/logo_tomoIA.png" alt="tomographIA logo" width="420"/>
</p>

Open-source Python library for 3-D CT defect analysis: segmentation, connected-component characterisation, classification (logistic regression or KMeans), and yield reporting.

This code originates from the internal research project "tomographIA" at IRT Saint Exupéry. We welcome potential collaborations. Please feel free to contact us.

---

## Features

- **Segmentation** — importable `run_inference()` API around a 3-D UNet; overlapping-patch inference with `PatchedVolumeBig`
- **Connected components** — `build_cc_mask` + `compute_descriptors` producing 12 canonical geometric features per component
- **Logistic classifier** — `fit_classifier()` trains a scikit-learn bundle with no threshold leakage; `classify_defects_logistic()` runs inference from saved bundles
- **KMeans clustering** — unsupervised alternative via `cluster_defects()`; no labelled data required
- **Yield reporting** — per-slice and overall pore/crack defect-density metrics from `compute_yield_report()`
- **PDF reports** — orthogonal-projection defect overlays via `generate_pdf_report()`
- **Segmentation training** — `SegmentationTrainer` trains the 3-D UNet from
  your own HDF5 patch cache; `PatchGenerator` handles patch extraction and
  80/10/10 splitting

---

## Installation

tomographia is not yet published to PyPI. Install directly from a clone:

```bash
git clone <repo-url> tomographia
cd tomographia
pip install -e ".[dev]"
```

### Optional extras

| Extra | Installs | Required for |
|:------|:---------|:-------------|
| `torch` | `torch>=2.0` | Segmentation inference, NN classification architectures |
| `report` | `fpdf2`, `matplotlib` | `generate_pdf_report()` |
| `cc3d` | `connected-components-3d` | Pipeline 1 (legacy `close_image` / `cc_report`) |
| `monai` | `monai>=1.2` | SwinUNETR / UNETR model variants |
| `train` | `h5py>=3.8`, `torchio>=0.19` | `PatchGenerator`, `SegmentationDataset`, `SegmentationTrainer` |
| `dev` | `torch`, `report`, `cc3d` + pytest | Development and testing |

Install a single extra, e.g.:

```bash
pip install -e ".[torch,report]"
```

---

## Quick-start: logistic classification pipeline

The classification step (`fit_classifier`) is fully trainable from labelled
descriptor CSVs. Segmentation (Step 1) requires a trained checkpoint; see
[Training on your own data](#training-on-your-own-data) for the full training
pipeline.

### Step 1 — Segmentation (requires a trained checkpoint)

```python
from tomographia.segmentation import run_inference

# Runs 3-D UNet inference on a directory of TIF slices.
# Writes labeled_volume.npy (binary prediction) to output_dir.
run_inference(
    input_dir="data_demo/ct_slices/",
    checkpoint="path/to/unet3d.pt",
    output_dir="outputs/",
    model_name="unet3d",
    normalization="robust_minmax",
    patch_shape=(16, 64, 64),
    device="cpu",      # or "cuda:0"
)
```

A trained checkpoint is not included in the repository. The segmentation model must be trained separately or obtained from the project maintainers.

### Step 2 — Connected components and geometric descriptors

```python
import numpy as np
from tomographia.components import build_cc_mask, compute_descriptors

binary_pred = np.load("outputs/labeled_volume.npy")

# Morphological closing + dilation → labeled integer CC array
cc = build_cc_mask(binary_pred, bridge_radius=1, dilation_radius=1)

# 12 canonical geometric descriptors per component
# voxel_size: isotropic physical voxel size in mm (here: 14 µm)
df = compute_descriptors(cc, voxel_size=(0.014, 0.014, 0.014))
df.to_csv("outputs/descriptors.csv", index=False)
```

The 12 features written to the CSV are:
`sphericity`, `anisotropy`, `flatness`, `elongation`, `fill_frac`,
`r1`, `r2`, `ar_zx`, `ar_zy`, `ar_yx`, `int_std_rel`, `log_voxels`.

### Step 3 — Train a logistic classifier

```python
from tomographia.classification import fit_classifier

# descriptor_csvs: one CSV per training volume, each with a 'cls_label'
# column containing "pore" or "crack".
# voxel_sizes: one scalar per CSV (isotropic voxel size in mm).
bundle = fit_classifier(
    descriptor_csvs=["outputs/descriptors.csv"],
    voxel_sizes=[0.014],
    label_col="cls_label",
    test_size=0.2,
    random_state=42,
    save_path="outputs/classifier.joblib",
)

print(bundle["holdout_results"])
# {'accuracy': ..., 'f1_crack': ..., 'f1_pore': ..., 'roc_auc': ...}
```

`fit_classifier` computes the decision threshold from training-set predictions only — no threshold leakage onto the held-out evaluation split.

### Step 4 — Classify and compute yield

```python
import joblib
from tomographia.classification import classify_defects_logistic, build_classified_volume
from tomographia.reporting import compute_yield_report

bundle = joblib.load("outputs/classifier.joblib")

df_classified = classify_defects_logistic(df, bundle)
classified_vol = build_classified_volume(cc, df_classified)
# classified_vol: uint8 array — 0=background, 1=pore, 2=crack

report = compute_yield_report(
    classified_vol,
    tif_dir="data_demo/ct_slices/",
    ref_value=30000,   # reference greyscale value for the matrix window
    lower=0.3,
    upper=0.7,
)
print(f"Total defect yield: {report['total_yield']:.2f}%")
print(f"  pore: {report['pore_yield']:.2f}%  crack: {report['crack_yield']:.2f}%")
```

### Unified dispatch API

```python
from tomographia.classification import classify_defects

# Logistic path — requires a bundle
df_out = classify_defects(df, method="logistic", bundle=bundle)

# KMeans path — unsupervised, no bundle needed
df_out = classify_defects(df, method="kmeans", features=["sphericity", "elongation", "log_voxels"])
```

---

## Alternative classification methods

### KMeans (unsupervised)

```python
from tomographia.classification import cluster_defects, make_cluster_volume

features = ["sphericity", "anisotropy", "flatness", "elongation",
            "fill_frac", "r1", "r2", "log_voxels"]

df_clustered, km_model, scaler = cluster_defects(df, features=features, n_clusters=4)
cluster_vol = make_cluster_volume(cc, df_clustered)
```

**Limitations:** Cluster indices are unsupervised — they do not map to "pore" / "crack" without manual inspection. `compute_yield_report` expects the logistic encoding (0=bg, 1=pore, 2=crack) and is not wired to the KMeans output. Use the logistic path for reporting.

### Neural-network embedding classifier (SimpleNN / MulticlassNN)

tomographia includes `SimpleNN`, `MulticlassNN`, and `HeatmapEmbedding` architectures that classify defects from 512-d heatmap embeddings (segment → extract heatmap → embed → classify), not from the tabular descriptor features.

**These are inference-only in 0.1.0.** No training code for these models existed in the original codebase and none is included here. Use them only if you have pre-trained weights for both `HeatmapEmbedding` and `SimpleNN`/`MulticlassNN`. For a fully trainable pipeline, use `fit_classifier()`.

**Reload caveat:** `HeatmapEmbedding` initialises its final linear layer (`fc1`) lazily on the first forward pass, so a freshly constructed instance has no `fc1` submodule. Any state dict saved from a trained instance includes `fc1` weights that cannot be loaded back into a fresh instance with `load_state_dict` — the call will raise `RuntimeError: unexpected key(s)`. The original codebase never saved `HeatmapEmbedding` weights at all, making inference non-reproducible across process restarts by design. See the CHANGELOG for a full description of this limitation.

If you have pre-trained weights that were saved and reloaded within the same process (i.e., inference was run immediately after training without serialisation), the forward pass looks like this:

```python
import torch
from tomographia.classification import SimpleNN, HeatmapEmbedding

embed_model = HeatmapEmbedding()          # fc1 initialised on first forward pass
classifier  = SimpleNN(input_size=512)    # must match embedding_dim

heatmap_tensor = ...   # (1, D, H, W) float32 tensor, values in [0, 1]
with torch.no_grad():
    embedding = embed_model(heatmap_tensor)  # (1, 512) — also triggers fc1 init
    logits    = classifier(embedding)        # (1, 1) — sigmoid P(crack)
```

---

## PDF defect report

Requires `pip install -e ".[report]"`.

`generate_pdf_report` is a legacy function designed around the Pipeline 1 report format. Its inputs do not come directly from the Pipeline 2 functions shown in the quick-start — two transformations are needed:

- **Key renaming:** `compute_yield_report` returns `"total_yield"` / `"pore_yield"` (as percentages); `generate_pdf_report` expects `"defect_yield"` / `"porosity_yield"` (as fractions 0–1).
- **Column renaming:** `compute_descriptors` outputs `zmin`/`zmax`/`ymin`/`ymax`/`xmin`/`xmax`; `generate_pdf_report` expects `bbox_z_min`/`bbox_z_max`/etc.

```python
import numpy as np
from tomographia.reporting import compute_yield_report, generate_pdf_report

# Step A — remap yield keys and convert % → fraction
yr = compute_yield_report(classified_vol, tif_dir="data_demo/ct_slices/",
                          ref_value=30000, lower=0.3, upper=0.7)
yield_dict = {
    "defect_yield"   : yr["total_yield"] / 100,
    "porosity_yield" : yr["pore_yield"]  / 100,
    "crack_yield"    : yr["crack_yield"] / 100,
}

# Step B — build largest-defect DataFrame with renamed columns
df_top = df_classified.nlargest(10, "voxels").rename(columns={
    "zmin": "bbox_z_min", "zmax": "bbox_z_max",
    "ymin": "bbox_y_min", "ymax": "bbox_y_max",
    "xmin": "bbox_x_min", "xmax": "bbox_x_max",
}).copy()
df_top["crack"] = df_top["cls_pred"]   # 0=pore, 1=crack

# Step C — load heatmap (requires run_inference(..., save_heatmap=True))
heatmap = np.load("outputs/labeled_heatmap.npy")   # float, shape (Z, Y, X), values in [0, 1]

generate_pdf_report(
    yield_dict=yield_dict,
    largest_df=df_top,
    heatmap=heatmap,
    data_path="data_demo/ct_slices/",
    save_path="outputs/defect_report.pdf",
    filename="demo_volume",
    scale=14.0,
)
```

See the `generate_pdf_report` docstring for full parameter specifications.

---

## Training on your own data

`SegmentationTrainer` trains the binary 3-D UNet segmentation model from your
own labelled volumes.  The configuration that produced the published results is
documented in [`configs/paper_sweep_anchor.md`](configs/paper_sweep_anchor.md).

### Prerequisites

```bash
pip install -e ".[torch,train]"   # adds h5py and torchio
```

A CUDA-capable GPU is strongly recommended; training on CPU is functional but
will be slow for production-size volumes.

### Data layout

Organise each volume as two flat directories of per-slice TIF files — one for
raw greyscale slices, one for binary labels:

```
data/
  volume_A/
    raw/
      slice_0001.tif
      slice_0002.tif
      ...
    labels/
      slice_0001.tif   # 0 = background, any non-zero value = defect
      slice_0002.tif
      ...
  volume_B/
    raw/   ...
    labels/ ...
```

Filenames in `raw/` and `labels/` are matched by the `prefix` argument
(default `"slice"`): `PatchGenerator` pairs `labels/slice_NNNN.tif` with
`raw/slice_NNNN.tif` in sorted order.  Label TIFs are binarised
(`label != 0`) during extraction — any non-zero value counts as a defect.

### HDF5 patch cache

`PatchGenerator` decompresses each volume once, extracts non-overlapping 3-D
patches, and writes them as individual `.h5` files under `patch_directory/`.
It also writes `splits.pkl` — an 80/10/10 train/val/test split per patch.
On subsequent runs, if `splits.pkl` exists and `overwrite=False`, extraction
is skipped and the cached splits are loaded directly.  Delete
`patch_directory/` or pass `overwrite=True` to re-extract.

### Training example

```python
import torch
from tomographia.segmentation import (
    PatchGenerator, SegmentationDataset, SegmentationTrainer,
    build_pos_sampler,
)
from tomographia.models import select_model

# 1. Extract patches (runs once; subsequent calls load from cache)
gen = PatchGenerator(
    directories={
        "volume_A": ("data/volume_A/raw/", "data/volume_A/labels/"),
        "volume_B": ("data/volume_B/raw/", "data/volume_B/labels/"),
    },
    patch_shape=(64, 64, 64),           # anchor: (64, 64, 64)
    patch_directory="patch_cache/",
    seed=999,                           # anchor seed
    normalization_name="robust_minmax", # anchor normalisation
    prefix="slice",
)
splits = gen.get_splits()

# 2. Datasets
train_ds = SegmentationDataset(splits["train"])
val_ds   = SegmentationDataset(splits["val"])

# 3. Loaders — build_pos_sampler weights patches so target_pos_frac of
#    each batch is positive; replaces shuffle=True on the train loader
sampler = build_pos_sampler(train_ds, target_pos_frac=0.99)  # anchor

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=8, sampler=sampler, num_workers=4,
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=4,
)

# 4. Model
model = select_model(
    "unet3d", num_channels=1, num_classes=1, patch_shape=(64, 64, 64),
)

# 5. Optimiser + scheduler (anchor: Adam + OneCycleLR)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-5,
    steps_per_epoch=len(train_loader), epochs=50,
)

# 6. Trainer
trainer = SegmentationTrainer(
    max_epochs=50,
    device="cuda:0",            # use "cpu" if no GPU available
    optimizer=optimizer,
    scheduler=scheduler,
    save_directory="checkpoints/",
    min_size=1,
    use_tta=True,
    tta_mode="flip",
    bce_or_focal="bce",
    bce_dice_weights=(0.5, 0.5),
    use_amp=False,              # see AMP note below
    on_log=None,                # pass callable(key, value, step) to hook into logging
)

trainer.fit(model, train_loader, val_loader)
```

Checkpoints are written to `save_directory/trained_model_epoch_{N}.bin` each
time the validation loss improves.  The last `log_n_last_models` checkpoints
are kept (default: 5).

### AMP (mixed precision)

Pass `use_amp=True` to enable `torch.cuda.amp` automatic mixed precision.

**Note on historical results**: all sweep runs in
[`configs/paper_sweep_anchor.md`](configs/paper_sweep_anchor.md) trained in
full FP32.  The original codebase contained a bug that prevented the
`GradScaler` from ever activating; AMP silently never ran.  AMP is functional
in this port but the published results were produced with `use_amp=False`,
which remains the default.

### Smoke test

To verify the pipeline runs end-to-end before committing to your own data:

```python
from tomographia.segmentation import make_synthetic_volume
volume, label = make_synthetic_volume(shape=(128, 128, 128), seed=0)
```

See `tests/test_integration.py` for a self-contained runnable example that
exercises `PatchGenerator → SegmentationTrainer` on synthetic data without
any real TIF files.

---

## Legacy connected-component path (Pipeline 1)

Pipeline 1 uses `connected-components-3d` (cc3d) + scipy instead of skimage. It predates Pipeline 2 and is retained for compatibility. Requires `pip install -e ".[cc3d]"`:

```python
from tomographia.components import close_image, cc_report, dict_to_df

binary_pred = ...   # uint8 array
cc_mask     = close_image(binary_pred)
report_dict = cc_report(cc_mask)
df          = dict_to_df(report_dict)
```

Pipeline 2 (`build_cc_mask` + `compute_descriptors`) is the default and recommended path for all new work.

---

## Project layout

```
src/tomographia/
    segmentation/
        _io.py             # TIF loading, normalisation utilities
        _dataset.py        # PatchedVolumeBig, SingleVolumeBig
        inference.py       # run_inference()
        _training_data.py  # PatchGenerator, SegmentationDataset
        _synthetic.py      # make_synthetic_volume() (smoke-test fixture)
        training.py        # SegmentationTrainer, SoftDiceLoss, FocalLoss
configs/
    paper_sweep_anchor.md  # ground-truth anchor config from launcher scripts
    components/
        pipeline2.py    # build_cc_mask, compute_descriptors (default)
        pipeline1.py    # close_image, cc_report (legacy, requires cc3d)
    classification/
        logistic.py     # fit_classifier, classify_defects_logistic, build_classified_volume
        kmeans.py       # cluster_defects, make_cluster_volume
        embeddings_nn.py# SimpleNN, MulticlassNN, HeatmapEmbedding (inference only)
    reporting/
        yield_report.py # compute_yield_report
        pdf_report.py   # generate_pdf_report
    models/
        unet.py         # Unet3D, Unet3d_classifier
        utils.py        # select_model()
data_demo/              # sample TIF slices (committed; see .gitignore exception)
tests/
    test_descriptors.py     # geometric descriptor formulas + analytical cross-check
    test_classification.py  # fit_classifier, classify_defects, build_classified_volume
    test_segmentation.py    # PatchedVolumeBig round-trip, normalisation
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

65 tests covering:
- Geometric descriptor formulas (`sphericity`, `log_voxels`, `fill_frac`, `elongation`, `r1`, `r2`, `anisotropy`) verified analytically against the closed-form inertia tensor formula for a scalene ellipsoid
- `fit_classifier` correctness: bundle keys, `feats_avail` tracking, threshold in `[0,1]`, holdout metrics, `len()` mismatch error, joblib round-trip
- Classification dispatch (`logistic`, `kmeans`, error paths), `build_classified_volume` encoding
- `PatchedVolumeBig` round-trip, patch shape, single-patch fallback, error handling
- Normalisation functions (`robust_minmax`, `standardization`, `minmax`, no-op)
- Integration: `make_synthetic_volume` → `PatchGenerator` → `SegmentationTrainer.fit()` → checkpoint written (end-to-end on CPU, ~50 s)
- `make_synthetic_volume`: shape/dtype, positive labels present, reproducibility, seed independence

**Not covered by the test suite:** `run_inference()` and `SingleVolumeBig` (require a model checkpoint and TIF files on disk), `compute_yield_report` and `generate_pdf_report` (require a TIF directory), all NN inference paths (`SimpleNN`/`MulticlassNN`/`HeatmapEmbedding`, require pre-trained weights), Pipeline 1 functions (require the optional `cc3d` package), and KMeans cluster assignment quality (unsupervised — dispatch is smoke-tested but cluster correctness is not verified).

---

## Attribution

Developed by **Antonio Peña Corredor** at [IRT Saint Exupéry](https://www.irt-saintexupery.com).

Former contributors to earlier versions of this codebase:
- Eduardo Hugo Sánchez
- Raphaël Ben-Zakour

Licensed under the [MIT License](LICENSE).

---

<p align="center">
  <img src="viz/logo_IRT Saint Exupery_RVB.png" alt="IRT Saint Exupéry logo" width="300"/>
</p>
