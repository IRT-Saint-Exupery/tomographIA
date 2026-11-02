<p align="center">
  <img src="viz/logo_tomoIA.png" alt="tomographIA logo" width="420"/>
</p>

Open-source Python library for defect analysis in X-ray computed tomography (CT):

- **3-D pipeline** — volume segmentation, connected-component characterisation, defect classification (logistic regression or KMeans) and yield reporting.
- **2-D pipeline** — patch-level defect detection with ResNet classifiers and **interpretable prototype networks** guided by expert anchors, calibrated evaluation, whole-slice inference and a patch-labelling tool.

This code originates from the internal research project "tomographIA" at IRT Saint Exupéry. The library provides the toolbox behind our publications; the scientific results and their discussion are in the corresponding articles. We welcome potential collaborations. Please feel free to contact us.

---

## Features

### 3-D volumes

- **Segmentation** — importable `run_inference()` API around a 3-D UNet; overlapping-patch inference with `PatchedVolumeBig`
- **Segmentation training** — `SegmentationTrainer` trains the 3-D UNet from
  your own HDF5 patch cache; `PatchGenerator` handles patch extraction and
  80/10/10 splitting
- **Connected components** — `build_cc_mask` + `compute_descriptors` producing 12 canonical geometric features per component
- **Logistic classifier** — `fit_classifier()` trains a scikit-learn bundle with no threshold leakage; `classify_defects_logistic()` runs inference from saved bundles
- **KMeans clustering** — unsupervised alternative via `cluster_defects()`; no labelled data required
- **Yield reporting** — per-slice and overall pore/crack defect-density metrics from `compute_yield_report()`
- **PDF reports** — orthogonal-projection defect overlays via `generate_pdf_report()`

### 2-D patches

- **Patch data** (`tomographia.patches`) — index `0/` / `1/` patch folders, reproducible 80/10/10 splits per folder or **grouped by specimen** (no leakage), augmentation
- **ResNet classifiers** (`tomographia.classification`) — `ResNetPatchClassifier` (ResNet-18…152, single-channel) and `PatchClassifierTrainer` (class-weighted BCE, AUC model selection, F1-optimal threshold)
- **Prototype networks** (`tomographia.prototypes`) — p-ResNet head on the CNN embedding, prototypes initialised at **expert-anchor medoids**, composite loss (class, pull, push, diversity, usage, anchor, medoid), training loop, semantic diagnostics and interpretability plots
- **Calibrated evaluation** (`tomographia.evaluation`) — temperature scaling, F1-optimal threshold on validation, ROC/PR-AUC, ECE, Brier, (cluster) bootstrap confidence intervals
- **Slice inference** (`tomographia.slice_inference`) — tile slices, predict whole volumes, stack into a 3-D patch grid, clean and mesh it
- **Labelling tool** (`tomographia.labelling`) — terminal CLI and Streamlit app to label patches, with intensity-based auto-labelling and optional model hints

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
| `vision` | `torch`, `torchvision>=0.15`, `tqdm` | 2-D patches, ResNet classifiers, prototype networks, slice inference |
| `labelling` | `matplotlib`, `streamlit` | Patch-labelling CLI and app |
| `dev` | `torch`, `vision`, `report`, `cc3d` + pytest | Development and testing |

Install a single extra, e.g.:

```bash
pip install -e ".[torch,report]"
```

---

## Quick-start (3-D): logistic classification pipeline

The classification step (`fit_classifier`) is fully trainable from labelled
descriptor CSVs. Segmentation (Step 1) requires a trained checkpoint; see
[Training the 3-D segmentation model](#training-the-3-d-segmentation-model-on-your-own-data) for the full training
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

## Training the 3-D segmentation model on your own data

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

## 2-D patch classification and prototype networks

Requires `pip install -e ".[vision]"` (plus `report` for the figures).

The 2-D pipeline works on square grey-level patches (64 × 64 px in the paper)
cut from CT slices, possibly along several slicing directions (xy / xz / yz).
It covers the full workflow of the article: a black-box ResNet baseline, an
interpretable prototype network built on top of it, their calibrated
evaluation, and their application to whole volumes.

### Patch data layout

```
patches/
  xy/
    0/  patch_000001.tif  ...   # no defect
    1/  patch_000123.tif  ...   # defect
  xz/
    0/ ...
    1/ ...
```

```python
from tomographia.patches import build_patch_index, split_patches, PatchDataset, PatchTransform

df = build_patch_index(["patches/xy", "patches/xz", "patches/yz"])

# Paper protocol: 80/10/10 per folder, random per patch
splits = split_patches(df, ratios=(0.8, 0.1, 0.1), seed=42)

# Recommended for new studies: keep every specimen in a single split
# (group_fn maps a patch path to its specimen id)
df = build_patch_index(["patches/xy"], group_fn=lambda path: path.split("_")[-2])
splits = split_patches(df, group_col="group", seed=42)

train_ds = PatchDataset(splits[splits.split == "train"], PatchTransform(augment=True))
val_ds   = PatchDataset(splits[splits.split == "val"])
test_ds  = PatchDataset(splits[splits.split == "test"])
```

> **Splitting and leakage.** A per-patch split places neighbouring slices and
> orthogonal views of the same region in different splits. Use `group_col`
> (specimen, volume…) whenever the split must measure generalisation to unseen
> parts. `split_patches` keeps the DataFrame index, which is the global sample
> id used by the expert anchors below.

### Step 1 — Black-box ResNet classifier

```python
from torch.utils.data import DataLoader
from tomographia.classification import ResNetPatchClassifier, PatchClassifierTrainer, PatchTrainConfig

model = ResNetPatchClassifier("resnet50", pretrained=True)   # ImageNet init, 1-channel conv1
trainer = PatchClassifierTrainer(model, PatchTrainConfig(lr=1e-4, max_epochs=100, device="cuda:0"))
result = trainer.fit(DataLoader(train_ds, batch_size=32, shuffle=True),
                     DataLoader(val_ds, batch_size=64))
print(result.best_val_auc, result.threshold)
```

The trainer uses class-weighted BCE (`pos_weight = n_neg / n_pos`), keeps the
weights with the best validation ROC-AUC, and restores them with a smaller
learning rate when validation AUC drops (`max_drop`, `patience_bad`,
`lr_reduction`).

### Step 2 — Embeddings and expert anchors

Anchors are training patches that an expert assigned to a defect **type**
(e.g. negatives: air / matrix / matrix + air; positives: pores / lines /
pores + lines). Each type becomes one prototype, initialised at the medoid of
its anchors in the standardised embedding space.

```python
import pandas as pd
from tomographia.prototypes import extract_embeddings, fit_embedding_scaler, prototypes_from_anchors

train_ids = PatchDataset(splits[splits.split == "train"], return_index=True)   # no augmentation
Z, y, idx = extract_embeddings(model, DataLoader(train_ids, batch_size=64), device="cuda:0")
scaler = fit_embedding_scaler(Z)

anchors = pd.read_csv("anchors.csv")      # columns: index (global sample id), cls (0/1), type
anchor_set = prototypes_from_anchors(scaler.transform(Z), idx, anchors,
                                     n_total=len(splits), labels=splits.label)
print(anchor_set.k_per_class)             # e.g. [3, 3]
```

### Step 3 — Prototype network

```python
from tomographia.prototypes import build_prototype_network, train_prototype_network, ProtoTrainConfig

pnet = build_prototype_network(model, anchor_set.centroids_by_class,
                               scaler.mean_, scaler.scale_)          # tau = 1, fixed

result = train_prototype_network(
    pnet,
    DataLoader(PatchDataset(splits[splits.split == "train"], PatchTransform(augment=True),
                            return_index=True), batch_size=64, shuffle=True),
    DataLoader(val_ds, batch_size=64),
    medoids=anchor_set.medoids,
    anchor_class=anchor_set.anchor_class,
    anchor_proto=anchor_set.anchor_proto,
    config=ProtoTrainConfig(device="cuda:0"),
)
```

`ProtoTrainConfig` and `ProtoLossWeights` default to the configuration of the
article:

| Setting | Value |
|:--|:--|
| Loss weights | class 0.1 · pull 0.05 · push 0.01 · diversity 1.0 · within-entropy 0.01 · usage 0.1 · anchor 2.0 · medoid 0.5 · proto-L2 1e-6 · τ-reg 1e-4 |
| Margins / temperatures | `tau_push` 0.1 · diversity margin δ 0.7 · τ = 1 (fixed) |
| Trainable parameters | prototype head + last ResNet block (`layer4`) |
| Optimiser | AdamW, lr head 5e-4, lr backbone 5e-6, weight decay 1e-4, grad clip 1.0 |
| Class weights | inverse frequency in the cross-entropy term |
| Model selection | lowest validation objective (prototype loss + medoid term); ReduceLROnPlateau (×0.5, patience 10, min 1e-7); early stopping patience 50 |

The prototype head receives the **pooled embedding** of the backbone
(`features()`, 2048-d for ResNet-50), never the classifier logit.

### Step 4 — Calibrated evaluation

```python
from tomographia.classification import predict_logits
from tomographia.prototypes import predict_prototype_network
from tomographia.evaluation import evaluate_calibrated

val  = predict_prototype_network(pnet, DataLoader(val_ds, batch_size=64), "cuda:0")
test = predict_prototype_network(pnet, DataLoader(test_ds, batch_size=64), "cuda:0")
report = evaluate_calibrated(val["logit"], val["label"], test["logit"], test["label"],
                             n_bootstrap=1000)
print(report["temperature"], report["threshold"], report["test"], report["ci"]["f1"])

# Black-box baseline: same protocol on its logits
zv, yv, _ = predict_logits(model, DataLoader(val_ds, batch_size=64), "cuda:0")
zt, yt, _ = predict_logits(model, DataLoader(test_ds, batch_size=64), "cuda:0")
baseline = evaluate_calibrated(zv, yv, zt, yt)
```

The temperature is fitted on validation; the operating threshold maximises F1
on *calibrated* validation probabilities; test metrics include accuracy,
precision, recall, F1, specificity, ROC-AUC, PR-AUC, ECE and Brier score with
bootstrap 95 % intervals (pass `test_groups=` for a cluster bootstrap by
specimen).

### Step 5 — Interpretability

```python
from tomographia.prototypes import (
    nearest_examples, coalescence_stats, prototype_drift, per_prototype_confusion)
from tomographia.prototypes.plotting import plot_prototype_examples, plot_prototype_errors

train_pred = predict_prototype_network(pnet, DataLoader(train_ids, batch_size=64), "cuda:0")
neigh = nearest_examples(pnet.head, train_pred["embedding"], k=20,
                         restrict_to_class=train_pred["label"])          # "learned anchors"
paths = splits.loc[train_pred["index"], "path"].to_numpy()
fig = plot_prototype_examples({pid: paths[rows] for pid, rows in neigh.items()})

print(coalescence_stats(pnet.head))                     # prototype collapse check
print(prototype_drift(pnet.head, anchor_set.medoids))   # distance from expert medoids
errors = per_prototype_confusion(test["proto_probs"].argmax(1), test["label"],
                                 (report["test_prob"] >= report["threshold"]).astype(int),
                                 pnet.head.k_per_class)
plot_prototype_errors(errors)
```

`plot_embedding_by_prototype` draws any 2-D projection (UMAP, t-SNE, PCA)
of the embeddings coloured by nearest prototype.

### Step 6 — Whole slices and volumes

```python
from tomographia.patches import load_patch
from tomographia.slice_inference import (
    predict_slice, predict_volume, predictions_to_volume, remove_sparse_voxels,
    isosurface_mesh, plot_slice_overlay)

pred = predict_slice(model, "volume/slice_0100.tif", patch_size=64, device="cuda:0")
plot_slice_overlay(load_patch("volume/slice_0100.tif"), pred, threshold=0.5)

preds = predict_volume(pnet, "volume/", patch_size=64, device="cuda:0", progress=True)
grid = remove_sparse_voxels(predictions_to_volume(preds, threshold=0.5))
verts, faces = isosurface_mesh(grid, spacing=(1, 64, 64))
```

Both classifiers and prototype networks are accepted. Patches are normalised
exactly as during training.

### Labelling patches

Requires `pip install -e ".[labelling]"`. The tool reads a CSV listing patch
centres (`id_global, x, y, z[, patch_size, intensity, prediction]`) next to
the `.tif` slices, auto-labels patches whose `intensity` is below a threshold
as background, and asks for `0` (no defect), `1` (defect) or `2` (unsure).

```bash
tomographia-label --data-path volume/ --identifier 123456 --threshold 50   # terminal
tomographia-label-app                                                      # browser (Streamlit)
```

---

## Project layout

```
src/tomographia/
    segmentation/          # 3-D
        _io.py             # TIF loading, normalisation utilities
        _dataset.py        # PatchedVolumeBig, SingleVolumeBig
        inference.py       # run_inference()
        _training_data.py  # PatchGenerator, SegmentationDataset
        _synthetic.py      # make_synthetic_volume() (smoke-test fixture)
        training.py        # SegmentationTrainer, SoftDiceLoss, FocalLoss
    components/
        pipeline2.py       # build_cc_mask, compute_descriptors (default)
        pipeline1.py       # close_image, cc_report (legacy, requires cc3d)
    classification/
        logistic.py        # fit_classifier, classify_defects_logistic, build_classified_volume
        kmeans.py          # cluster_defects, make_cluster_volume
        embeddings_nn.py   # SimpleNN, MulticlassNN, HeatmapEmbedding (inference only)
        resnet2d.py        # ResNetPatchClassifier (2-D)
        patch_training.py  # PatchClassifierTrainer, predict_logits / predict_proba (2-D)
    reporting/
        yield_report.py    # compute_yield_report
        pdf_report.py      # generate_pdf_report
    models/
        unet.py            # Unet3D, Unet3d_classifier
        utils.py           # select_model()
    patches/               # 2-D
        dataset.py         # build_patch_index, split_patches, PatchDataset, PatchTransform
    prototypes/
        model.py           # PrototypeHead, PrototypeNetwork, build_prototype_network
        anchors.py         # extract_embeddings, prototypes_from_anchors (expert-anchor medoids)
        losses.py          # ProtoLossWeights, prototype_loss
        training.py        # ProtoTrainConfig, train_prototype_network, predict_prototype_network
        diagnostics.py     # coalescence, drift, nearest examples, purity, per-prototype errors
        plotting.py        # prototype example grids, embedding scatter, error bars
    evaluation/
        metrics.py         # F1-optimal threshold, point metrics, ECE, Brier, bootstrap CIs
        calibration.py     # fit_temperature, evaluate_calibrated
    slice_inference/
        predict.py         # tile_slice, predict_slice, predict_volume, predictions_to_volume
        volume.py          # remove_sparse_voxels, isosurface_mesh, plot_slice_overlay
    labelling/
        session.py         # LabelingSession (shared by CLI and app)
        cli.py             # tomographia-label
        app.py             # tomographia-label-app (Streamlit)
configs/
    paper_sweep_anchor.md  # ground-truth configuration of the 3-D segmentation sweep
tests/
    test_descriptors.py     # geometric descriptor formulas + analytical cross-check
    test_classification.py  # fit_classifier, classify_defects, build_classified_volume
    test_segmentation.py    # PatchedVolumeBig round-trip, normalisation
    test_integration.py     # synthetic volume → PatchGenerator → SegmentationTrainer
    test_patches.py         # patch indexing, (grouped) splitting, dataset, transforms
    test_patch_models.py    # ResNet, prototype network, anchors, losses, slice inference
    test_evaluation.py      # metrics, temperature scaling, bootstrap
    test_labelling.py       # labelling session (no GUI)
viz/                        # logos
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

95 tests covering:
- Geometric descriptor formulas (`sphericity`, `log_voxels`, `fill_frac`, `elongation`, `r1`, `r2`, `anisotropy`) verified analytically against the closed-form inertia tensor formula for a scalene ellipsoid
- `fit_classifier` correctness: bundle keys, `feats_avail` tracking, threshold in `[0,1]`, holdout metrics, `len()` mismatch error, joblib round-trip
- Classification dispatch (`logistic`, `kmeans`, error paths), `build_classified_volume` encoding
- `PatchedVolumeBig` round-trip, patch shape, single-patch fallback, error handling
- Normalisation functions (`robust_minmax`, `standardization`, `minmax`, no-op)
- Integration: `make_synthetic_volume` → `PatchGenerator` → `SegmentationTrainer.fit()` → checkpoint written (end-to-end on CPU, ~50 s)
- `make_synthetic_volume`: shape/dtype, positive labels present, reproducibility, seed independence
- 2-D patches: deterministic indexing, per-folder and grouped splits (no group crosses splits), normalisation identical to the training pipeline, augmentation
- ResNet patch classifiers: shapes, state-dict layout, a short training run with index-aligned predictions
- Prototype networks: anchor medoids are real samples, loss terms and gradients, end-to-end training, diagnostics, and a regression test that the head consumes the embedding (not the classifier logit)
- Evaluation: F1-optimal threshold, ECE/Brier, plain and cluster bootstrap, temperature scaling recovering a known over-confidence
- Slice inference (tiling, volume stacking, sparse-voxel removal) and the labelling session (auto-labelling, autosave)

**Not covered by the test suite:** the Streamlit labelling app and plotting helpers (GUI), `run_inference()` and `SingleVolumeBig` (require a model checkpoint and TIF files on disk), `compute_yield_report` and `generate_pdf_report` (require a TIF directory), all NN inference paths (`SimpleNN`/`MulticlassNN`/`HeatmapEmbedding`, require pre-trained weights), Pipeline 1 functions (require the optional `cc3d` package), and KMeans cluster assignment quality (unsupervised — dispatch is smoke-tested but cluster correctness is not verified).

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
