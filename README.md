<p align="center">
  <img src="viz/logo_tomoIA.png" alt="tomographIA logo" width="420"/>
</p>

**tomographIA** is a Python toolbox to **find, measure and classify internal
defects in X-ray computed tomography (CT) scans** of materials and parts.

A CT scan is a stack of grey-level slices of the inside of a specimen.
Inspecting it by eye is slow and subjective; tomographIA turns it into
quantitative, reproducible results:

- **Where are the defects?** Segment them in the 3-D volume (3-D UNet) or
  detect them patch by patch in the slices (2-D ResNet or prototype network).
- **What are they?** Measure each defect (size, shape, intensity in physical
  units) and classify it, e.g. pore vs. crack.
- **How much is there?** Compute defect density per slice and for the whole
  volume, and produce a PDF report.
- **Can the decision be trusted?** Calibrated evaluation with confidence
  intervals, and interpretable prototype networks that explain each
  prediction with typical examples chosen by an expert.

It provides the building blocks (training, inference, evaluation and
labelling tools) to apply these methods to your own scans. No data or
trained models are distributed with the toolbox.

### Two complementary pipelines

- **3-D pipeline**, conceived for **metallic parts**: volume segmentation, connected-component characterisation, defect classification (logistic regression or KMeans) and defect-fraction reporting.
- **2-D pipeline**, conceived for **composite parts**: patch-level defect detection with ResNet classifiers and **interpretable prototype networks** guided by expert anchors, calibrated evaluation, whole-slice inference and a patch-labelling tool.

tomographIA is developed at [IRT Saint Exupéry](https://www.irt-saintexupery.com). We welcome collaborations: please open an [issue](https://github.com/IRT-Saint-Exupery/tomographIA/issues) to get in touch.

---

## Features

### 3-D volumes

- **Segmentation**: `run_inference()` runs a 3-D UNet on a stack of TIF slices with overlapping patches whose probabilities are averaged before thresholding; the threshold selected on validation is read from the checkpoint
- **Segmentation training**: `SegmentationTrainer` (BCE / focal + Dice, validation-selected threshold, single final test evaluation) and `PatchGenerator` (HDF5 patch cache, per-volume or per-patch splits)
- **Connected components**: `build_cc_mask` + `compute_descriptors`: shape and intensity descriptors in physical units, with documented definitions
- **Logistic classifier**: `fit_classifier()` trains a pore / crack model (threshold chosen on out-of-fold training predictions, optional volume-level hold-out); `classify_defects_logistic()` applies it
- **KMeans clustering**: unsupervised alternative via `cluster_defects()`
- **Defect fraction**: per-slice and overall pore / crack fraction with `compute_yield_report()`
- **PDF reports**: defect-fraction table and views of the largest defects with `generate_pdf_report()`

### 2-D patches

- **Patch data** (`tomographia.patches`): index `0/` / `1/` patch folders, reproducible 80/10/10 splits per folder or **grouped by specimen** (no leakage), augmentation
- **ResNet classifiers** (`tomographia.classification`): `ResNetPatchClassifier` (ResNet-18…152, single-channel) and `PatchClassifierTrainer` (class-weighted BCE, AUC model selection, F1-optimal threshold)
- **Prototype networks** (`tomographia.prototypes`): p-ResNet head on the CNN embedding, prototypes initialised at **expert-anchor medoids**, composite loss (class, pull, push, diversity, usage, anchor, medoid), training loop, semantic diagnostics and interpretability plots
- **Calibrated evaluation** (`tomographia.evaluation`): temperature scaling, F1-optimal threshold on validation, ROC/PR-AUC, ECE, Brier, (cluster) bootstrap confidence intervals
- **Slice inference** (`tomographia.slice_inference`): tile slices, predict whole volumes, stack into a 3-D patch grid, clean and mesh it
- **Labelling tool** (`tomographia.labelling`): terminal CLI and Streamlit app to label patches, with intensity-based auto-labelling and optional model hints

---

## Installation

tomographia is not yet published to PyPI. Install it from a clone:

```bash
git clone https://github.com/IRT-Saint-Exupery/tomographIA.git
cd tomographIA
pip install -e ".[dev]"
```

### Optional extras

| Extra | Installs | Required for |
|:------|:---------|:-------------|
| `torch` | `torch>=2.0` | Segmentation inference (`run_inference`) |
| `train` | `torch`, `h5py>=3.8`, `torchio>=0.19` | `PatchGenerator`, `SegmentationDataset`, `SegmentationTrainer` (torchio only for augmentation) |
| `report` | `fpdf2`, `matplotlib` | `generate_pdf_report()` |
| `cc3d` | `connected-components-3d` | Pipeline 1 (`close_image` / `cc_report`) |
| `vision` | `torch`, `torchvision>=0.15`, `tqdm` | 2-D patches, ResNet classifiers, prototype networks, slice inference |
| `labelling` | `matplotlib`, `streamlit` | Patch-labelling CLI and app |
| `dev` | all of the above except `labelling` + pytest | Development and testing |

The descriptor, logistic / KMeans and yield functions only need the core dependencies.

---

## Quick-start (3-D): segmentation → descriptors → classification → defect fraction

### Step 1: Segmentation (requires a trained checkpoint)

```python
from tomographia.segmentation import run_inference

run_inference(
    input_dir="ct_slices/",                # .tif slices, read in natural order
    checkpoint="checkpoints/trained_model_epoch_42.bin",
    output_dir="outputs/",
    device="cpu",                           # or "cuda:0"
)
```

`labeled_volume.npy` is the binary prediction (`uint8`, 0/1). Overlapping
patches are averaged as probabilities and thresholded once. The settings the
model was trained and validated with are read from the checkpoint written by
`SegmentationTrainer`: patch shape, normalisation, validation threshold,
`min_size` and flip TTA. Passing a different value explicitly raises a warning;
`save_heatmap=True` keeps the probability map. The checkpoint is loaded
strictly: a missing file or weights that do not match the architecture raise an
error.

No trained checkpoint is distributed with the toolbox; see
[Training the 3-D segmentation model](#training-the-3-d-segmentation-model-on-your-own-data).

### Step 2: Connected components and descriptors

```python
import numpy as np
from tomographia.components import build_cc_mask, compute_descriptors, load_grayscale_volume

binary_pred = np.load("outputs/labeled_volume.npy")
cc = build_cc_mask(binary_pred, bridge_radius=1)          # label the defects

df = compute_descriptors(
    cc,
    voxel_size=14.0, unit="µm",                             # or (z, y, x) sizes
    grayscale_volume=load_grayscale_volume("ct_slices/"),   # raw intensities
)
df.to_csv("outputs/descriptors.csv", index=False)
```

`build_cc_mask` groups fragments separated by gaps narrower than the closing
ball into one defect (`bridge_radius`; `dilation_radius` merges further), but
the returned labels cover exactly the predicted voxels, so sizes and shapes are
those of the segmentation. Components smaller than `min_size` voxels can be
dropped.

The voxel size is given with its unit and every descriptor is stored in mm
(mm², mm³), so volumes scanned at different resolutions can be compared and
pooled. The default classifier features
(`tomographia.components.CANONICAL_FEATURES`):

| Feature | Definition | Range |
|:--|:--|:--|
| `elongation` | `axis_mid / axis_major` (IBSI-style) | (0, 1]; → 0 for needles |
| `flatness` | `axis_minor / axis_major` (IBSI-style) | (0, 1]; → 0 for plates and needles |
| `sphericity` | `π^(1/3) (6V)^(2/3) / A`, area and volume of the same marching-cubes mesh | (0, 1] |
| `fill_frac` | voxels / bounding-box voxels | (0, 1] |
| `ar_zx`, `ar_zy`, `ar_yx` | ratios of the physical bounding-box extents along the scan axes | > 0 |
| `log_volume` | `log10` of the volume in mm³ | |
| `int_std_rel` | `std / mean` of the raw grey values inside the defect | ≥ 0 |

`axis_major ≥ axis_mid ≥ axis_minor` are the semi-axes `sqrt(5 λ)` of the
ellipsoid with the same second moments, from the eigenvalues `λ` of the
covariance of the component (voxels treated as boxes). The table also holds
`voxels`, `volume`, centroid, bounding box (`zmin…xmax`, max exclusive) and its
extents `ext_z, ext_y, ext_x` (voxels), `surface_area` and intensity statistics.

Points to keep in mind when interpreting the features:

- Including the voxel-box variance departs from IBSI for tiny components (a
  single voxel gets elongation = flatness = 1); shapes of components of a few
  voxels are dominated by discretisation.
- The mesh of a binary mask over-estimates the area, so sphericity is biased
  low and depends on size: a voxelised sphere of radius 20 voxels gives about
  0.92, a 10-voxel cube 0.85 (exact value 0.81).
- `ar_*` depend on how the part is oriented in the scanner; the other shape
  features are rotation-invariant.
- `int_std_rel` needs the raw (non-negative) CT intensities and depends on the
  scanner and reconstruction settings.

### Step 3: Train a logistic classifier

```python
from tomographia.classification import fit_classifier

# One descriptor table per volume, with a 'label' column holding the expert
# labels "pore" or "crack" (empty = unlabelled, ignored).
bundle = fit_classifier(
    ["outputs/vol_A_descriptors.csv", "outputs/vol_B_descriptors.csv",
     "outputs/vol_C_descriptors.csv"],
    holdout="volume",          # hold out whole volumes; "random" splits components
    test_size=0.3,
    save_path="outputs/classifier.joblib",
)
print(bundle["holdout_results"])
# {'accuracy': ..., 'f1_crack': ..., 'f1_pore': ..., 'roc_auc': ..., ...}
```

The model is a median imputer, a standard scaler and a class-balanced logistic
regression fitted on the training split. The decision threshold maximises F1
on 5-fold out-of-fold predictions of the training split (folds grouped by
volume with `holdout="volume"`); the held-out split is used only for the
reported metrics. `holdout="random"` splits components of the same volumes and
therefore measures within-specimen performance. With few volumes a single
hold-out is a noisy estimate. The bundle records the held-out volumes, the
training range of each feature (inference warns when a table lies far outside
it, e.g. wrong units) and the missing values per feature and volume.

### Step 4: Classify and compute the defect fraction

```python
from tomographia.classification import classify_defects_logistic, build_classified_volume
from tomographia.reporting import compute_yield_report

df_classified = classify_defects_logistic(df, "outputs/classifier.joblib")
classified_vol = build_classified_volume(cc, df_classified)   # 0 = bg, 1 = pore, 2 = crack

report = compute_yield_report(
    classified_vol,
    tif_dir="ct_slices/",
    ref_value=30000,   # reference grey value of the matrix
    lower=0.3,
    upper=0.7,
)
print(f"Defect fraction: {report['total_yield']:.2f}%")
print(f"  pores: {report['pore_yield']:.2f}%  cracks: {report['crack_yield']:.2f}%")
```

The "yield" is the defect fraction `defect voxels / (defect + matrix voxels)`,
where matrix voxels are non-defect voxels with a grey value in
`[ref_value·(1 − lower), ref_value·(1 + upper)]`. The `classified_vol` slices
must match the slice files from `slice_start` on.

`classify_defects_logistic` raises if a model feature is missing from the table
and replaces missing values (NaN) by the training median, with a warning.

### Unified dispatch API

```python
from tomographia.classification import classify_defects

df_out = classify_defects(df, method="logistic", bundle="outputs/classifier.joblib")
df_out = classify_defects(df, method="kmeans", n_clusters=4)    # unsupervised
```

---

## KMeans clustering (unsupervised)

```python
from tomographia.classification import cluster_defects, make_cluster_volume

df_clustered, km_model, scaler = cluster_defects(df, n_clusters=4)   # default: shape features
cluster_vol = make_cluster_volume(cc, df_clustered)
```

Cluster indices do not map to "pore" / "crack" without inspection, and
`compute_yield_report` expects the logistic encoding (0 = background,
1 = pore, 2 = crack).

---

## PDF defect report

Requires `pip install -e ".[report]"`.

```python
import numpy as np
from tomographia.reporting import generate_pdf_report

generate_pdf_report(
    yield_report=report,                    # compute_yield_report output
    defects=df_classified,                  # compute_descriptors (+ classification) output
    data_path="ct_slices/",
    save_path="outputs/defect_report.pdf",
    title="specimen A",
    voxel_size=14.0, unit="µm",
    heatmap=np.load("outputs/labeled_heatmap.npy"),   # optional, run_inference(save_heatmap=True)
    classified_vol=classified_vol,                     # optional overlay
    ref_value=30000, lower=0.3, upper=0.7,
)
```

If the segmentation was run on a subset of slices (`min_file=`), pass the same
index as `slice_start=` to both `compute_yield_report` and
`generate_pdf_report`.

---

## Training the 3-D segmentation model on your own data

`SegmentationTrainer` trains the binary 3-D UNet from your own labelled volumes.

### Prerequisites

```bash
pip install -e ".[train]"   # torch, h5py and torchio
```

A CUDA-capable GPU is strongly recommended: the 3-D UNet has about 90 M
parameters, and 64³ patches with batch size 8 need a large GPU (reduce the
batch or the patch size otherwise).

### Data layout

Each volume is two flat directories of per-slice TIF files, one for the raw
grey levels and one for the binary labels:

```
data/
  volume_A/
    raw/
      scan_0001.tif
      ...
    labels/
      slice_0001.tif   # 0 = background, any non-zero value = defect
      ...
  volume_B/ ...
```

Slices are read in natural order. Labels are matched on the slice number (the
last number in the file name): `raw/scan_0001.tif` or `raw/scan_0001_rec.tif`
→ `labels/<prefix>_0001.tif` (default `prefix="slice"`). Duplicate or missing
matches raise an error.

### Patches and splits

`PatchGenerator` normalises each volume, cuts it into patches (stride =
patch size by default; the last patch of each axis is aligned with the border)
and writes them as `.h5` files with a `splits.pkl` index. The cache is reused
when the settings and input files are unchanged and rebuilt otherwise. Voxels
shared by the border-aligned last patch and its neighbour are counted once in
validation / test metrics.

Use `split_volumes` to assign **whole volumes** to train / val / test: test
metrics then measure generalisation to unseen specimens. Without it, patches
are assigned at random (`ratios`, default 80/10/10); neighbouring patches of the
same volume then fall into different splits and validation / test metrics are
optimistic.

### Training example

```python
import torch
from tomographia.segmentation import (
    PatchGenerator, SegmentationDataset, SegmentationTrainer,
    build_pos_sampler, estimate_pos_weight,
)
from tomographia.models import select_model

gen = PatchGenerator(
    directories={
        "volume_A": ("data/volume_A/raw/", "data/volume_A/labels/"),
        "volume_B": ("data/volume_B/raw/", "data/volume_B/labels/"),
        "volume_C": ("data/volume_C/raw/", "data/volume_C/labels/"),
    },
    patch_shape=(64, 64, 64),                 # multiples of 16
    patch_directory="patch_cache/",
    normalization_name="robust_minmax",
    split_volumes={"volume_A": "train", "volume_B": "val", "volume_C": "test"},
)
splits = gen.get_splits()
train_ds = SegmentationDataset(splits["train"])
val_ds   = SegmentationDataset(splits["val"])
test_ds  = SegmentationDataset(splits["test"])

# Oversample patches containing defects; replaces shuffle=True
sampler = build_pos_sampler(train_ds, target_pos_frac=0.5)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=8, num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=8, num_workers=4)

model = select_model("unet3d", num_channels=1, num_classes=1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=50,
)

trainer = SegmentationTrainer(
    max_epochs=50,
    device="cuda:0",
    optimizer=optimizer,
    scheduler=scheduler,
    save_directory="checkpoints/",
    min_size=1,
    use_tta=True,                    # average probabilities over flipped inputs
    bce_or_focal="bce",
    pos_weight=None,                 # e.g. float(estimate_pos_weight(train_ds).clamp(max=50))
    bce_dice_weights=(0.5, 0.5),
    on_log=None,                     # callable(key, value, step) for MLflow, W&B, ...
    metadata=gen.metadata,           # normalisation and patch shape, stored in checkpoints
)
test_metrics = trainer.fit(model, train_loader, val_loader, test_loader)
print(trainer.best_epoch_, trainer.best_threshold_, test_metrics["f1_score"])
```

Protocol:

- After every epoch the threshold maximising voxel F1 on **validation** is
  selected (after `min_size` filtering when `min_size > 1`).
- The epoch with the lowest validation loss is kept; its checkpoint
  `trained_model_epoch_<N>.bin` stores the weights, that validation threshold,
  the validation metrics, `min_size`, `use_tta` and `metadata` (the last
  `log_n_last_models` best checkpoints are kept). `run_inference` reads them.
- The test split, if given, is evaluated **once** after training, with the best
  weights and the validation threshold; its metrics are logged as `Test/*`.
- Empty splits and a validation loss that is never finite raise an error.
- Metrics are computed per patch (`min_size` filtering per patch, no overlap
  averaging), whereas `run_inference` averages overlapping patches over the
  whole volume: expect small differences between validation and deployment.
- `bce_or_focal="focal"` uses the class-balanced focal loss
  (`alpha` = weight of defect voxels, `1 − alpha` of background voxels).
- `use_amp=True` enables CUDA mixed precision for the training step.

### Smoke test

```python
from tomographia.segmentation import make_synthetic_volume
volume, label = make_synthetic_volume(shape=(128, 128, 128), seed=0)
```

`tests/test_integration.py` runs `PatchGenerator → SegmentationTrainer →
run_inference` end to end on synthetic data (CPU). Synthetic volumes are a test
fixture: a model trained on them does not segment real CT data.

---

## Connected-component summary with cc3d (Pipeline 1)

A lighter alternative to `compute_descriptors` that only reports size,
bounding box (max exclusive) and centroid. As in `build_cc_mask`, the closing
only groups fragments: sizes are those of the predicted voxels. Requires
`pip install -e ".[cc3d]"`:

```python
from tomographia.components import close_image, cc_report, dict_to_df

labels, n = close_image(binary_pred)      # closing + cc3d labelling
df_cc = dict_to_df(cc_report(labels, min_pixels=10))
```

---

## 2-D patch classification and prototype networks

Requires `pip install -e ".[vision]"` (plus `report` for the figures).

The 2-D pipeline works on square grey-level patches (e.g. 64 × 64 px)
cut from CT slices, possibly along several slicing directions (xy / xz / yz).
It covers a black-box ResNet baseline, an interpretable prototype network
built on top of it, their calibrated evaluation, and their application to
whole volumes.

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

# 80/10/10 per folder, random per patch
splits = split_patches(df, ratios=(0.8, 0.1, 0.1), seed=42)

# Keep every specimen in a single split (no leakage). group_fn maps a patch
# path to its specimen id, here from file names like "<specimen>_<n>.tif"
import os
df = build_patch_index(["patches/xy"],
                       group_fn=lambda path: os.path.basename(path).split("_")[0])
splits = split_patches(df, group_col="group", seed=42)

train_ds = PatchDataset(splits[splits.split == "train"], PatchTransform(augment=True))
val_ds   = PatchDataset(splits[splits.split == "val"])
test_ds  = PatchDataset(splits[splits.split == "test"])
```

> **Splitting and leakage.** A per-patch split places neighbouring slices and
> orthogonal views of the same region in different splits. Use `group_col`
> (specimen, volume…) whenever the split must measure generalisation to unseen
> parts. The grouped split is not stratified: `split_patches` warns if a split
> ends up with a single class, or if the groups are just the source folders
> (no `group_fn`), which would put each slicing direction in its own split. It keeps the DataFrame index (which must be
> unique), the global sample id used by the expert anchors below.
>
> **Bit depth.** Patches are normalised as `(x / max_pixel_value − 0.5) / 0.5`
> with `max_pixel_value=255`. For 16-bit data pass
> `PatchTransform(max_pixel_value=65535)` and use the same value at inference
> (`predict_slice(..., max_pixel_value=65535)`); a warning is raised when
> values exceed `max_pixel_value`.

### Step 1: Black-box ResNet classifier

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
weights with the best validation ROC-AUC, restores them (with the optimiser
state) at a smaller learning rate when validation AUC drops (`max_drop`,
`patience_bad`, `lr_reduction`) and stops after `early_stop_patience` (30)
epochs without improvement.

### Step 2: Embeddings and expert anchors

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

### Step 3: Prototype network

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

Defaults of `ProtoTrainConfig` and `ProtoLossWeights`:

| Setting | Value |
|:--|:--|
| Loss weights | class 0.1 · pull 0.05 · push 0.01 · diversity 1.0 · within-entropy 0.01 · usage 0.1 · anchor 2.0 · medoid 0.5 · proto-L2 1e-6 · τ-reg 1e-4 |
| Margins / temperatures | `tau_push` 0.1 · diversity margin δ 0.7 · τ = 1 (fixed) |
| Trainable parameters | prototype head + last ResNet block (`layer4`) |
| Optimiser | AdamW, lr head 5e-4, lr backbone 5e-6, weight decay 1e-4, grad clip 1.0 |
| Class weights | inverse frequency in the cross-entropy term |
| Model selection | lowest validation objective (prototype loss + medoid term); ReduceLROnPlateau (×0.5, patience 10, min 1e-7); early stopping patience 50 |

The prototype head receives the **pooled embedding** of the backbone
(`features()`, 2048-d for ResNet-50), never the classifier logit. BatchNorm
layers of frozen blocks stay in eval mode, so the embedding space the scaler and
prototypes were fitted on does not drift. Squared distances are divided by the
embedding dimension (O(1) values): with `tau_push=0.1` the push term is weak,
raise `tau_push` for a stronger inter-class repulsion.

### Step 4: Calibrated evaluation

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
specimen; it needs at least two groups and warns below ten, where percentile
intervals under-cover). If the validation set is perfectly separated the
temperature has no finite optimum: it is set to its lower bound with a warning.

### Step 5: Interpretability

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

### Step 6: Whole slices and volumes

```python
from tomographia.patches import load_patch
from tomographia.slice_inference import (
    predict_slice, predict_volume, predictions_to_volume, remove_sparse_voxels,
    isosurface_mesh, plot_slice_overlay)

# Black-box baseline at its calibrated operating point (Step 4)
pred = predict_slice(model, "volume/slice_0100.tif", patch_size=64, device="cuda:0",
                     temperature=baseline["temperature"])
plot_slice_overlay(load_patch("volume/slice_0100.tif"), pred, threshold=baseline["threshold"])

# Prototype network at its calibrated operating point
preds = predict_volume(pnet, "volume/", patch_size=64, device="cuda:0", progress=True,
                       temperature=report["temperature"])
grid = remove_sparse_voxels(predictions_to_volume(preds, threshold=report["threshold"]))
verts, faces = isosurface_mesh(grid, spacing=(1, 64, 64))
```

Both classifiers and prototype networks are accepted. Patches are normalised
exactly as during training. The threshold of `evaluate_calibrated` is chosen
on temperature-scaled probabilities, so pass the same `temperature` when
predicting.

### Labelling patches

Requires `pip install -e ".[labelling]"`. The tool reads a CSV listing patch
centres (`id_global, x, y, z[, patch_size, intensity, prediction]`) next to
the `.tif` slices, auto-labels patches whose `intensity` is below a threshold
as background (unlabelled rows only; existing labels are kept), and asks for `0` (no defect), `1` (defect) or `2` (unsure).

```bash
tomographia-label --data-path volume/ --identifier 123456 --threshold 50   # terminal
tomographia-label-app                                                      # browser (Streamlit)
```

---

## Project layout

```
src/tomographia/
    _slices.py             # slice listing (natural order) and reading, shared by all readers
    segmentation/          # 3-D
        _io.py             # TIF loading, normalisation, small-component removal
        _dataset.py        # PatchedVolumeBig, SingleVolumeBig (inference patches)
        inference.py       # run_inference(), load_segmentation_checkpoint()
        _training_data.py  # PatchGenerator, SegmentationDataset
        _synthetic.py      # make_synthetic_volume() (smoke-test fixture)
        training.py        # SegmentationTrainer, SoftDiceLoss, FocalLoss
    components/
        pipeline2.py       # build_cc_mask, compute_descriptors, CANONICAL_FEATURES
        pipeline1.py       # close_image, cc_report (requires cc3d)
    classification/
        logistic.py        # fit_classifier, classify_defects_logistic, build_classified_volume
        kmeans.py          # cluster_defects, make_cluster_volume
        resnet2d.py        # ResNetPatchClassifier (2-D)
        patch_training.py  # PatchClassifierTrainer, predict_logits / predict_proba (2-D)
    reporting/
        yield_report.py    # compute_yield_report
        pdf_report.py      # generate_pdf_report
    models/
        unet.py            # Unet3D
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
tests/
    test_descriptors.py     # descriptor definitions on analytical shapes, build_cc_mask
    test_classification.py  # fit_classifier, classify_defects, build_classified_volume
    test_segmentation.py    # PatchedVolumeBig round-trip, normalisation
    test_integration.py     # synthetic volume → PatchGenerator → SegmentationTrainer → run_inference
    test_3d_regressions.py  # checkpoints, stitching, slice order, patch splits, reports
    test_patches.py         # patch indexing, (grouped) splitting, dataset, transforms
    test_patch_models.py    # ResNet, prototype network, anchors, losses, slice inference
    test_evaluation.py      # metrics, temperature scaling, bootstrap
    test_labelling.py       # labelling session (no GUI)
    test_2d_regressions.py  # 2-D edge cases (thresholds, splits, BatchNorm, anchors)
viz/                        # logos
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers:
- Descriptor definitions checked on analytical shapes (ellipsoid semi-axes, IBSI elongation / flatness, sphericity ≤ 1, anisotropic voxels, scale invariance) and `build_cc_mask` grouping without growing defects
- `fit_classifier` / `classify_defects_logistic`: bundle content, out-of-fold threshold, volume hold-out, label parsing, missing-feature errors, median imputation, joblib round-trip
- Segmentation: patch round-trip and probability stitching, small volumes, strict checkpoint loading, normalisation, class-balanced focal loss, `PatchGenerator` splits and cache
- Integration: `make_synthetic_volume` → `PatchGenerator` → `SegmentationTrainer.fit()` (single test evaluation at the validation threshold) → checkpoint → `run_inference()` (CPU)
- Slice ordering shared by all readers, yield and PDF reports
- 2-D patches: deterministic indexing, per-folder and grouped splits, normalisation identical to training, augmentation
- ResNet patch classifiers and prototype networks (anchors, losses, training, diagnostics, embedding-not-logit regression test, frozen BatchNorm)
- Evaluation: F1-optimal threshold, ECE/Brier, plain and cluster bootstrap, temperature scaling
- Slice inference and the labelling session

Not covered: the Streamlit app and plotting helpers (GUI), Pipeline 1 without the optional `cc3d` package, and the quality of KMeans clusters (unsupervised).

---

## Citation

If you use tomographIA in your work, please cite it. The citation metadata is in
[`CITATION.cff`](CITATION.cff), and GitHub shows a "Cite this repository" button.

---

## Attribution

Developed by **Antonio Peña Corredor** at [IRT Saint Exupéry](https://www.irt-saintexupery.com).

Former contributors to earlier versions of this codebase:
- Eduardo Hugo Sánchez
- Raphaël Ben-Zakour

Licensed under the [MIT License](LICENSE).

---

<p align="center">
  <img src="viz/logo_IRT%20Saint%20Exupery_RVB.png" alt="IRT Saint Exupéry logo" width="300"/>
</p>
