<<<<<<< HEAD
# tomographia

Open-source Python library for 3-D CT defect analysis: segmentation, connected-component characterisation, classification (logistic regression or KMeans), and yield reporting.

Extracted and cleaned from the internal `tomographIA` research codebase at IRT Saint Exupéry.

---

## Features

- **Segmentation** — importable `run_inference()` API around a 3-D UNet; overlapping-patch inference with `PatchedVolumeBig`
- **Connected components** — `build_cc_mask` + `compute_descriptors` producing 12 canonical geometric features per component
- **Logistic classifier** — `fit_classifier()` trains a scikit-learn bundle with no threshold leakage; `classify_defects_logistic()` runs inference from saved bundles
- **KMeans clustering** — unsupervised alternative via `cluster_defects()`; no labelled data required
- **Yield reporting** — per-slice and overall pore/crack defect-density metrics from `compute_yield_report()`
- **PDF reports** — orthogonal-projection defect overlays via `generate_pdf_report()`

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
| `dev` | `torch`, `report`, `cc3d` + pytest | Development and testing |

Install a single extra, e.g.:

```bash
pip install -e ".[torch,report]"
```

---

## Quick-start: logistic classification pipeline

The classification step (`fit_classifier`) is fully trainable from labelled descriptor CSVs. Segmentation (Step 1) requires a separately-supplied model checkpoint; nothing in this library trains the segmentation model.

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
        _io.py          # TIF loading, normalisation utilities
        _dataset.py     # PatchedVolumeBig, SingleVolumeBig
        inference.py    # run_inference()
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

60 tests covering:
- Geometric descriptor formulas (`sphericity`, `log_voxels`, `fill_frac`, `elongation`, `r1`, `r2`, `anisotropy`) verified analytically against the closed-form inertia tensor formula for a scalene ellipsoid
- `fit_classifier` correctness: bundle keys, `feats_avail` tracking, threshold in `[0,1]`, holdout metrics, `len()` mismatch error, joblib round-trip
- Classification dispatch (`logistic`, `kmeans`, error paths), `build_classified_volume` encoding
- `PatchedVolumeBig` round-trip, patch shape, single-patch fallback, error handling
- Normalisation functions (`robust_minmax`, `standardization`, `minmax`, no-op)

**Not covered by the test suite:** `run_inference()` and `SingleVolumeBig` (require a model checkpoint and TIF files on disk), `compute_yield_report` and `generate_pdf_report` (require a TIF directory), all NN inference paths (`SimpleNN`/`MulticlassNN`/`HeatmapEmbedding`, require pre-trained weights), Pipeline 1 functions (require the optional `cc3d` package), and KMeans cluster assignment quality (unsupervised — dispatch is smoke-tested but cluster correctness is not verified).

---

## Attribution

Developed by **Antonio Peña Corredor** at [IRT Saint Exupéry](https://www.irt-saintexupery.com).

Former contributors to earlier versions of this codebase:
- Eduardo Hugo Sánchez
- Raphaël Ben-Zakour

Licensed under the [MIT License](LICENSE).
=======
# Tomo Open



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://forge-is.pf.irt-saintexupery.com/tomographia/tomo_open.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://forge-is.pf.irt-saintexupery.com/tomographia/tomo_open/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
* [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
>>>>>>> 143d809c399c4700187e72c81cdec8d5ef7a3b47
