# Paper Sweep Anchor Configuration

**Source of truth:** `my_launchers/launcher_paper_sweep/step1_loss/step1_loss_sweep.sh`  
**Confirmed consistent across:** `step1_loss_sweep.sh`, `step1_bce_50_50.sh`,
`step1_bce_70_30.sh`, `step1_focal_30_70.sh` (all four read in full on 2026-06-19).

**Reconciliation note:** `train_complex_oversample.py`'s own argparse defaults
(`batch_size=32`, `lr=0.0001`, `num_epochs=20`, `patch_shape=(16,64,64)`,
`normalization_name="minmax"`, `seed=42`, `oversample_pos=False`, …) are
**not** the research configuration — they are fallback values for standalone
demo use.  An audit that reads only the script will get every anchor value
wrong.  These values come from the launcher.

---

## Fixed anchor (identical across all sweep runs)

| Parameter | Value | Launcher line |
|-----------|-------|---------------|
| `model_name` | `unet3d` | `--model_name unet3d` |
| `batch_size` | `8` | `--batch_size 8` |
| `lr` | `5e-5` | `--lr 0.00005` |
| `num_epochs` | `50` | `--num_epochs 50` |
| `patch_shape` | `(64, 64, 64)` | `--patch_shape 64 64 64` |
| `normalization_name` | `robust_minmax` | `--normalization_name robust_minmax` |
| `seed` | `999` | `--seed 999` |
| `oversample_pos` | `True` | `--oversample_pos True` |
| `pos_patch_frac` | `0.99` | `--pos_patch_frac 0.99` |
| `pos_weight` | `True` (auto-estimated) | *(argparse default; not overridden)* |
| `pos_weight_cap` | `15.0` | `--pos_weight_cap 15` |
| `use_tta` | `True` | `--use_tta True` |
| `tta_mode` | `flip` | `--tta_mode flip` |
| `augmentation` | `False` | `--augmentation False` |
| `min_size` | `1` (disabled) | `--min_size 1` |
| `train_ratio` | `1.0` | `--train_ratio 1.0` |
| `pretrain_eval_frac` | `0` (disabled) | `--pretrain_eval_frac 0` |
| `num_channels` | `1` | `--num_channels 1` |
| `scheduler` | `OneCycleLR(max_lr=5e-5)` | *(constructed in script body)* |
| `log_n_last_models` | `3` | `--log_n_last_models 3` |
| `log_every_n_steps` | `50` | `--log_every_n_steps 50` |
| `num_workers` | `4` | `--num_workers 4` |
| `wd` | `1e-6` | `--wd 0.000001` |
| `warm-start checkpoint` | `experiments/mlruns/560396612400109348/2bf3f894.../trained_model_epoch_17.bin` | `--checkpoint ${CHECKPOINT}` |

## Training data (5 metallic volumes)

```
inputs:  /workspace/magma/D-SS90/inputs
         /workspace/magma/crochet_tempo/inputs
         /workspace/magma/volume_crack_pores/inputs
         /workspace/magma/glad_r77/inputs
         /workspace/magma/crochet_rapide/inputs

labels:  /workspace/magma/D-SS90/labels_mod
         /workspace/magma/crochet_tempo/labels
         /workspace/magma/volume_crack_pores/labels
         /workspace/magma/glad_r77/labels_mod
         /workspace/magma/crochet_rapide/labels

extracted patches: /opt/results/six_metallic_multivolume_64_whole
```

---

## Varied parameters (sweep axis)

| Run | `bce_or_focal` | `bce_weight` | `dice_weight` | `alpha_focal` | `gamma_focal` |
|-----|----------------|--------------|---------------|---------------|---------------|
| BCE 50/50 | `bce` | 0.5 | 0.5 | — | — |
| BCE 70/30 | `bce` | 0.7 | 0.3 | — | — |
| Focal 30/70 | `focal` | 0.3 | 0.7 | 0.75 | 2.0 |
| Focal 50/50 | `focal` | 0.5 | 0.5 | 0.75 | 2.0 |
| Focal 70/30 | `focal` | 0.7 | 0.3 | 0.75 | 2.0 |

Two additional BCE 30/70 runs exist from earlier experiments (MLflow run IDs
`70ec17…` with `augmentation=True` confound, and `4722460…`/`f5c66d…` with
Tversky loss variant).  These are **not** part of the primary sweep.

---

## AMP / FP32 note

All sweep runs above trained in **full FP32**.  The original
`train_complex_oversample.py` contained a bug: `torch.cuda.amp.GradScaler()`
was constructed at line 757 but never passed to `SegmentationTrainer` or
`fit()`, so `training_step` always received `scaler=None` and the AMP branch
(`if use_cuda_amp and scaler is not None`) never executed.

AMP is functional in the ported library (`tomo_open`) via
`SegmentationTrainer(use_amp=True)` but was not used to produce the results
above.  Setting `use_amp=True` produces numerically different (but equivalent
in accuracy) results from those in the article.
