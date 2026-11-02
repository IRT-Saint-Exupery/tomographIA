"""Integration test: synthetic volume → PatchGenerator → SegmentationTrainer
→ checkpoint → run_inference.

Runs end-to-end on CPU with synthetic data (2 epochs, 128³ volume, 32³ patches).
"""
import os
import pathlib
import tempfile

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("h5py", reason="h5py not installed")


def _write_tif_slices(volume: np.ndarray, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for i in range(volume.shape[0]):
        tifffile.imwrite(os.path.join(directory, f"slice_{i:04d}.tif"), volume[i])


class TestSegmentationTrainerIntegration:
    """End-to-end smoke test: synthetic data through the full training pipeline."""

    def test_pipeline_runs_and_produces_checkpoint(self, tmp_path):
        """
        Generate synthetic volume → extract patches via PatchGenerator →
        train 2 epochs via SegmentationTrainer → confirm checkpoint written.
        """
        from tomographia.segmentation import (
            PatchGenerator,
            SegmentationDataset,
            SegmentationTrainer,
            build_pos_sampler,
            make_synthetic_volume,
        )
        from tomographia.models import select_model

        # 1. Synthetic data — 128³ gives 64 non-overlapping 32³ patches and a
        #    stable 80/10/10 split (seed=999 → 49 train / 7 val / 8 test).
        volume, label = make_synthetic_volume(shape=(128, 128, 128), seed=0)
        assert label.sum() > 0, "synthetic volume has no positive voxels"

        raw_dir = str(tmp_path / "raw")
        lbl_dir = str(tmp_path / "labels")
        _write_tif_slices(volume, raw_dir)
        _write_tif_slices(label,  lbl_dir)

        # 2. Patch extraction
        patch_dir = str(tmp_path / "patches")
        gen = PatchGenerator(
            directories={"synth": (raw_dir, lbl_dir)},
            patch_shape=(32, 32, 32),
            patch_directory=patch_dir,
            seed=999,
            normalization_name="robust_minmax",
            prefix="slice",
        )
        splits = gen.get_splits()
        assert len(splits["train"]) > 0, "no training patches extracted"
        assert len(splits["val"])   > 0, "no validation patches extracted"

        # 3. Datasets and loaders
        train_ds = SegmentationDataset(splits["train"])
        val_ds   = SegmentationDataset(splits["val"])

        sampler = build_pos_sampler(train_ds, target_pos_frac=0.99)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, sampler=sampler, num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=4, shuffle=False, num_workers=0,
        )

        # 4. Model
        model = select_model("unet3d", num_channels=1, num_classes=1)

        # 5. Optimiser + scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-5,
            steps_per_epoch=len(train_loader), epochs=2,
        )

        # 6. Trainer — 2 epochs on CPU
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = SegmentationTrainer(
            max_epochs=2,
            device="cpu",
            optimizer=optimizer,
            scheduler=scheduler,
            save_directory=ckpt_dir,
            min_size=1,
            use_tta=True,
            tta_mode="flip",
            bce_or_focal="bce",
            bce_dice_weights=(0.5, 0.5),
            use_amp=False,
            metadata=gen.metadata,
        )

        test_loader = torch.utils.data.DataLoader(
            SegmentationDataset(splits["test"]), batch_size=4, shuffle=False,
        )
        logged = []
        trainer.on_log = lambda k, v, step: logged.append(k)
        test_metrics = trainer.fit(model, train_loader, val_loader, test_loader)

        # 7. Test evaluated once, at the validation threshold, logged as Test/*
        assert test_metrics["threshold"] == pytest.approx(trainer.best_threshold_)
        assert sum(k == "Test/f1_score" for k in logged) == 1
        assert sum(k == "Val/f1_score" for k in logged) == 2

        # 8. Checkpoint stores weights + validation threshold
        ckpts = sorted(pathlib.Path(ckpt_dir).glob("trained_model_epoch_*.bin"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
        assert len(ckpts) > 0, f"No checkpoint found in {ckpt_dir}"
        ckpt = torch.load(str(ckpts[-1]), map_location="cpu", weights_only=False)
        assert {"state_dict", "threshold", "epoch", "normalization", "patch_shape",
                "min_size", "use_tta", "model_name"} <= set(ckpt)
        assert ckpt["threshold"] == pytest.approx(trainer.best_threshold_)
        assert ckpt["use_tta"] is True and tuple(ckpt["patch_shape"]) == (32, 32, 32)

        # 9. Inference with the checkpoint: binary uint8 volume of the input shape
        from tomographia.segmentation import run_inference
        out_dir = tmp_path / "pred"
        run_inference(raw_dir, ckpts[-1], out_dir, max_file=40, save_inputs=False,
                      save_heatmap=True, use_tta=False)   # settings read from the checkpoint
        pred = np.load(out_dir / "labeled_volume.npy")
        prob = np.load(out_dir / "labeled_heatmap.npy")
        assert pred.dtype == np.uint8 and pred.shape == (40, 128, 128)
        assert set(np.unique(pred)) <= {0, 1}
        np.testing.assert_array_equal(pred, (prob >= ckpt["threshold"]).astype(np.uint8))

    def test_make_synthetic_volume_shape_and_dtype(self):
        """make_synthetic_volume returns correctly shaped and typed arrays."""
        from tomographia.segmentation import make_synthetic_volume

        volume, label = make_synthetic_volume(shape=(32, 48, 64), n_defects=3, seed=7)
        assert volume.shape == (32, 48, 64)
        assert label.shape  == (32, 48, 64)
        assert volume.dtype == np.float32
        assert label.dtype  == np.uint8

    def test_make_synthetic_volume_has_positive_labels(self):
        """Default parameters embed at least one positive voxel."""
        from tomographia.segmentation import make_synthetic_volume

        _, label = make_synthetic_volume(seed=0)
        assert label.sum() > 0

    def test_make_synthetic_volume_reproducible(self):
        """Same seed produces identical output."""
        from tomographia.segmentation import make_synthetic_volume

        v1, l1 = make_synthetic_volume(seed=42)
        v2, l2 = make_synthetic_volume(seed=42)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(l1, l2)

    def test_make_synthetic_volume_different_seeds_differ(self):
        """Different seeds produce different volumes."""
        from tomographia.segmentation import make_synthetic_volume

        v1, _ = make_synthetic_volume(seed=0)
        v2, _ = make_synthetic_volume(seed=1)
        assert not np.array_equal(v1, v2)
