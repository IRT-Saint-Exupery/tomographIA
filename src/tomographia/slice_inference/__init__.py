"""Apply a 2-D patch classifier to whole slices and volumes.

Quick-start
-----------
::

    from tomographia.slice_inference import predict_volume, predictions_to_volume, remove_sparse_voxels

    preds = predict_volume(model, "ct_slices/", patch_size=64, device="cuda:0")
    grid = predictions_to_volume(preds, threshold=0.5)   # [slices, rows, cols]
    grid = remove_sparse_voxels(grid, min_neighbors=1)
"""
from .predict import (
    SlicePrediction,
    list_slices,
    predict_patches,
    predict_slice,
    predict_volume,
    predictions_to_volume,
    tile_slice,
)
from .volume import (
    downsample_volume,
    isosurface_mesh,
    plot_slice_overlay,
    remove_sparse_voxels,
    surface_voxels,
)

__all__ = [
    "tile_slice",
    "predict_patches",
    "predict_slice",
    "predict_volume",
    "predictions_to_volume",
    "list_slices",
    "SlicePrediction",
    "downsample_volume",
    "remove_sparse_voxels",
    "surface_voxels",
    "isosurface_mesh",
    "plot_slice_overlay",
]
