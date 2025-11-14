"""Utility functions and classes for Dyna3DGR."""

from .loss import Dyna3DGRLoss
from .knn import knn_search, knn_search_batch, knn_search_auto

# Alias for backward compatibility
find_knn = knn_search
from .metrics import (
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_ncc,
    compute_dice,
    compute_hausdorff_distance,
    compute_iou,
)

# Alias for backward compatibility
compute_hausdorff = compute_hausdorff_distance

__all__ = [
    'Dyna3DGRLoss',
    'find_knn',
    'knn_search',
    'knn_search_batch',
    'knn_search_auto',
    'compute_mae',
    'compute_mse',
    'compute_psnr',
    'compute_ssim',
    'compute_ncc',
    'compute_dice',
    'compute_hausdorff',
    'compute_hausdorff_distance',
    'compute_iou',
]
