"""
Evaluation metrics for image reconstruction quality.

This module provides various metrics for evaluating the quality
of rendered images compared to ground truth.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import directed_hausdorff


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        pred: Predicted image
        target: Target image
    
    Returns:
        mae: Mean absolute error
    """
    return np.mean(np.abs(pred - target))


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        pred: Predicted image
        target: Target image
    
    Returns:
        mse: Mean squared error
    """
    return np.mean((pred - target) ** 2)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        pred: Predicted image
        target: Target image
    
    Returns:
        rmse: Root mean squared error
    """
    return np.sqrt(compute_mse(pred, target))


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image
        target: Target image
        data_range: Data range (max - min). If None, computed from target.
    
    Returns:
        psnr_value: PSNR in dB
    """
    if data_range is None:
        data_range = target.max() - target.min()
    
    return psnr(target, pred, data_range=data_range)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None,
    multichannel: bool = False,
) -> float:
    """
    Compute Structural Similarity Index.
    
    Args:
        pred: Predicted image
        target: Target image
        data_range: Data range (max - min). If None, computed from target.
        multichannel: Whether image is multichannel
    
    Returns:
        ssim_value: SSIM value [0, 1]
    """
    if data_range is None:
        data_range = target.max() - target.min()
    
    return ssim(target, pred, data_range=data_range, channel_axis=-1 if multichannel else None)


def compute_ncc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation.
    
    Args:
        pred: Predicted image
        target: Target image
    
    Returns:
        ncc: Normalized cross-correlation [-1, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    numerator = np.sum((pred_flat - pred_mean) * (target_flat - target_mean))
    denominator = np.sqrt(
        np.sum((pred_flat - pred_mean) ** 2) * np.sum((target_flat - target_mean) ** 2)
    )
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Dice Similarity Coefficient.
    
    Args:
        pred: Predicted segmentation (continuous or binary)
        target: Target segmentation (binary)
        threshold: Threshold for binarizing prediction
    
    Returns:
        dice: Dice coefficient [0, 1]
    """
    # Binarize if needed
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = (pred > threshold).astype(float)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU).
    
    Args:
        pred: Predicted segmentation
        target: Target segmentation
        threshold: Threshold for binarizing prediction
    
    Returns:
        iou: IoU value [0, 1]
    """
    # Binarize if needed
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = (pred > threshold).astype(float)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Hausdorff Distance between two binary masks.
    
    Args:
        pred: Predicted binary mask
        target: Target binary mask
    
    Returns:
        hausdorff_dist: Hausdorff distance in pixels
    """
    # Get coordinates of foreground pixels
    pred_coords = np.argwhere(pred > 0.5)
    target_coords = np.argwhere(target > 0.5)
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return float('inf')
    
    # Compute directed Hausdorff distances
    dist_pred_to_target = directed_hausdorff(pred_coords, target_coords)[0]
    dist_target_to_pred = directed_hausdorff(target_coords, pred_coords)[0]
    
    # Return maximum
    return max(dist_pred_to_target, dist_target_to_pred)


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None,
    include_segmentation: bool = False,
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        pred: Predicted image/segmentation
        target: Target image/segmentation
        data_range: Data range for PSNR/SSIM
        include_segmentation: Whether to include segmentation metrics
    
    Returns:
        metrics: Dictionary of metric names to values
    """
    metrics = {}
    
    # Image quality metrics
    metrics['MAE'] = compute_mae(pred, target)
    metrics['MSE'] = compute_mse(pred, target)
    metrics['RMSE'] = compute_rmse(pred, target)
    metrics['PSNR'] = compute_psnr(pred, target, data_range)
    metrics['SSIM'] = compute_ssim(pred, target, data_range)
    metrics['NCC'] = compute_ncc(pred, target)
    
    # Segmentation metrics (if requested)
    if include_segmentation:
        metrics['Dice'] = compute_dice(pred, target)
        metrics['IoU'] = compute_iou(pred, target)
        try:
            metrics['Hausdorff'] = compute_hausdorff_distance(pred, target)
        except:
            metrics['Hausdorff'] = float('inf')
    
    return metrics


def compute_temporal_consistency(sequence: np.ndarray) -> float:
    """
    Compute temporal consistency of a sequence.
    
    Measures how smooth the sequence is over time.
    
    Args:
        sequence: Image sequence [T, H, W] or [T, H, W, D]
    
    Returns:
        consistency: Temporal consistency score (lower is better)
    """
    T = sequence.shape[0]
    
    if T < 2:
        return 0.0
    
    # Compute frame-to-frame differences
    diffs = []
    for t in range(T - 1):
        diff = np.mean(np.abs(sequence[t + 1] - sequence[t]))
        diffs.append(diff)
    
    # Return mean difference
    return np.mean(diffs)


def compute_sequence_metrics(
    pred_sequence: np.ndarray,
    target_sequence: np.ndarray,
    data_range: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute metrics for each frame in a sequence.
    
    Args:
        pred_sequence: Predicted sequence [T, H, W] or [T, num_slices, H, W]
        target_sequence: Target sequence [T, H, W] or [T, num_slices, H, W]
        data_range: Data range for PSNR/SSIM
    
    Returns:
        metrics: Dictionary of metric names to arrays of values over time
    """
    T = pred_sequence.shape[0]
    
    metrics = {
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'PSNR': [],
        'SSIM': [],
        'NCC': [],
    }
    
    for t in range(T):
        pred_frame = pred_sequence[t]
        target_frame = target_sequence[t]
        
        metrics['MAE'].append(compute_mae(pred_frame, target_frame))
        metrics['MSE'].append(compute_mse(pred_frame, target_frame))
        metrics['RMSE'].append(compute_rmse(pred_frame, target_frame))
        metrics['PSNR'].append(compute_psnr(pred_frame, target_frame, data_range))
        metrics['SSIM'].append(compute_ssim(pred_frame, target_frame, data_range))
        metrics['NCC'].append(compute_ncc(pred_frame, target_frame))
    
    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics


def compute_motion_error(
    pred_motion: np.ndarray,
    target_motion: np.ndarray,
) -> Dict[str, float]:
    """
    Compute motion tracking error.
    
    Args:
        pred_motion: Predicted motion field [N, 3] or [T, N, 3]
        target_motion: Target motion field [N, 3] or [T, N, 3]
    
    Returns:
        errors: Dictionary of error metrics
    """
    # Compute displacement error
    displacement_error = np.linalg.norm(pred_motion - target_motion, axis=-1)
    
    errors = {
        'Mean_Displacement_Error': np.mean(displacement_error),
        'Median_Displacement_Error': np.median(displacement_error),
        'Max_Displacement_Error': np.max(displacement_error),
        'Std_Displacement_Error': np.std(displacement_error),
    }
    
    return errors


def compute_endpoint_error(
    pred_points: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """
    Compute endpoint error for point tracking.
    
    Args:
        pred_points: Predicted point positions [N, 3]
        target_points: Target point positions [N, 3]
    
    Returns:
        epe: Endpoint error (mean Euclidean distance)
    """
    distances = np.linalg.norm(pred_points - target_points, axis=1)
    return np.mean(distances)


class MetricsTracker:
    """Track metrics over training/evaluation."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update tracked metrics.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value
            self.counts[name] += 1
    
    def get_average(self) -> Dict[str, float]:
        """
        Get average of tracked metrics.
        
        Returns:
            avg_metrics: Dictionary of metric names to average values
        """
        avg_metrics = {}
        for name in self.metrics:
            if self.counts[name] > 0:
                avg_metrics[name] = self.metrics[name] / self.counts[name]
            else:
                avg_metrics[name] = 0.0
        
        return avg_metrics
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}
    
    def __repr__(self) -> str:
        """String representation."""
        avg_metrics = self.get_average()
        lines = ["MetricsTracker:"]
        for name, value in avg_metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names to values
        title: Title for the table
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name:30s}: {value:10.4f}")
        else:
            print(f"{name:30s}: {value}")
    
    print(f"{'=' * 50}\n")
