"""
Initialization utilities for Dyna3DGR.

This module provides functions to initialize 3D Gaussians from medical image data,
particularly from segmentation masks.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def initialize_from_segmentation(
    segmentation: Union[np.ndarray, torch.Tensor],
    num_gaussians: int = 5000,
    spacing: Optional[Tuple[float, float, float]] = None,
    normalize: bool = True,
    foreground_labels: Optional[list] = None,
    add_noise: bool = True,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Initialize Gaussian positions from segmentation mask.
    
    This function samples points from the foreground region of a segmentation mask,
    which provides a better initialization than uniform grid sampling.
    
    Args:
        segmentation: Segmentation mask [H, W, D] or [D, H, W]
            - Background: 0
            - Foreground: > 0 (e.g., RV=1, MYO=2, LV=3)
        num_gaussians: Number of Gaussians to initialize
        spacing: Voxel spacing (H, W, D) in mm. If None, assumes isotropic spacing
        normalize: If True, normalize positions to [0, 1]
        foreground_labels: List of foreground labels to sample from.
            If None, uses all non-zero labels
        add_noise: If True, add small random noise to positions
        noise_std: Standard deviation of noise (relative to spacing)
    
    Returns:
        positions: [num_gaussians, 3] tensor of Gaussian positions
    
    Example:
        >>> # Load ED segmentation
        >>> ed_seg = load_segmentation('patient001_ED.nii.gz')  # [H, W, D]
        >>> 
        >>> # Initialize Gaussians from cardiac structures (RV, MYO, LV)
        >>> positions = initialize_from_segmentation(
        ...     ed_seg,
        ...     num_gaussians=5000,
        ...     foreground_labels=[1, 2, 3],  # RV, MYO, LV
        ... )
        >>> 
        >>> # Create Gaussians
        >>> gaussians = initialize_gaussians_from_point_cloud(
        ...     points=positions,
        ...     num_gaussians=5000,
        ... )
    """
    # Convert to numpy if tensor
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    # Ensure 3D
    assert segmentation.ndim == 3, f"Segmentation must be 3D, got {segmentation.ndim}D"
    
    H, W, D = segmentation.shape
    
    # Get foreground mask
    if foreground_labels is None:
        # Use all non-zero labels
        foreground_mask = segmentation > 0
    else:
        # Use specific labels
        foreground_mask = np.zeros_like(segmentation, dtype=bool)
        for label in foreground_labels:
            foreground_mask |= (segmentation == label)
    
    # Get foreground voxel coordinates
    foreground_coords = np.argwhere(foreground_mask)  # [N_fg, 3]
    
    num_foreground = len(foreground_coords)
    
    if num_foreground == 0:
        raise ValueError("No foreground voxels found in segmentation mask")
    
    print(f"  Found {num_foreground} foreground voxels")
    
    # Sample num_gaussians points from foreground
    if num_foreground >= num_gaussians:
        # Random sampling without replacement
        indices = np.random.choice(num_foreground, num_gaussians, replace=False)
        sampled_coords = foreground_coords[indices]
    else:
        # If not enough foreground voxels, sample with replacement
        print(f"  Warning: Only {num_foreground} foreground voxels, sampling with replacement")
        indices = np.random.choice(num_foreground, num_gaussians, replace=True)
        sampled_coords = foreground_coords[indices]
    
    # Convert to float
    positions = sampled_coords.astype(np.float32)  # [num_gaussians, 3]
    
    # Apply spacing if provided
    if spacing is not None:
        spacing_array = np.array(spacing, dtype=np.float32)  # [3]
        positions = positions * spacing_array[None, :]  # [num_gaussians, 3]
    
    # Add noise to avoid exact grid alignment
    if add_noise:
        if spacing is not None:
            noise_scale = spacing_array * noise_std
        else:
            noise_scale = np.ones(3, dtype=np.float32) * noise_std
        
        noise = np.random.randn(*positions.shape).astype(np.float32) * noise_scale[None, :]
        positions = positions + noise
    
    # Normalize to [0, 1] if requested
    if normalize:
        # Get bounding box
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        
        # Normalize
        positions = (positions - min_pos) / (max_pos - min_pos + 1e-8)
    
    # Convert to tensor
    positions = torch.from_numpy(positions)
    
    print(f"  Initialized {num_gaussians} Gaussians from segmentation")
    print(f"  Position range: [{positions.min():.4f}, {positions.max():.4f}]")
    
    return positions


def initialize_from_image(
    image: Union[np.ndarray, torch.Tensor],
    num_gaussians: int = 5000,
    spacing: Optional[Tuple[float, float, float]] = None,
    normalize: bool = True,
    intensity_threshold: Optional[float] = None,
    percentile_threshold: float = 50.0,
    add_noise: bool = True,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Initialize Gaussian positions from intensity image.
    
    This function samples points from high-intensity regions of an image,
    useful when segmentation is not available.
    
    Args:
        image: Intensity image [H, W, D] or [D, H, W]
        num_gaussians: Number of Gaussians to initialize
        spacing: Voxel spacing (H, W, D) in mm. If None, assumes isotropic spacing
        normalize: If True, normalize positions to [0, 1]
        intensity_threshold: Absolute intensity threshold. If None, uses percentile
        percentile_threshold: Percentile threshold (0-100) if intensity_threshold is None
        add_noise: If True, add small random noise to positions
        noise_std: Standard deviation of noise (relative to spacing)
    
    Returns:
        positions: [num_gaussians, 3] tensor of Gaussian positions
    
    Example:
        >>> # Load ED image
        >>> ed_image = load_image('patient001_ED.nii.gz')  # [H, W, D]
        >>> 
        >>> # Initialize Gaussians from high-intensity regions
        >>> positions = initialize_from_image(
        ...     ed_image,
        ...     num_gaussians=5000,
        ...     percentile_threshold=60.0,  # Sample from top 40% intensity
        ... )
    """
    # Convert to numpy if tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Ensure 3D
    assert image.ndim == 3, f"Image must be 3D, got {image.ndim}D"
    
    H, W, D = image.shape
    
    # Determine threshold
    if intensity_threshold is None:
        intensity_threshold = np.percentile(image, percentile_threshold)
    
    # Get high-intensity mask
    high_intensity_mask = image > intensity_threshold
    
    # Get high-intensity voxel coordinates
    high_intensity_coords = np.argwhere(high_intensity_mask)  # [N_hi, 3]
    
    num_high_intensity = len(high_intensity_coords)
    
    if num_high_intensity == 0:
        raise ValueError(f"No voxels above threshold {intensity_threshold}")
    
    print(f"  Found {num_high_intensity} high-intensity voxels (threshold={intensity_threshold:.4f})")
    
    # Sample num_gaussians points
    if num_high_intensity >= num_gaussians:
        indices = np.random.choice(num_high_intensity, num_gaussians, replace=False)
        sampled_coords = high_intensity_coords[indices]
    else:
        print(f"  Warning: Only {num_high_intensity} high-intensity voxels, sampling with replacement")
        indices = np.random.choice(num_high_intensity, num_gaussians, replace=True)
        sampled_coords = high_intensity_coords[indices]
    
    # Convert to float
    positions = sampled_coords.astype(np.float32)  # [num_gaussians, 3]
    
    # Apply spacing if provided
    if spacing is not None:
        spacing_array = np.array(spacing, dtype=np.float32)  # [3]
        positions = positions * spacing_array[None, :]  # [num_gaussians, 3]
    
    # Add noise
    if add_noise:
        if spacing is not None:
            noise_scale = spacing_array * noise_std
        else:
            noise_scale = np.ones(3, dtype=np.float32) * noise_std
        
        noise = np.random.randn(*positions.shape).astype(np.float32) * noise_scale[None, :]
        positions = positions + noise
    
    # Normalize to [0, 1] if requested
    if normalize:
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        positions = (positions - min_pos) / (max_pos - min_pos + 1e-8)
    
    # Convert to tensor
    positions = torch.from_numpy(positions)
    
    print(f"  Initialized {num_gaussians} Gaussians from image")
    print(f"  Position range: [{positions.min():.4f}, {positions.max():.4f}]")
    
    return positions


def initialize_uniform_grid(
    shape: Tuple[int, int, int],
    num_gaussians: int = 5000,
    spacing: Optional[Tuple[float, float, float]] = None,
    normalize: bool = True,
    add_noise: bool = True,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Initialize Gaussian positions from uniform grid.
    
    This is a fallback method when segmentation is not available.
    
    Args:
        shape: Image shape (H, W, D)
        num_gaussians: Number of Gaussians to initialize
        spacing: Voxel spacing (H, W, D) in mm. If None, assumes isotropic spacing
        normalize: If True, normalize positions to [0, 1]
        add_noise: If True, add small random noise to positions
        noise_std: Standard deviation of noise
    
    Returns:
        positions: [num_gaussians, 3] tensor of Gaussian positions
    """
    H, W, D = shape
    
    # Calculate grid size
    points_per_dim = int(np.ceil(num_gaussians ** (1/3)))
    
    # Create grid in normalized space [0, 1]
    x = np.linspace(0, 1, points_per_dim)
    y = np.linspace(0, 1, points_per_dim)
    z = np.linspace(0, 1, points_per_dim)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # [N, 3]
    
    # Randomly sample if too many points
    if len(positions) > num_gaussians:
        indices = np.random.choice(len(positions), num_gaussians, replace=False)
        positions = positions[indices]
    
    # Scale to image size if not normalizing
    if not normalize:
        positions[:, 0] *= H
        positions[:, 1] *= W
        positions[:, 2] *= D
        
        # Apply spacing if provided
        if spacing is not None:
            spacing_array = np.array(spacing, dtype=np.float32)
            positions = positions * spacing_array[None, :]
    
    # Add noise
    if add_noise:
        noise = np.random.randn(*positions.shape).astype(np.float32) * noise_std
        positions = positions + noise
    
    # Convert to tensor
    positions = torch.from_numpy(positions.astype(np.float32))
    
    print(f"  Initialized {num_gaussians} Gaussians from uniform grid")
    print(f"  Position range: [{positions.min():.4f}, {positions.max():.4f}]")
    
    return positions
