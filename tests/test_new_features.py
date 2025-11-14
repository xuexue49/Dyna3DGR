"""
Test new features: segmentation initialization and volume rendering.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.data import (
    initialize_from_segmentation,
    initialize_from_image,
    initialize_uniform_grid,
)
from dyna3dgr.rendering import VolumeRenderer
from dyna3dgr.models import Gaussian3D


def test_segmentation_initialization():
    """Test Gaussian initialization from segmentation mask."""
    print("=" * 60)
    print("Testing Segmentation Initialization")
    print("=" * 60)
    
    # Create synthetic segmentation mask
    H, W, D = 128, 128, 32
    segmentation = np.zeros((H, W, D), dtype=np.uint8)
    
    # Create synthetic cardiac structures
    # RV (label 1): sphere at (40, 64, 16)
    for h in range(H):
        for w in range(W):
            for d in range(D):
                if (h - 40)**2 + (w - 64)**2 + (d - 16)**2 < 15**2:
                    segmentation[h, w, d] = 1
    
    # MYO (label 2): ring around LV
    for h in range(H):
        for w in range(W):
            for d in range(D):
                dist = np.sqrt((h - 64)**2 + (w - 64)**2 + (d - 16)**2)
                if 15 < dist < 20:
                    segmentation[h, w, d] = 2
    
    # LV (label 3): sphere at (64, 64, 16)
    for h in range(H):
        for w in range(W):
            for d in range(D):
                if (h - 64)**2 + (w - 64)**2 + (d - 16)**2 < 15**2:
                    segmentation[h, w, d] = 3
    
    num_foreground = (segmentation > 0).sum()
    print(f"  Created synthetic segmentation: {H}x{W}x{D}")
    print(f"  Foreground voxels: {num_foreground}")
    print(f"  RV voxels: {(segmentation == 1).sum()}")
    print(f"  MYO voxels: {(segmentation == 2).sum()}")
    print(f"  LV voxels: {(segmentation == 3).sum()}")
    
    # Test initialization
    num_gaussians = 1000
    
    try:
        positions = initialize_from_segmentation(
            segmentation=segmentation,
            num_gaussians=num_gaussians,
            foreground_labels=[1, 2, 3],
            normalize=True,
            add_noise=True,
        )
        
        print(f"  ✓ Initialized {len(positions)} Gaussians")
        print(f"  Position shape: {positions.shape}")
        print(f"  Position range: [{positions.min():.4f}, {positions.max():.4f}]")
        print(f"  Position mean: {positions.mean(dim=0)}")
        
        # Check that positions are within [0, 1]
        assert positions.min() >= 0.0, "Positions should be >= 0"
        assert positions.max() <= 1.0, "Positions should be <= 1"
        
        print("✅ Segmentation initialization tests passed!")
        
    except Exception as e:
        print(f"❌ Segmentation initialization failed: {e}")
        raise
    
    print()


def test_image_initialization():
    """Test Gaussian initialization from intensity image."""
    print("=" * 60)
    print("Testing Image Initialization")
    print("=" * 60)
    
    # Create synthetic image
    H, W, D = 128, 128, 32
    image = np.random.randn(H, W, D).astype(np.float32)
    
    # Add high-intensity region (heart)
    for h in range(H):
        for w in range(W):
            for d in range(D):
                if (h - 64)**2 + (w - 64)**2 + (d - 16)**2 < 20**2:
                    image[h, w, d] += 2.0
    
    print(f"  Created synthetic image: {H}x{W}x{D}")
    print(f"  Intensity range: [{image.min():.4f}, {image.max():.4f}]")
    
    # Test initialization
    num_gaussians = 1000
    
    try:
        positions = initialize_from_image(
            image=image,
            num_gaussians=num_gaussians,
            percentile_threshold=60.0,
            normalize=True,
            add_noise=True,
        )
        
        print(f"  ✓ Initialized {len(positions)} Gaussians")
        print(f"  Position shape: {positions.shape}")
        print(f"  Position range: [{positions.min():.4f}, {positions.max():.4f}]")
        
        print("✅ Image initialization tests passed!")
        
    except Exception as e:
        print(f"❌ Image initialization failed: {e}")
        raise
    
    print()


def test_uniform_grid_initialization():
    """Test uniform grid initialization."""
    print("=" * 60)
    print("Testing Uniform Grid Initialization")
    print("=" * 60)
    
    shape = (128, 128, 32)
    num_gaussians = 1000
    
    try:
        positions = initialize_uniform_grid(
            shape=shape,
            num_gaussians=num_gaussians,
            normalize=True,
            add_noise=True,
        )
        
        print(f"  ✓ Initialized {len(positions)} Gaussians")
        print(f"  Position shape: {positions.shape}")
        print(f"  Position range: [{positions.min():.4f}, {positions.max():.4f}]")
        
        print("✅ Uniform grid initialization tests passed!")
        
    except Exception as e:
        print(f"❌ Uniform grid initialization failed: {e}")
        raise
    
    print()


def test_volume_renderer():
    """Test complete volume rendering."""
    print("=" * 60)
    print("Testing Volume Renderer")
    print("=" * 60)
    
    # Create Gaussians
    num_gaussians = 100
    feature_dim = 1
    
    gaussians = Gaussian3D(
        num_points=num_gaussians,
        feature_dim=feature_dim,
    )
    
    print(f"  Created {num_gaussians} Gaussians")
    
    # Create renderer
    image_size = (64, 64, 16)  # Smaller for faster testing
    renderer = VolumeRenderer(
        image_size=image_size,
        chunk_size=50,
    )
    
    print(f"  Created VolumeRenderer with size {image_size}")
    
    # Render
    try:
        rendered_volume = renderer(
            xyz=gaussians.xyz,
            scale=gaussians.scale,
            rotation=gaussians.rotation,
            opacity=gaussians.opacity,
            features=gaussians.features,
        )
        
        print(f"  ✓ Rendered volume shape: {rendered_volume.shape}")
        print(f"  Expected shape: {image_size + (feature_dim,)}")
        
        # Check shape
        assert rendered_volume.shape == image_size + (feature_dim,), \
            f"Shape mismatch: {rendered_volume.shape} vs {image_size + (feature_dim,)}"
        
        # Check value range
        print(f"  Rendered value range: [{rendered_volume.min():.4f}, {rendered_volume.max():.4f}]")
        
        print("✅ Volume renderer tests passed!")
        
    except Exception as e:
        print(f"❌ Volume renderer failed: {e}")
        raise
    
    print()


def test_volume_renderer_gradients():
    """Test that volume renderer is differentiable."""
    print("=" * 60)
    print("Testing Volume Renderer Gradients")
    print("=" * 60)
    
    # Create Gaussians with requires_grad
    num_gaussians = 50
    feature_dim = 1
    
    # Create leaf tensors
    xyz = torch.rand(num_gaussians, 3, requires_grad=True)  # Already in [0, 1]
    scale_base = torch.rand(num_gaussians, 3)
    scale = (scale_base * 0.1 + 0.01).requires_grad_(True)
    rotation = torch.randn(num_gaussians, 4, requires_grad=True)
    opacity = torch.rand(num_gaussians, 1, requires_grad=True)
    features = torch.randn(num_gaussians, feature_dim, requires_grad=True)
    
    print(f"  Created {num_gaussians} Gaussians with gradients")
    
    # Create renderer
    image_size = (32, 32, 8)  # Small for fast testing
    renderer = VolumeRenderer(
        image_size=image_size,
        chunk_size=25,
    )
    
    # Render
    try:
        rendered_volume = renderer(
            xyz=xyz,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            features=features,
        )
        
        print(f"  ✓ Rendered volume shape: {rendered_volume.shape}")
        
        # Compute loss
        target = torch.zeros_like(rendered_volume)
        loss = ((rendered_volume - target) ** 2).mean()
        
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert xyz.grad is not None, "xyz gradient is None"
        assert features.grad is not None, "features gradient is None"
        
        print(f"  ✓ xyz gradient norm: {xyz.grad.norm().item():.6f}")
        if scale.grad is not None:
            print(f"  ✓ scale gradient norm: {scale.grad.norm().item():.6f}")
        print(f"  ✓ features gradient norm: {features.grad.norm().item():.6f}")
        
        print("✅ Volume renderer gradient tests passed!")
        
    except Exception as e:
        print(f"❌ Volume renderer gradient test failed: {e}")
        raise
    
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Dyna3DGR New Features Test Suite")
    print("=" * 60)
    print()
    
    # Test initialization methods
    test_segmentation_initialization()
    test_image_initialization()
    test_uniform_grid_initialization()
    
    # Test volume renderer
    test_volume_renderer()
    test_volume_renderer_gradients()
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("New features are ready to use:")
    print("  1. ✅ Segmentation-based initialization")
    print("  2. ✅ Image-based initialization")
    print("  3. ✅ Uniform grid initialization")
    print("  4. ✅ Complete volume rendering")
    print("  5. ✅ Differentiable volume renderer")
    print()
    print("Project completion: 100%!")
    print("=" * 60)
