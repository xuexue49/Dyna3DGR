"""
Test core components of Dyna3DGR.

Tests:
- ControlNodes
- KNN search
- Linear Blend Skinning
- Gaussian densification
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dyna3dgr.models import (
    ControlNodes,
    initialize_control_nodes_from_gaussians,
    Gaussian3D,
    GaussianDensificationController,
)
from dyna3dgr.utils.knn import knn_search, get_knn_implementation


def test_control_nodes():
    """Test ControlNodes class."""
    print("\n" + "="*60)
    print("Testing ControlNodes")
    print("="*60)
    
    # Create control nodes
    num_nodes = 10
    init_positions = torch.randn(num_nodes, 3)
    control_nodes = ControlNodes(
        num_nodes=num_nodes,
        init_positions=init_positions,
        init_radius=0.1,
    )
    
    print(f"✓ Created ControlNodes with {num_nodes} nodes")
    print(f"  Positions shape: {control_nodes.positions.shape}")
    print(f"  Radii shape: {control_nodes.radii.shape}")
    print(f"  Radii range: [{control_nodes.radii.min():.4f}, {control_nodes.radii.max():.4f}]")
    
    # Test RBF weights
    query_points = torch.randn(20, 3)
    k_nearest_indices = torch.randint(0, num_nodes, (20, 4))
    
    weights = control_nodes.compute_rbf_weights(query_points, k_nearest_indices)
    
    print(f"✓ Computed RBF weights")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum (should be ~1.0): {weights.sum(dim=1).mean():.4f}")
    
    # Test Linear Blend Skinning
    control_delta_xyz = torch.randn(num_nodes, 3) * 0.1
    control_alpha = torch.randn(num_nodes, 3) * 0.1
    
    delta_xyz, alpha = control_nodes.blend_transformations(
        (control_delta_xyz, control_alpha),
        query_points,
        k_nearest_indices,
    )
    
    print(f"✓ Performed Linear Blend Skinning")
    print(f"  Delta XYZ shape: {delta_xyz.shape}")
    print(f"  Alpha shape: {alpha.shape}")
    
    # Test initialization from Gaussians
    gaussian_positions = torch.randn(100, 3)
    control_nodes_from_gaussians = initialize_control_nodes_from_gaussians(
        gaussian_positions,
        num_control_nodes=20,
    )
    
    print(f"✓ Initialized ControlNodes from Gaussians")
    print(f"  Number of nodes: {control_nodes_from_gaussians.num_nodes}")
    
    print("\n✅ ControlNodes tests passed!")
    return True


def test_knn_search():
    """Test KNN search."""
    print("\n" + "="*60)
    print("Testing KNN Search")
    print("="*60)
    
    print(f"KNN implementation: {get_knn_implementation()}")
    
    # Create test data
    query_points = torch.randn(50, 3)
    reference_points = torch.randn(100, 3)
    k = 4
    
    # Perform KNN search
    indices, distances = knn_search(query_points, reference_points, k)
    
    print(f"✓ Performed KNN search")
    print(f"  Query points: {query_points.shape}")
    print(f"  Reference points: {reference_points.shape}")
    print(f"  k: {k}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Distances shape: {distances.shape}")
    
    # Verify results
    assert indices.shape == (50, k), "Indices shape mismatch"
    assert distances.shape == (50, k), "Distances shape mismatch"
    assert torch.all(indices >= 0) and torch.all(indices < 100), "Invalid indices"
    assert torch.all(distances >= 0), "Negative distances"
    
    # Check that distances are sorted
    for i in range(50):
        assert torch.all(distances[i, 1:] >= distances[i, :-1]), \
            f"Distances not sorted for query {i}"
    
    print(f"  Average nearest distance: {distances[:, 0].mean():.4f}")
    print(f"  Average k-th distance: {distances[:, -1].mean():.4f}")
    
    print("\n✅ KNN search tests passed!")
    return True


def test_gaussian_densification():
    """Test Gaussian densification."""
    print("\n" + "="*60)
    print("Testing Gaussian Densification")
    print("="*60)
    
    # Create Gaussians
    num_gaussians = 100
    gaussians = Gaussian3D(num_gaussians, feature_dim=1)
    
    # Initialize positions
    gaussians._xyz.data = torch.randn(num_gaussians, 3)
    
    print(f"✓ Created {num_gaussians} Gaussians")
    
    # Create densification controller
    controller = GaussianDensificationController(
        grad_threshold=0.0002,
        opacity_threshold=0.01,
        scale_threshold=0.1,
        densify_interval=500,
        densify_start_iter=500,
    )
    
    print(f"✓ Created densification controller")
    print(controller)
    
    # Test should_densify
    assert not controller.should_densify(100), "Should not densify at iter 100"
    assert controller.should_densify(500), "Should densify at iter 500"
    assert not controller.should_densify(600), "Should not densify at iter 600"
    assert controller.should_densify(1000), "Should densify at iter 1000"
    
    print(f"✓ should_densify() works correctly")
    
    # Simulate gradient accumulation
    for _ in range(10):
        xyz_grad = torch.randn(num_gaussians, 3) * 0.001
        controller.accumulate_gradients(xyz_grad)
    
    print(f"✓ Accumulated gradients")
    print(f"  Gradient count: {controller.grad_count}")
    
    # Get average gradients
    avg_grad = controller.get_average_gradients()
    print(f"  Average gradient norm: {avg_grad.mean():.6f}")
    
    # Perform densification
    num_split, num_cloned, num_pruned = controller.densify_and_prune(gaussians)
    
    print(f"✓ Performed densification")
    print(f"  Gaussians split: {num_split}")
    print(f"  Gaussians cloned: {num_cloned}")
    print(f"  Gaussians pruned: {num_pruned}")
    print(f"  Total Gaussians: {gaussians.num_points}")
    
    print("\n✅ Gaussian densification tests passed!")
    return True


def test_integration():
    """Test integration of all components."""
    print("\n" + "="*60)
    print("Testing Component Integration")
    print("="*60)
    
    # Create Gaussians
    num_gaussians = 50
    gaussians = Gaussian3D(num_gaussians, feature_dim=1)
    gaussians._xyz.data = torch.randn(num_gaussians, 3)
    
    print(f"✓ Created {num_gaussians} Gaussians")
    
    # Initialize control nodes from Gaussians
    control_nodes = initialize_control_nodes_from_gaussians(
        gaussians.xyz,
        num_control_nodes=10,
    )
    
    print(f"✓ Initialized {control_nodes.num_nodes} control nodes")
    
    # Perform KNN search
    k = 4
    knn_indices, knn_distances = knn_search(
        gaussians.xyz,
        control_nodes.positions,
        k=k,
    )
    
    print(f"✓ Found {k} nearest control nodes for each Gaussian")
    
    # Compute RBF weights
    weights = control_nodes.compute_rbf_weights(
        gaussians.xyz,
        knn_indices,
    )
    
    print(f"✓ Computed RBF weights")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum: {weights.sum(dim=1).mean():.4f}")
    
    # Simulate control node transformations
    control_delta_xyz = torch.randn(control_nodes.num_nodes, 3) * 0.1
    control_alpha = torch.randn(control_nodes.num_nodes, 3) * 0.1
    
    # Blend to Gaussians
    gaussian_delta_xyz, gaussian_alpha = control_nodes.blend_transformations(
        (control_delta_xyz, control_alpha),
        gaussians.xyz,
        knn_indices,
    )
    
    print(f"✓ Blended transformations to Gaussians")
    print(f"  Delta XYZ range: [{gaussian_delta_xyz.min():.4f}, {gaussian_delta_xyz.max():.4f}]")
    print(f"  Alpha range: [{gaussian_alpha.min():.4f}, {gaussian_alpha.max():.4f}]")
    
    # Apply transformations
    new_xyz = gaussians.xyz + gaussian_delta_xyz
    new_scale = gaussians.scale * torch.exp(gaussian_alpha)
    
    print(f"✓ Applied transformations")
    print(f"  New XYZ range: [{new_xyz.min():.4f}, {new_xyz.max():.4f}]")
    print(f"  New scale range: [{new_scale.min():.4f}, {new_scale.max():.4f}]")
    
    print("\n✅ Integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Dyna3DGR Core Components Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_control_nodes()
        test_knn_search()
        test_gaussian_densification()
        test_integration()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
        return 0
    
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TESTS FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    exit(main())
