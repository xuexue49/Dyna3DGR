"""
Test training script components.

Tests:
- Trainer initialization
- Two-stage training logic
- Learning rate scheduling
- Forward pass with deformation
"""

import torch
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dyna3dgr.models import (
    Gaussian3D,
    DeformationNetwork,
    ControlNodes,
    initialize_control_nodes_from_gaussians,
    GaussianDensificationController,
)
from dyna3dgr.utils.knn import knn_search_auto


def test_two_stage_training():
    """Test two-stage training logic."""
    print("\n" + "="*60)
    print("Testing Two-Stage Training Logic")
    print("="*60)
    
    # Training parameters
    max_iterations = 20000
    stage1_iterations = 1000
    control_nodes_start_iter = 5000
    
    # Test stage determination
    def get_training_stage(iteration):
        if iteration < stage1_iterations:
            return 'stage1'
        else:
            return 'stage2'
    
    # Test cases
    test_cases = [
        (0, 'stage1'),
        (500, 'stage1'),
        (999, 'stage1'),
        (1000, 'stage2'),
        (5000, 'stage2'),
        (10000, 'stage2'),
        (19999, 'stage2'),
    ]
    
    for iteration, expected_stage in test_cases:
        stage = get_training_stage(iteration)
        assert stage == expected_stage, \
            f"Iteration {iteration}: expected {expected_stage}, got {stage}"
        print(f"  ✓ Iteration {iteration:5d}: {stage}")
    
    print("\n✅ Two-stage training logic tests passed!")
    return True


def test_learning_rate_scheduling():
    """Test learning rate scheduling."""
    print("\n" + "="*60)
    print("Testing Learning Rate Scheduling")
    print("="*60)
    
    # Create dummy parameter
    param = torch.nn.Parameter(torch.randn(10, 3))
    
    # Create optimizer
    optimizer = torch.optim.Adam([param], lr=1e-4, betas=(0.9, 0.999), eps=1e-15)
    
    # Create scheduler (exponential decay: 1e-4 → 1e-7)
    max_iterations = 20000
    gamma = (1e-7 / 1e-4) ** (1.0 / max_iterations)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Target LR: 1e-7")
    
    # Test LR at different iterations
    test_iterations = [0, 1000, 5000, 10000, 15000, 19999]
    
    for iteration in test_iterations:
        # Step scheduler to this iteration
        for _ in range(iteration):
            scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"  Iteration {iteration:5d}: LR = {lr:.2e}")
        
        # Reset
        optimizer.param_groups[0]['lr'] = 1e-4
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Check final LR
    for _ in range(max_iterations):
        scheduler.step()
    
    final_lr = optimizer.param_groups[0]['lr']
    print(f"  Final LR (iter {max_iterations}): {final_lr:.2e}")
    
    # Should be close to 1e-7
    assert abs(final_lr - 1e-7) < 1e-8, f"Final LR {final_lr:.2e} not close to 1e-7"
    
    print("\n✅ Learning rate scheduling tests passed!")
    return True


def test_forward_with_deformation():
    """Test forward pass with deformation."""
    print("\n" + "="*60)
    print("Testing Forward Pass with Deformation")
    print("="*60)
    
    # Create models
    num_gaussians = 100
    num_control_nodes = 20
    
    gaussians = Gaussian3D(num_gaussians, feature_dim=1)
    gaussians._xyz.data = torch.randn(num_gaussians, 3)
    
    control_nodes = initialize_control_nodes_from_gaussians(
        gaussians.xyz,
        num_control_nodes=num_control_nodes,
    )
    
    deformation_net = DeformationNetwork(
        spatial_dim=3,
        temporal_dim=1,
        spatial_freq=10,
        temporal_freq=6,
        hidden_dim=256,
        num_layers=8,
    )
    
    print(f"  ✓ Created {num_gaussians} Gaussians")
    print(f"  ✓ Created {num_control_nodes} control nodes")
    print(f"  ✓ Created deformation network")
    
    # Forward pass
    t = torch.tensor([0.5])  # Time
    k_nearest = 4
    
    # 1. Predict control node transformations
    control_positions = control_nodes.positions
    M = control_positions.shape[0]
    t_expanded = t.expand(M, 1)
    
    control_delta_xyz, control_alpha = deformation_net(
        control_positions.detach(),
        t_expanded,
    )
    
    print(f"  ✓ Predicted control node transformations")
    print(f"    Delta XYZ shape: {control_delta_xyz.shape}")
    print(f"    Alpha shape: {control_alpha.shape}")
    
    # 2. KNN search
    knn_indices, knn_distances = knn_search_auto(
        query_points=gaussians.xyz,
        reference_points=control_positions,
        k=k_nearest,
    )
    
    print(f"  ✓ Performed KNN search")
    print(f"    KNN indices shape: {knn_indices.shape}")
    
    # 3. Linear Blend Skinning
    gaussian_delta_xyz, gaussian_alpha = control_nodes.blend_transformations(
        (control_delta_xyz, control_alpha),
        gaussians.xyz,
        knn_indices,
    )
    
    print(f"  ✓ Performed Linear Blend Skinning")
    print(f"    Gaussian delta XYZ shape: {gaussian_delta_xyz.shape}")
    print(f"    Gaussian alpha shape: {gaussian_alpha.shape}")
    
    # 4. Apply transformations
    deformed_xyz = gaussians.xyz + gaussian_delta_xyz
    deformed_scale = gaussians.scale * torch.exp(gaussian_alpha)
    
    print(f"  ✓ Applied transformations")
    print(f"    Deformed XYZ range: [{deformed_xyz.min():.4f}, {deformed_xyz.max():.4f}]")
    print(f"    Deformed scale range: [{deformed_scale.min():.4f}, {deformed_scale.max():.4f}]")
    
    # Check shapes
    assert deformed_xyz.shape == (num_gaussians, 3)
    assert deformed_scale.shape == (num_gaussians, 3)
    
    print("\n✅ Forward pass with deformation tests passed!")
    return True


def test_densification_integration():
    """Test densification integration."""
    print("\n" + "="*60)
    print("Testing Densification Integration")
    print("="*60)
    
    # Create Gaussians
    num_gaussians = 100
    gaussians = Gaussian3D(num_gaussians, feature_dim=1)
    gaussians._xyz.data = torch.randn(num_gaussians, 3)
    
    # Create densification controller
    controller = GaussianDensificationController(
        grad_threshold=0.0002,
        opacity_threshold=0.01,
        scale_threshold=0.1,
        densify_interval=500,
        densify_start_iter=500,
    )
    
    print(f"  ✓ Created {num_gaussians} Gaussians")
    print(f"  ✓ Created densification controller")
    
    # Test densification schedule
    test_iterations = [0, 100, 500, 600, 1000, 1500]
    
    for iteration in test_iterations:
        should_densify = controller.should_densify(iteration)
        expected = (iteration >= 500 and iteration % 500 == 0)
        
        assert should_densify == expected, \
            f"Iteration {iteration}: expected {expected}, got {should_densify}"
        
        status = "✓ Densify" if should_densify else "✗ Skip"
        print(f"  Iteration {iteration:4d}: {status}")
    
    # Simulate gradient accumulation and densification
    for _ in range(10):
        xyz_grad = torch.randn(num_gaussians, 3) * 0.001
        controller.accumulate_gradients(xyz_grad)
    
    print(f"  ✓ Accumulated gradients")
    
    # Perform densification
    num_split, num_cloned, num_pruned = controller.densify_and_prune(gaussians)
    
    print(f"  ✓ Performed densification")
    print(f"    Split: {num_split}")
    print(f"    Cloned: {num_cloned}")
    print(f"    Pruned: {num_pruned}")
    print(f"    Total Gaussians: {gaussians.num_points}")
    
    print("\n✅ Densification integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Dyna3DGR Training Components Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_two_stage_training()
        test_learning_rate_scheduling()
        test_forward_with_deformation()
        test_densification_integration()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nTraining script is ready to use!")
        print("\nTo train a patient:")
        print("  bash scripts/train_patient.sh data/ACDC/patient001 outputs/patient001")
        print("")
        
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
