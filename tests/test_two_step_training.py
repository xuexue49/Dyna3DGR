"""
Test script for two-step training pipeline.

This script validates that both training scripts can be imported and
their core components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from pathlib import Path


def test_step1_imports():
    """Test that Step 1 script can be imported."""
    print("Testing Step 1 imports...")
    
    # Import Step 1 trainer
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from train_step1_gaussians import Step1GaussianTrainer
    
    print("  ✓ Step1GaussianTrainer imported successfully")


def test_step2_imports():
    """Test that Step 2 script can be imported."""
    print("\nTesting Step 2 imports...")
    
    # Import Step 2 trainer
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from train_step2_dynamic import Step2DynamicTrainer
    
    print("  ✓ Step2DynamicTrainer imported successfully")


def test_config_loading():
    """Test that config file can be loaded."""
    print("\nTesting config loading...")
    
    import yaml
    
    config_path = Path(__file__).parent.parent / 'configs' / 'two_step_training.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check Step 1 config
    assert 'step1' in config, "Missing step1 config"
    assert 'max_iterations' in config['step1'], "Missing step1.max_iterations"
    assert 'num_gaussians' in config['step1'], "Missing step1.num_gaussians"
    
    print(f"  ✓ Step 1 config loaded: {config['step1']['max_iterations']} iterations")
    
    # Check Step 2 config
    assert 'step2' in config, "Missing step2 config"
    assert 'max_iterations' in config['step2'], "Missing step2.max_iterations"
    assert 'num_control_nodes' in config['step2'], "Missing step2.num_control_nodes"
    
    print(f"  ✓ Step 2 config loaded: {config['step2']['max_iterations']} iterations")


def test_step1_components():
    """Test Step 1 core components."""
    print("\nTesting Step 1 components...")
    
    from dyna3dgr.models import Gaussian3D
    from dyna3dgr.rendering import VolumeRenderer
    from dyna3dgr.data import initialize_uniform_grid
    
    # Create Gaussians
    positions = initialize_uniform_grid(
        shape=(32, 32, 8),
        num_gaussians=100,
        normalize=True,
    )
    
    gaussians = Gaussian3D(
        num_points=100,
        feature_dim=1,
    )
    # Set positions manually
    with torch.no_grad():
        if isinstance(positions, np.ndarray):
            gaussians.xyz.copy_(torch.from_numpy(positions).float())
        else:
            gaussians.xyz.copy_(positions.float())
    
    print(f"  ✓ Created {len(gaussians.xyz)} Gaussians")
    
    # Create renderer
    renderer = VolumeRenderer(
        image_size=(32, 32, 8),
        chunk_size=50,
    )
    
    print(f"  ✓ Created VolumeRenderer")
    
    # Test rendering
    with torch.no_grad():
        rendered = renderer(
            xyz=gaussians.xyz,
            scale=gaussians.scale,
            rotation=gaussians.rotation,
            opacity=gaussians.opacity,
            features=gaussians.features,
        )
    
    assert rendered.shape == (32, 32, 8, 1), f"Unexpected shape: {rendered.shape}"
    
    print(f"  ✓ Rendering works: {rendered.shape}")


def test_step2_components():
    """Test Step 2 core components."""
    print("\nTesting Step 2 components...")
    
    from dyna3dgr.models import Gaussian3D, DeformationNetwork, ControlNodes
    from dyna3dgr.utils import find_knn
    
    # Create Gaussians
    gaussians = Gaussian3D(num_points=100, feature_dim=1)
    print(f"  ✓ Created {len(gaussians.xyz)} Gaussians")
    
    # Create control nodes
    control_nodes = ControlNodes(
        num_nodes=50,
        init_positions=gaussians.xyz[:50].detach().clone(),
    )
    print(f"  ✓ Created {len(control_nodes.positions)} control nodes")
    
    # Create deformation network
    deform_net = DeformationNetwork(
        spatial_freq=10,
        temporal_freq=6,
        hidden_dim=128,
        num_layers=4,
    )
    print(f"  ✓ Created deformation network")
    
    # Test KNN
    knn_indices, knn_weights = find_knn(
        query_points=gaussians.xyz,
        reference_points=control_nodes.positions,
        k=4,
    )
    
    assert knn_indices.shape == (100, 4), f"Unexpected KNN indices shape: {knn_indices.shape}"
    assert knn_weights.shape == (100, 4), f"Unexpected KNN weights shape: {knn_weights.shape}"
    
    print(f"  ✓ KNN computed: {knn_indices.shape}")
    
    # Test deformation (simplified)
    print(f"  ✓ All Step 2 components initialized successfully")


def main():
    print("=" * 60)
    print("Two-Step Training Pipeline Test Suite")
    print("=" * 60)
    
    try:
        # Test imports
        test_step1_imports()
        test_step2_imports()
        
        # Test config
        test_config_loading()
        
        # Test components
        test_step1_components()
        test_step2_components()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nTwo-step training pipeline is ready to use!")
        print("\nQuick start:")
        print("  ./scripts/train_two_step.sh data/ACDC/training/patient001 outputs/patient001")
        print("")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
