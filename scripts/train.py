"""
Training script for Dyna3DGR.

This script implements the complete training pipeline as described in the paper:
- Per-case optimization (single patient training)
- Two-stage training strategy
- Control nodes with Linear Blend Skinning
- Gaussian densification and pruning
- Precise learning rate scheduling

Paper: Dyna3DGR: 4D Cardiac Motion Tracking with Dynamic 3D Gaussian Representation
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.models import (
    Gaussian3D,
    DeformationNetwork,
    ControlNodes,
    initialize_gaussians_from_point_cloud,
    initialize_control_nodes_from_gaussians,
    GaussianDensificationController,
)
from dyna3dgr.utils.loss import Dyna3DGRLoss
from dyna3dgr.utils.knn import knn_search_auto
from dyna3dgr.data import (
    PatientDataset,
    initialize_from_segmentation,
    initialize_from_image,
    initialize_uniform_grid,
)
from dyna3dgr.rendering import Medical2DSliceRenderer, VolumeRenderer, render_volume


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Dyna3DGR (Per-Case Optimization)')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/acdc.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--patient_dir',
        type=str,
        required=True,
        help='Path to single patient directory (per-case optimization)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/patient',
        help='Path to output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (faster iterations, more logging)'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(output_dir: str) -> Path:
    """Create output directories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    return output_dir


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Dyna3DGRTrainer:
    """
    Trainer for Dyna3DGR with per-case optimization.
    
    Implements:
    - Two-stage training (Stage 1: Gaussians only, Stage 2: Joint optimization)
    - Control nodes with Linear Blend Skinning
    - Gaussian densification and pruning
    - Precise learning rate scheduling (as per paper)
    """
    
    def __init__(
        self,
        config: dict,
        patient_dir: str,
        output_dir: Path,
        device: str = 'cuda',
        debug: bool = False,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            patient_dir: Path to patient directory
            output_dir: Output directory
            device: Device to use
            debug: Debug mode flag
        """
        self.config = config
        self.patient_dir = patient_dir
        self.output_dir = output_dir
        self.device = device
        self.debug = debug
        
        # Training parameters (from paper)
        self.max_iterations = config.get('max_iterations', 20000)
        self.stage1_iterations = config.get('stage1_iterations', 1000)
        self.control_nodes_start_iter = config.get('control_nodes_start_iter', 5000)
        
        # Setup components
        self.setup_data()
        self.setup_models()
        self.setup_renderer()
        self.setup_optimizers()
        self.setup_loss()
        self.setup_densification()
        self.setup_logging()
        
        # Training state
        self.current_iteration = 0
        self.best_loss = float('inf')
    
    def setup_data(self):
        """Setup data loading."""
        print(f"Loading patient data from: {self.patient_dir}")
        
        self.dataset = PatientDataset(
            patient_dir=self.patient_dir,
            image_size=tuple(self.config.get('image_size', [128, 128, 32])),
            load_segmentation=True,
        )
        
        print(f"  Loaded {len(self.dataset)} frames")
        
        # Get ED frame for initialization
        self.ed_frame = self.dataset[0]
        print(f"  Image shape: {self.ed_frame['image'].shape}")
    
    def setup_models(self):
        """Setup models: Gaussians, Control Nodes, Deformation Network."""
        print("\nInitializing models...")
        
        # Initialize 3D Gaussians from ED frame
        num_gaussians = self.config.get('num_gaussians', 5000)
        
        # Try to initialize from segmentation if available
        if 'segmentation' in self.ed_frame and self.ed_frame['segmentation'] is not None:
            print("  Initializing from segmentation mask...")
            try:
                gaussian_positions = initialize_from_segmentation(
                    segmentation=self.ed_frame['segmentation'],
                    num_gaussians=num_gaussians,
                    foreground_labels=[1, 2, 3],  # RV, MYO, LV
                    normalize=True,
                    add_noise=True,
                )
            except Exception as e:
                print(f"  Warning: Failed to initialize from segmentation: {e}")
                print("  Falling back to uniform grid initialization")
                image_shape = self.ed_frame['image'].shape
                gaussian_positions = initialize_uniform_grid(
                    shape=image_shape,
                    num_gaussians=num_gaussians,
                    normalize=True,
                )
        else:
            print("  No segmentation available, using uniform grid initialization")
            image_shape = self.ed_frame['image'].shape
            gaussian_positions = initialize_uniform_grid(
                shape=image_shape,
                num_gaussians=num_gaussians,
                normalize=True,
            )
        
        self.gaussians = initialize_gaussians_from_point_cloud(
            points=gaussian_positions,
            num_gaussians=num_gaussians,
            feature_dim=1,  # Intensity
        ).to(self.device)
        
        print(f"  ✓ Initialized {self.gaussians.num_points} Gaussians")
        
        # Initialize control nodes from Gaussian positions
        num_control_nodes = self.config.get('num_control_nodes', num_gaussians)
        self.control_nodes = initialize_control_nodes_from_gaussians(
            gaussian_positions=self.gaussians.xyz,
            num_control_nodes=num_control_nodes,
            init_radius=self.config.get('control_node_radius', 0.1),
        ).to(self.device)
        
        print(f"  ✓ Initialized {self.control_nodes.num_nodes} control nodes")
        
        # Initialize deformation network
        self.deformation_net = DeformationNetwork(
            spatial_dim=3,
            temporal_dim=1,
            spatial_freq=self.config.get('spatial_freq', 10),
            temporal_freq=self.config.get('temporal_freq', 6),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 8),
        ).to(self.device)
        
        print(f"  ✓ Initialized deformation network")
        
        # KNN parameters
        self.k_nearest = self.config.get('k_nearest', 4)
    
    def _create_uniform_grid(self, shape: tuple, num_points: int) -> np.ndarray:
        """Create uniform grid of points."""
        # Create grid in normalized space [0, 1]
        H, W, D = shape
        
        # Calculate grid size
        points_per_dim = int(np.ceil(num_points ** (1/3)))
        
        # Create grid
        x = np.linspace(0, 1, points_per_dim)
        y = np.linspace(0, 1, points_per_dim)
        z = np.linspace(0, 1, points_per_dim)
        
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        
        # Randomly sample if too many points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        
        return points
    
    def setup_renderer(self):
        """Setup renderer."""
        print("\nInitializing renderer...")
        
        # Get actual image size from data
        sample = self.ed_frame
        actual_shape = sample['image'].shape  # [H, W, D] or [H, W]
        
        if len(actual_shape) == 3:
            image_size = actual_shape  # [H, W, D]
        else:
            # 2D image, use config for D
            image_size = (*actual_shape, self.config.get('image_size', [128, 128, 32])[2])
        
        print(f"  Actual image size: {image_size}")
        
        # Choose renderer based on configuration
        use_volume_renderer = self.config.get('use_volume_renderer', True)
        
        if use_volume_renderer:
            self.renderer = VolumeRenderer(
                image_size=tuple(image_size),
                chunk_size=self.config.get('chunk_size', 1000),
            ).to(self.device)
            print(f"  ✓ Initialized VolumeRenderer (complete 3D rendering)")
        else:
            self.renderer = Medical2DSliceRenderer(
                image_size=tuple(image_size[:2]),
                num_slices=image_size[2] if len(image_size) > 2 else 32,
                chunk_size=self.config.get('chunk_size', 1000),
            ).to(self.device)
            print(f"  ✓ Initialized Medical2DSliceRenderer (single slice)")
        
        self.use_volume_renderer = use_volume_renderer
    
    def setup_optimizers(self):
        """
        Setup optimizers with precise learning rates as per paper.
        
        Learning rates (from paper):
        - Gaussian positions: 1e-4
        - Features (intensity): 5e-3
        - Rotation: 1e-4
        - Scale: 1e-4
        - Control nodes: 1e-4 (start from iter 5000)
        - Deformation network: 1e-6
        
        Optimizer: Adam with β=(0.9, 0.999), ε=1e-15
        LR decay: Exponential from 1e-4 to 1e-7
        """
        print("\nSetting up optimizers...")
        
        # Adam parameters (from paper)
        adam_betas = (0.9, 0.999)
        adam_eps = 1e-15
        
        self.optimizers = {}
        self.schedulers = {}
        
        # Gaussian position optimizer
        self.optimizers['xyz'] = optim.Adam(
            [self.gaussians._xyz],
            lr=1e-4,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Gaussian features (intensity) optimizer
        self.optimizers['features'] = optim.Adam(
            [self.gaussians._features],
            lr=5e-3,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Gaussian rotation optimizer
        self.optimizers['rotation'] = optim.Adam(
            [self.gaussians._rotation],
            lr=1e-4,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Gaussian scale optimizer
        self.optimizers['scale'] = optim.Adam(
            [self.gaussians._scale],
            lr=1e-4,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Gaussian opacity optimizer
        self.optimizers['opacity'] = optim.Adam(
            [self.gaussians._opacity],
            lr=1e-4,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Control nodes optimizer
        self.optimizers['control_nodes'] = optim.Adam(
            self.control_nodes.parameters(),
            lr=1e-4,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Deformation network optimizer
        self.optimizers['deformation_net'] = optim.Adam(
            self.deformation_net.parameters(),
            lr=1e-6,
            betas=adam_betas,
            eps=adam_eps,
        )
        
        # Setup learning rate schedulers (exponential decay: 1e-4 → 1e-7)
        # gamma = (1e-7 / 1e-4) ^ (1 / max_iterations)
        gamma = (1e-7 / 1e-4) ** (1.0 / self.max_iterations)
        
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma,
            )
        
        print(f"  ✓ Created {len(self.optimizers)} optimizers")
        print(f"  ✓ LR decay gamma: {gamma:.6f}")
        print(f"  ✓ LR range: 1e-4 → 1e-7 over {self.max_iterations} iterations")
    
    def setup_loss(self):
        """Setup loss function."""
        print("\nSetting up loss function...")
        
        self.loss_fn = Dyna3DGRLoss(
            recon_weight=self.config.get('reconstruction_weight', 1.0),
            temporal_weight=self.config.get('temporal_weight', 0.1),
            reg_weight=self.config.get('regularization_weight', 0.01),
            cyclic_weight=self.config.get('cycle_weight', 0.1),
        ).to(self.device)
        
        print(f"  ✓ Initialized Dyna3DGRLoss")
    
    def setup_densification(self):
        """Setup Gaussian densification controller."""
        print("\nSetting up densification controller...")
        
        self.densification_controller = GaussianDensificationController(
            grad_threshold=self.config.get('grad_threshold', 0.0002),
            opacity_threshold=self.config.get('opacity_threshold', 0.01),
            scale_threshold=self.config.get('scale_threshold', 0.1),
            densify_interval=self.config.get('densify_interval', 500),
            densify_start_iter=self.config.get('densify_start_iter', 500),
            densify_end_iter=self.config.get('densify_end_iter', None),
            max_gaussians=self.config.get('max_gaussians', None),
        )
        
        print(f"  ✓ Initialized densification controller")
        print(f"    - Start iteration: {self.densification_controller.densify_start_iter}")
        print(f"    - Interval: {self.densification_controller.densify_interval}")
    
    def setup_logging(self):
        """Setup logging."""
        print("\nSetting up logging...")
        
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        print(f"  ✓ TensorBoard logs: {self.output_dir / 'logs'}")
    
    def get_training_stage(self, iteration: int) -> str:
        """
        Get current training stage.
        
        Stage 1 (0-1000): Optimize Gaussians only
        Stage 2 (1000-20000): Joint optimization
        """
        if iteration < self.stage1_iterations:
            return 'stage1'
        else:
            return 'stage2'
    
    def freeze_parameters(self, stage: str, iteration: int):
        """
        Freeze/unfreeze parameters based on training stage.
        
        Stage 1 (0-1000):
          - Optimize: Gaussians (xyz, features, rotation, scale, opacity)
          - Freeze: Deformation network, Control nodes
        
        Stage 2 (1000-20000):
          - Optimize: Gaussians, Deformation network
          - Control nodes: Start optimizing from iteration 5000
        """
        if stage == 'stage1':
            # Freeze deformation network and control nodes
            for param in self.deformation_net.parameters():
                param.requires_grad = False
            for param in self.control_nodes.parameters():
                param.requires_grad = False
        
        elif stage == 'stage2':
            # Unfreeze deformation network
            for param in self.deformation_net.parameters():
                param.requires_grad = True
            
            # Control nodes: start from iteration 5000
            if iteration >= self.control_nodes_start_iter:
                for param in self.control_nodes.parameters():
                    param.requires_grad = True
            else:
                for param in self.control_nodes.parameters():
                    param.requires_grad = False
    
    def forward_with_deformation(
        self,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with deformation.
        
        Args:
            t: Time [1] or [N, 1]
        
        Returns:
            deformed_xyz: [N, 3] deformed positions
            deformed_scale: [N, 3] deformed scales
            deformed_features: [N, F] deformed features
        """
        # Get canonical Gaussian parameters
        canonical_xyz = self.gaussians.xyz  # [N, 3]
        canonical_scale = self.gaussians.scale  # [N, 3]
        canonical_features = self.gaussians.features  # [N, F]
        
        # Stage 1: No deformation
        if self.get_training_stage(self.current_iteration) == 'stage1':
            return canonical_xyz, canonical_scale, canonical_features
        
        # Stage 2: Apply deformation via control nodes
        # 1. Predict control node transformations
        control_positions = self.control_nodes.positions  # [M, 3]
        M = control_positions.shape[0]
        
        # Expand time dimension
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t_expanded = t.expand(M, 1) if t.dim() == 1 else t.unsqueeze(0).expand(M, 1)
        else:
            t_expanded = t
        
        # Predict control node deformations (with stop-gradient on positions)
        control_delta_xyz, control_alpha = self.deformation_net(
            control_positions.detach(),  # Stop gradient
            t_expanded,
        )
        
        # 2. KNN search: Find k nearest control nodes for each Gaussian
        knn_indices, knn_distances = knn_search_auto(
            query_points=canonical_xyz,
            reference_points=control_positions,
            k=self.k_nearest,
        )
        
        # 3. Linear Blend Skinning: Blend control node transformations to Gaussians
        gaussian_delta_xyz, gaussian_alpha = self.control_nodes.blend_transformations(
            control_transformations=(control_delta_xyz, control_alpha),
            query_points=canonical_xyz,
            k_nearest_indices=knn_indices,
        )
        
        # 4. Apply transformations
        deformed_xyz = canonical_xyz + gaussian_delta_xyz
        deformed_scale = canonical_scale * torch.exp(gaussian_alpha)
        deformed_features = canonical_features  # Features don't change
        
        return deformed_xyz, deformed_scale, deformed_features
    
    def train_iteration(self, batch: dict) -> dict:
        """
        Train one iteration.
        
        Args:
            batch: Batch data
        
        Returns:
            metrics: Dictionary of metrics
        """
        # Get data
        images = batch['images'].to(self.device)  # [T, H, W, D]
        timestamps = batch['timestamps'].to(self.device)  # [T]
        
        T = images.shape[0]
        
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Render all frames
        if self.use_volume_renderer:
            # Complete volume rendering
            rendered_volumes = []
            for t_idx in range(T):
                t = timestamps[t_idx:t_idx+1]  # [1]
                
                # Forward pass with deformation
                deformed_xyz, deformed_scale, deformed_features = self.forward_with_deformation(t)
                
                # Render complete volume
                rendered_volume = self.renderer(
                    xyz=deformed_xyz,
                    scale=deformed_scale,
                    rotation=self.gaussians.rotation,
                    opacity=self.gaussians.opacity,
                    features=deformed_features,
                )  # [H, W, D, F]
                
                rendered_volumes.append(rendered_volume)
            
            rendered_volumes = torch.stack(rendered_volumes, dim=0)  # [T, H, W, D, F]
            
            # Compute loss on complete volumes
            gt_volumes = images.unsqueeze(-1)  # [T, H, W, D, 1]
        else:
            # Single slice rendering (for debugging/faster training)
            rendered_images = []
            slice_idx = images.shape[3] // 2
            
            for t_idx in range(T):
                t = timestamps[t_idx:t_idx+1]  # [1]
                
                # Forward pass with deformation
                deformed_xyz, deformed_scale, deformed_features = self.forward_with_deformation(t)
                
                # Render middle slice
                rendered_slice = self.renderer(
                    means=deformed_xyz,
                    scales=deformed_scale,
                    rotations=self.gaussians.rotation,
                    opacities=self.gaussians.opacity,
                    features=deformed_features,
                )
                
                rendered_images.append(rendered_slice)
            
            rendered_images = torch.stack(rendered_images, dim=0)  # [T, H, W]
            gt_images = images[:, :, :, slice_idx]  # [T, H, W]
        
        # Compute loss
        if self.use_volume_renderer:
            # Get actual dimensions from data
            T, H, W, D, F = rendered_volumes.shape
            _, gt_H, gt_W, gt_D, _ = gt_volumes.shape
            
            # Reshape for loss computation: [T, H, W, D, F] -> [T*D, F, H, W]
            rendered_flat = rendered_volumes.permute(0, 3, 4, 1, 2).reshape(T*D, F, H, W)
            gt_flat = gt_volumes.permute(0, 3, 4, 1, 2).reshape(T*gt_D, 1, gt_H, gt_W)
            
            loss_dict = self.loss_fn(
                pred_images=rendered_flat,
                target_images=gt_flat,
            )
        else:
            loss_dict = self.loss_fn(
                pred_images=rendered_images.unsqueeze(1),  # [T, 1, H, W]
                target_images=gt_images.unsqueeze(1),  # [T, 1, H, W]
            )
        
        total_loss = loss_dict['total']
        
        # Backward pass
        total_loss.backward()
        
        # Accumulate gradients for densification
        if self.gaussians._xyz.grad is not None:
            self.densification_controller.accumulate_gradients(
                self.gaussians._xyz.grad
            )
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.gaussians.parameters(),
            max_norm=1.0,
        )
        
        # Optimizer step (based on training stage)
        stage = self.get_training_stage(self.current_iteration)
        
        # Always optimize Gaussians
        for name in ['xyz', 'features', 'rotation', 'scale', 'opacity']:
            self.optimizers[name].step()
            self.schedulers[name].step()
        
        # Stage 2: Also optimize deformation network and control nodes
        if stage == 'stage2':
            self.optimizers['deformation_net'].step()
            self.schedulers['deformation_net'].step()
            
            if self.current_iteration >= self.control_nodes_start_iter:
                self.optimizers['control_nodes'].step()
                self.schedulers['control_nodes'].step()
        
        # Prepare metrics
        metrics = {
            'total_loss': total_loss.item(),
            'stage': stage,
            'num_gaussians': self.gaussians.num_points,
        }
        metrics.update({k: v.item() for k, v in loss_dict.items() if k != 'total'})
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Stage 1 (Gaussians only): 0-{self.stage1_iterations}")
        print(f"Stage 2 (Joint optimization): {self.stage1_iterations}-{self.max_iterations}")
        print(f"Control nodes start: {self.control_nodes_start_iter}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        # Progress bar
        pbar = tqdm(range(self.max_iterations), desc='Training', dynamic_ncols=True, mininterval=1.0, file=sys.stderr, ncols=100)
        
        try:
            for iteration in pbar:
                self.current_iteration = iteration
                
                # Get training stage
                stage = self.get_training_stage(iteration)
                
                # Freeze/unfreeze parameters
                self.freeze_parameters(stage, iteration)
                
                # Get batch (cycle through dataset)
                batch_idx = iteration % len(self.dataset)
                batch = self.dataset.get_sequence()
                
                # Train iteration
                metrics = self.train_iteration(batch)
                
                # Gaussian densification
                if self.densification_controller.should_densify(iteration):
                    num_split, num_cloned, num_pruned = \
                        self.densification_controller.densify_and_prune(self.gaussians)
                    
                    metrics['densify_split'] = num_split
                    metrics['densify_cloned'] = num_cloned
                    metrics['densify_pruned'] = num_pruned
                    
                    pbar.write(
                        f"[Iter {iteration}] Densification: "
                        f"split={num_split}, cloned={num_cloned}, pruned={num_pruned}, "
                        f"total={self.gaussians.num_points}"
                    )
                
                # Logging
                if iteration % 10 == 0:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'train/{key}', value, iteration)
                    
                    # Log learning rates
                    for name, optimizer in self.optimizers.items():
                        lr = optimizer.param_groups[0]['lr']
                        self.writer.add_scalar(f'lr/{name}', lr, iteration)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'stage': stage,
                    'gaussians': self.gaussians.num_points,
                })
                
                # Save checkpoint
                if iteration % 1000 == 0 and iteration > 0:
                    self.save_checkpoint(f'iter_{iteration}.pth')
                
                # Save best checkpoint
                if metrics['total_loss'] < self.best_loss:
                    self.best_loss = metrics['total_loss']
                    self.save_checkpoint('best.pth')
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            self.save_checkpoint('interrupted.pth')
        
        finally:
            # Save final checkpoint
            self.save_checkpoint('final.pth')
            self.writer.close()
            
            print("\n" + "="*60)
            print("Training Completed")
            print("="*60)
            print(f"Total iterations: {self.current_iteration + 1}")
            print(f"Best loss: {self.best_loss:.6f}")
            print(f"Final Gaussians: {self.gaussians.num_points}")
            print(f"Checkpoints saved to: {self.output_dir / 'checkpoints'}")
            print("="*60 + "\n")
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / 'checkpoints' / filename
        
        checkpoint = {
            'iteration': self.current_iteration,
            'gaussians': self.gaussians.state_dict(),
            'control_nodes': self.control_nodes.get_state_dict_with_metadata(),
            'deformation_net': self.deformation_net.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_iteration = checkpoint['iteration']
        self.gaussians.load_state_dict(checkpoint['gaussians'])
        self.control_nodes = ControlNodes.from_state_dict_with_metadata(
            checkpoint['control_nodes']
        ).to(self.device)
        self.deformation_net.load_state_dict(checkpoint['deformation_net'])
        
        for name, opt in self.optimizers.items():
            if name in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][name])
        
        for name, sch in self.schedulers.items():
            if name in checkpoint['schedulers']:
                sch.load_state_dict(checkpoint['schedulers'][name])
        
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Loaded checkpoint from iteration {self.current_iteration}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    output_dir = setup_directories(args.output_dir)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create trainer
    trainer = Dyna3DGRTrainer(
        config=config,
        patient_dir=args.patient_dir,
        output_dir=output_dir,
        device=args.device,
        debug=args.debug,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
