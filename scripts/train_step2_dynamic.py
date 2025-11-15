"""
Step 2: Train Complete Dynamic Model

This script trains the complete dynamic model including deformation network
and control nodes, starting from a pre-trained Gaussian representation.

This is the second step of a two-step training pipeline:
  Step 1: Train Gaussians only (train_step1_gaussians.py)
  Step 2: Train complete dynamic model (this script)
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.models import (
    Gaussian3D,
    DeformationNetwork,
    ControlNodes,
    GaussianDensificationController,
)
from dyna3dgr.data import PatientDataset
from dyna3dgr.rendering import VolumeRenderer, Medical2DSliceRenderer
from dyna3dgr.utils import Dyna3DGRLoss, find_knn


class Step2DynamicTrainer:
    """
    Trainer for Step 2: Complete dynamic model.
    
    This trainer loads pre-trained Gaussians from Step 1 and trains
    the complete dynamic model including deformation network and control nodes.
    """
    
    def __init__(self, config, patient_dir, step1_checkpoint, output_dir, device='cuda'):
        self.config = config
        self.patient_dir = Path(patient_dir)
        self.step1_checkpoint = step1_checkpoint
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup components
        self.setup_data()
        self.load_step1_gaussians()
        self.setup_dynamic_models()
        self.setup_renderer()
        self.setup_optimizers()
        self.setup_loss()
        self.setup_densification()
        self.setup_logging()
        
        print("\n" + "=" * 60)
        print("Step 2: Training Complete Dynamic Model")
        print("=" * 60)
        print(f"Patient: {self.patient_dir.name}")
        print(f"Step 1 checkpoint: {self.step1_checkpoint}")
        print(f"Max iterations: {self.config['max_iterations']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")
    
    def setup_data(self):
        """Load patient data (all frames for dynamic modeling)."""
        print(f"Loading patient data from: {self.patient_dir}")
        
        self.dataset = PatientDataset(
            patient_dir=str(self.patient_dir),
            load_segmentation=True,
        )
        
        print(f"  Loaded {len(self.dataset)} frames")
        
        # Load all frames
        self.frames = []
        for i in range(len(self.dataset)):
            frame = self.dataset[i]
            frame['image'] = frame['image'].to(self.device)
            if 'segmentation' in frame:
                frame['segmentation'] = frame['segmentation'].to(self.device)
            self.frames.append(frame)
        
        print(f"  Frame shape: {self.frames[0]['image'].shape}")
    
    def load_step1_gaussians(self):
        """Load pre-trained Gaussians from Step 1."""
        print(f"\nLoading Step 1 Gaussians from: {self.step1_checkpoint}")
        
        checkpoint = torch.load(self.step1_checkpoint, map_location=self.device)
        
        # Create Gaussian model
        num_gaussians = checkpoint['gaussians']['xyz'].shape[0]
        feature_dim = checkpoint['gaussians']['features'].shape[1]
        
        self.gaussians = Gaussian3D(
            num_points=num_gaussians,
            feature_dim=feature_dim,
        ).to(self.device)
        
        # Load state dict
        self.gaussians.load_state_dict(checkpoint['gaussians'])
        
        print(f"  ✓ Loaded {num_gaussians} Gaussians")
        print(f"  Step 1 loss: {checkpoint.get('loss', 'N/A')}")
    
    def setup_dynamic_models(self):
        """Initialize deformation network and control nodes."""
        print("\nInitializing dynamic models...")
        
        # Control nodes
        num_control_nodes = self.config.get('num_control_nodes', 5000)
        self.control_nodes = ControlNodes(
            num_nodes=num_control_nodes,
            init_positions=self.gaussians.xyz.detach().clone(),  # Initialize from Gaussians
        ).to(self.device)
        
        print(f"  ✓ Initialized {num_control_nodes} control nodes")
        
        # Deformation network
        self.deformation_net = DeformationNetwork(
            spatial_freq=self.config.get('spatial_freq', 10),
            temporal_freq=self.config.get('temporal_freq', 6),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 8),
        ).to(self.device)
        
        print(f"  ✓ Initialized deformation network")
        
        # Precompute KNN for Linear Blend Skinning
        k = self.config.get('k_nearest', 4)
        print(f"\nPrecomputing KNN (k={k})...")
        
        self.knn_indices, self.knn_weights = find_knn(
            query_points=self.gaussians.xyz.detach(),
            source_points=self.control_nodes.positions.detach(),
            k=k,
        )
        
        print(f"  ✓ KNN computed: {self.knn_indices.shape}")
    
    def setup_renderer(self):
        """Initialize renderer."""
        print("\nInitializing renderer...")
        
        image_size = tuple(self.frames[0]['image'].shape)
        use_volume_renderer = self.config.get('use_volume_renderer', True)
        
        if use_volume_renderer:
            self.renderer = VolumeRenderer(
                image_size=image_size,
                chunk_size=self.config.get('chunk_size', 1000),
            ).to(self.device)
            print(f"  ✓ Initialized VolumeRenderer")
        else:
            self.renderer = Medical2DSliceRenderer(
                image_size=image_size[:2],
                num_slices=image_size[2] if len(image_size) > 2 else 32,
                chunk_size=self.config.get('chunk_size', 1000),
            ).to(self.device)
            print(f"  ✓ Initialized Medical2DSliceRenderer")
    
    def setup_optimizers(self):
        """Setup optimizers for all components."""
        print("\nSetting up optimizers...")
        
        # Gaussians (fine-tune with lower LR)
        lr_xyz = self.config.get('lr_xyz', 1.6e-4) * 0.1  # 10x lower
        lr_scale = self.config.get('lr_scale', 5e-3) * 0.1
        lr_rotation = self.config.get('lr_rotation', 1e-3) * 0.1
        lr_opacity = self.config.get('lr_opacity', 5e-2) * 0.1
        lr_features = self.config.get('lr_features', 2.5e-3) * 0.1
        
        gaussian_params = [
            {'params': [self.gaussians.xyz], 'lr': lr_xyz, 'name': 'xyz'},
            {'params': [self.gaussians.scale], 'lr': lr_scale, 'name': 'scale'},
            {'params': [self.gaussians.rotation], 'lr': lr_rotation, 'name': 'rotation'},
            {'params': [self.gaussians.opacity], 'lr': lr_opacity, 'name': 'opacity'},
            {'params': [self.gaussians.features], 'lr': lr_features, 'name': 'features'},
        ]
        
        # Control nodes
        lr_control = self.config.get('lr_control_nodes', 1.6e-4)
        control_params = [
            {'params': self.control_nodes.parameters(), 'lr': lr_control, 'name': 'control_nodes'},
        ]
        
        # Deformation network
        lr_deform = self.config.get('lr_deformation', 1e-4)
        deform_params = [
            {'params': self.deformation_net.parameters(), 'lr': lr_deform, 'name': 'deformation'},
        ]
        
        # Create optimizers
        self.optimizer_gaussians = torch.optim.Adam(gaussian_params)
        self.optimizer_control = torch.optim.Adam(control_params)
        self.optimizer_deform = torch.optim.Adam(deform_params)
        
        print(f"  ✓ Optimizer for Gaussians (fine-tune)")
        print(f"  ✓ Optimizer for Control Nodes")
        print(f"  ✓ Optimizer for Deformation Network")
    
    def setup_loss(self):
        """Setup loss function."""
        self.loss_fn = Dyna3DGRLoss(
            reconstruction_weight=self.config.get('reconstruction_weight', 1.0),
            temporal_weight=self.config.get('temporal_weight', 0.1),
            regularization_weight=self.config.get('regularization_weight', 0.01),
            cycle_weight=self.config.get('cycle_weight', 0.1),
        ).to(self.device)
        
        print("\nLoss function:")
        print(f"  Reconstruction weight: {self.config.get('reconstruction_weight', 1.0)}")
        print(f"  Temporal weight: {self.config.get('temporal_weight', 0.1)}")
        print(f"  Regularization weight: {self.config.get('regularization_weight', 0.01)}")
        print(f"  Cycle weight: {self.config.get('cycle_weight', 0.1)}")
    
    def setup_densification(self):
        """Setup Gaussian densification controller."""
        self.densification = GaussianDensificationController(
            grad_threshold=self.config.get('grad_threshold', 0.0002),
            opacity_threshold=self.config.get('opacity_threshold', 0.01),
            densify_interval=self.config.get('densify_interval', 500),
            densify_start_iter=self.config.get('densify_start_iter', 500),
            densify_stop_iter=self.config.get('densify_stop_iter', 15000),
        )
        
        print("\nGaussian densification:")
        print(f"  Interval: every {self.config.get('densify_interval', 500)} iterations")
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_loss = float('inf')
        
        # Save config
        config_path = self.output_dir / 'config_step2.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        print(f"\nConfig saved to: {config_path}")
    
    def forward_with_deformation(self, frame_idx):
        """Forward pass with deformation."""
        # Get timestamp
        t = frame_idx / len(self.frames)  # Normalize to [0, 1]
        t_tensor = torch.tensor([t], device=self.device)
        
        # Predict control node deformations
        control_deformations = self.deformation_net(
            self.control_nodes.positions,
            t_tensor.expand(len(self.control_nodes.positions)),
        )  # [M, 3]
        
        # Apply Linear Blend Skinning to get Gaussian deformations
        gaussian_deformations = self.control_nodes.apply_lbs(
            control_deformations=control_deformations,
            knn_indices=self.knn_indices,
            knn_weights=self.knn_weights,
        )  # [N, 3]
        
        # Deformed Gaussian positions
        deformed_xyz = self.gaussians.xyz + gaussian_deformations
        
        # Render with deformed positions
        if isinstance(self.renderer, VolumeRenderer):
            rendered = self.renderer(
                xyz=deformed_xyz,
                scale=self.gaussians.scale,
                rotation=self.gaussians.rotation,
                opacity=self.gaussians.opacity,
                features=self.gaussians.features,
            )
        else:
            D = self.frames[0]['image'].shape[2]
            slice_idx = D // 2
            slice_z = slice_idx / D
            
            rendered = self.renderer(
                xyz=deformed_xyz,
                scale=self.gaussians.scale,
                rotation=self.gaussians.rotation,
                opacity=self.gaussians.opacity,
                features=self.gaussians.features,
                slice_z=slice_z,
            )
            
            # Expand to volume
            H, W, F = rendered.shape
            rendered_volume = torch.zeros(H, W, D, F, device=rendered.device)
            rendered_volume[:, :, slice_idx, :] = rendered
            rendered = rendered_volume
        
        return rendered, gaussian_deformations
    
    def train_iteration(self, iteration):
        """Single training iteration."""
        # Zero gradients
        self.optimizer_gaussians.zero_grad()
        self.optimizer_control.zero_grad()
        self.optimizer_deform.zero_grad()
        
        # Sample random frames
        num_frames = len(self.frames)
        frame_indices = torch.randperm(num_frames)[:min(4, num_frames)]  # Sample up to 4 frames
        
        rendered_images = []
        target_images = []
        
        for frame_idx in frame_indices:
            # Forward pass
            rendered, deformations = self.forward_with_deformation(frame_idx.item())
            target = self.frames[frame_idx.item()]['image']
            
            rendered_images.append(rendered)
            target_images.append(target.unsqueeze(-1))
        
        # Stack
        rendered_images = torch.stack(rendered_images, dim=0)  # [B, H, W, D, F]
        target_images = torch.stack(target_images, dim=0)  # [B, H, W, D, 1]
        
        # Compute loss
        loss_dict = self.loss_fn(
            rendered_images=rendered_images,
            target_images=target_images,
            gaussians=self.gaussians,
            deformation_net=self.deformation_net,
        )
        
        total_loss = loss_dict['total']
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.gaussians.parameters()) +
                list(self.control_nodes.parameters()) +
                list(self.deformation_net.parameters()),
                self.config['grad_clip']
            )
        
        # Optimizer steps
        self.optimizer_gaussians.step()
        self.optimizer_control.step()
        self.optimizer_deform.step()
        
        # Densification
        if self.densification.should_densify(iteration):
            with torch.no_grad():
                old_num = len(self.gaussians.xyz)
                self.gaussians = self.densification.densify_and_prune(
                    gaussians=self.gaussians,
                    iteration=iteration,
                )
                new_num = len(self.gaussians.xyz)
                
                # Update KNN if Gaussians changed
                if new_num != old_num:
                    self.knn_indices, self.knn_weights = find_knn(
                        query_points=self.gaussians.xyz.detach(),
                        source_points=self.control_nodes.positions.detach(),
                        k=self.config.get('k_nearest', 4),
                    )
        
        return loss_dict
    
    def train(self):
        """Main training loop."""
        max_iterations = self.config.get('max_iterations', 10000)
        log_interval = self.config.get('log_interval', 10)
        save_interval = self.config.get('save_interval', 1000)
        
        print("\nStarting training...")
        print(f"Max iterations: {max_iterations}\n")
        
        pbar = tqdm(range(max_iterations), desc="Training")
        
        for iteration in pbar:
            # Train iteration
            loss_dict = self.train_iteration(iteration)
            
            # Logging
            if iteration % log_interval == 0:
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'recon': f"{loss_dict['reconstruction'].item():.4f}",
                    'temporal': f"{loss_dict['temporal'].item():.4f}",
                    'gaussians': len(self.gaussians.xyz),
                })
                
                # TensorBoard
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Loss/{key}', value.item(), iteration)
                
                self.writer.add_scalar('Model/num_gaussians', len(self.gaussians.xyz), iteration)
            
            # Save checkpoints
            if iteration % save_interval == 0 or iteration == max_iterations - 1:
                self.save_checkpoint(iteration, loss_dict['total'].item())
            
            # Update best model
            if loss_dict['total'].item() < self.best_loss:
                self.best_loss = loss_dict['total'].item()
                self.save_checkpoint('best', loss_dict['total'].item())
        
        print("\n" + "=" * 60)
        print("Training Completed")
        print("=" * 60)
        print(f"Total iterations: {max_iterations}")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Final Gaussians: {len(self.gaussians.xyz)}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("=" * 60 + "\n")
        
        self.writer.close()
    
    def save_checkpoint(self, iteration, loss):
        """Save checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'loss': loss,
            'gaussians': self.gaussians.state_dict(),
            'control_nodes': self.control_nodes.state_dict(),
            'deformation_net': self.deformation_net.state_dict(),
            'optimizer_gaussians': self.optimizer_gaussians.state_dict(),
            'optimizer_control': self.optimizer_control.state_dict(),
            'optimizer_deform': self.optimizer_deform.state_dict(),
            'config': self.config,
        }
        
        if isinstance(iteration, int):
            path = self.checkpoint_dir / f'iter_{iteration:06d}.pth'
        else:
            path = self.checkpoint_dir / f'{iteration}.pth'
        
        torch.save(checkpoint, path)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)


def main():
    parser = argparse.ArgumentParser(description='Step 2: Train Complete Dynamic Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--patient_dir', type=str, required=True, help='Path to patient directory')
    parser.add_argument('--step1_checkpoint', type=str, required=True, help='Path to Step 1 checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with step2 specific config if exists
    if 'step2' in config:
        config.update(config['step2'])
    
    # Create trainer
    trainer = Step2DynamicTrainer(
        config=config,
        patient_dir=args.patient_dir,
        step1_checkpoint=args.step1_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint('interrupted', trainer.best_loss)
        print(f"Checkpoint saved to: {trainer.checkpoint_dir}/interrupted.pth")


if __name__ == '__main__':
    main()
