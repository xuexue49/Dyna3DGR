"""
Step 1: Train Gaussian Representation Only (Static Reconstruction)

This script trains only the 3D Gaussian representation on the ED (End-Diastole) frame
to establish a stable static reconstruction before introducing dynamic deformation.

This is the first step of a two-step training pipeline:
  Step 1: Train Gaussians only (this script)
  Step 2: Train complete dynamic model (train_step2_dynamic.py)
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.models import Gaussian3D, GaussianDensificationController
from dyna3dgr.data import PatientDataset, initialize_from_segmentation, initialize_uniform_grid
from dyna3dgr.rendering import VolumeRenderer, Medical2DSliceRenderer
from dyna3dgr.utils import Dyna3DGRLoss


class Step1GaussianTrainer:
    """
    Trainer for Step 1: Gaussian representation only.
    
    This trainer focuses on optimizing the 3D Gaussian representation
    on the ED frame to establish a high-quality static reconstruction.
    """
    
    def __init__(self, config, patient_dir, output_dir, device='cuda'):
        self.config = config
        self.patient_dir = Path(patient_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup components
        self.setup_data()
        self.setup_models()
        self.setup_renderer()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_densification()
        self.setup_logging()
        
        print("\n" + "=" * 60)
        print("Step 1: Training Gaussian Representation Only")
        print("=" * 60)
        print(f"Patient: {self.patient_dir.name}")
        print(f"Max iterations: {self.config['max_iterations']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")
    
    def setup_data(self):
        """Load patient data (ED frame only for static reconstruction)."""
        print(f"Loading patient data from: {self.patient_dir}")
        
        self.dataset = PatientDataset(
            patient_dir=str(self.patient_dir),
            load_segmentation=True,
        )
        
        # Get ED frame (first frame, typically)
        self.ed_frame = self.dataset[0]
        
        print(f"  Loaded {len(self.dataset)} frames")
        print(f"  ED frame shape: {self.ed_frame['image'].shape}")
        
        if 'segmentation' in self.ed_frame:
            print(f"  ED segmentation available: {self.ed_frame['segmentation'].shape}")
        
        # Move to device
        self.ed_image = self.ed_frame['image'].to(self.device)
        if 'segmentation' in self.ed_frame:
            self.ed_seg = self.ed_frame['segmentation'].to(self.device)
        else:
            self.ed_seg = None
    
    def setup_models(self):
        """Initialize Gaussian model."""
        print("\nInitializing Gaussian model...")
        
        num_gaussians = self.config.get('num_gaussians', 5000)
        feature_dim = self.config.get('feature_dim', 1)
        
        # Initialize Gaussian positions
        if self.config.get('init_from_segmentation', True) and self.ed_seg is not None:
            print("  Initializing from segmentation mask...")
            positions = initialize_from_segmentation(
                segmentation=self.ed_seg.cpu().numpy(),
                num_gaussians=num_gaussians,
                foreground_labels=self.config.get('foreground_labels', [1, 2, 3]),
                normalize=True,
                add_noise=True,
            )
        else:
            print("  Initializing from uniform grid...")
            positions = initialize_uniform_grid(
                shape=self.ed_image.shape,
                num_gaussians=num_gaussians,
                normalize=True,
                add_noise=True,
            )
        
        # Create Gaussian model
        self.gaussians = Gaussian3D(
            num_points=num_gaussians,
            feature_dim=feature_dim,
            init_positions=positions,
        ).to(self.device)
        
        print(f"  ✓ Initialized {num_gaussians} Gaussians")
        print(f"  Position range: [{self.gaussians.xyz.min():.4f}, {self.gaussians.xyz.max():.4f}]")
    
    def setup_renderer(self):
        """Initialize renderer."""
        print("\nInitializing renderer...")
        
        image_size = tuple(self.ed_image.shape)
        use_volume_renderer = self.config.get('use_volume_renderer', True)
        
        if use_volume_renderer:
            self.renderer = VolumeRenderer(
                image_size=image_size,
                chunk_size=self.config.get('chunk_size', 1000),
            ).to(self.device)
            print(f"  ✓ Initialized VolumeRenderer (complete 3D rendering)")
        else:
            self.renderer = Medical2DSliceRenderer(
                image_size=image_size[:2],
                num_slices=image_size[2] if len(image_size) > 2 else 32,
                chunk_size=self.config.get('chunk_size', 1000),
            ).to(self.device)
            print(f"  ✓ Initialized Medical2DSliceRenderer (single slice)")
    
    def setup_optimizer(self):
        """Setup optimizer for Gaussians only."""
        print("\nSetting up optimizer...")
        
        # Learning rates from config
        lr_xyz = self.config.get('lr_xyz', 1.6e-4)
        lr_scale = self.config.get('lr_scale', 5e-3)
        lr_rotation = self.config.get('lr_rotation', 1e-3)
        lr_opacity = self.config.get('lr_opacity', 5e-2)
        lr_features = self.config.get('lr_features', 2.5e-3)
        
        # Create parameter groups
        param_groups = [
            {'params': [self.gaussians.xyz], 'lr': lr_xyz, 'name': 'xyz'},
            {'params': [self.gaussians.scale], 'lr': lr_scale, 'name': 'scale'},
            {'params': [self.gaussians.rotation], 'lr': lr_rotation, 'name': 'rotation'},
            {'params': [self.gaussians.opacity], 'lr': lr_opacity, 'name': 'opacity'},
            {'params': [self.gaussians.features], 'lr': lr_features, 'name': 'features'},
        ]
        
        self.optimizer = torch.optim.Adam(param_groups)
        
        # Learning rate scheduler
        scheduler_type = self.config.get('lr_scheduler', 'exponential')
        if scheduler_type == 'exponential':
            gamma = self.config.get('lr_gamma', 0.99)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        else:
            self.scheduler = None
        
        print(f"  ✓ Optimizer: Adam")
        print(f"  ✓ LR scheduler: {scheduler_type}")
    
    def setup_loss(self):
        """Setup loss function."""
        self.loss_fn = Dyna3DGRLoss(
            recon_weight=self.config.get('reconstruction_weight', 1.0),
            temporal_weight=0.0,  # No temporal loss for static reconstruction
            reg_weight=self.config.get('regularization_weight', 0.01),
            cyclic_weight=0.0,  # No cycle loss for static reconstruction
        ).to(self.device)
        
        print("\nLoss function:")
        print(f"  Reconstruction weight: {self.config.get('reconstruction_weight', 1.0)}")
        print(f"  Regularization weight: {self.config.get('regularization_weight', 0.01)}")
    
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
        print(f"  Start: iteration {self.config.get('densify_start_iter', 500)}")
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_loss = float('inf')
        
        # Save config
        config_path = self.output_dir / 'config_step1.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        print(f"\nConfig saved to: {config_path}")
    
    def render_ed_frame(self):
        """Render the ED frame."""
        if isinstance(self.renderer, VolumeRenderer):
            # Complete volume rendering
            rendered = self.renderer(
                xyz=self.gaussians.xyz,
                scale=self.gaussians.scale,
                rotation=self.gaussians.rotation,
                opacity=self.gaussians.opacity,
                features=self.gaussians.features,
            )  # [H, W, D, F]
        else:
            # Single slice rendering (middle slice)
            D = self.ed_image.shape[2]
            slice_idx = D // 2
            slice_z = slice_idx / D
            
            rendered = self.renderer(
                xyz=self.gaussians.xyz,
                scale=self.gaussians.scale,
                rotation=self.gaussians.rotation,
                opacity=self.gaussians.opacity,
                features=self.gaussians.features,
                slice_z=slice_z,
            )  # [H, W, F]
            
            # Expand to volume (only middle slice is valid)
            H, W, F = rendered.shape
            rendered_volume = torch.zeros(H, W, D, F, device=rendered.device)
            rendered_volume[:, :, slice_idx, :] = rendered
            rendered = rendered_volume
        
        return rendered
    
    def train_iteration(self, iteration):
        """Single training iteration."""
        self.optimizer.zero_grad()
        
        # Render ED frame
        rendered = self.render_ed_frame()  # [H, W, D, F]
        
        # Compute loss
        target = self.ed_image.unsqueeze(-1)  # [H, W, D, 1]
        
        loss_dict = self.loss_fn(
            rendered_images=rendered.unsqueeze(0),  # [1, H, W, D, F]
            target_images=target.unsqueeze(0),  # [1, H, W, D, 1]
            gaussians=self.gaussians,
        )
        
        total_loss = loss_dict['total']
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.gaussians.parameters(),
                self.config['grad_clip']
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Densification
        if self.densification.should_densify(iteration):
            with torch.no_grad():
                self.gaussians = self.densification.densify_and_prune(
                    gaussians=self.gaussians,
                    iteration=iteration,
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
                    'gaussians': len(self.gaussians.xyz),
                })
                
                # TensorBoard
                self.writer.add_scalar('Loss/total', loss_dict['total'].item(), iteration)
                self.writer.add_scalar('Loss/reconstruction', loss_dict['reconstruction'].item(), iteration)
                self.writer.add_scalar('Loss/regularization', loss_dict['regularization'].item(), iteration)
                self.writer.add_scalar('Model/num_gaussians', len(self.gaussians.xyz), iteration)
                
                # Learning rates
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(
                        f'LR/{param_group["name"]}',
                        param_group['lr'],
                        iteration
                    )
            
            # Save checkpoints
            if iteration % save_interval == 0 or iteration == max_iterations - 1:
                self.save_checkpoint(iteration, loss_dict['total'].item())
            
            # Update best model
            if loss_dict['total'].item() < self.best_loss:
                self.best_loss = loss_dict['total'].item()
                self.save_checkpoint('best', loss_dict['total'].item())
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
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
            'optimizer': self.optimizer.state_dict(),
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
    parser = argparse.ArgumentParser(description='Step 1: Train Gaussian Representation Only')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--patient_dir', type=str, required=True, help='Path to patient directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with step1 specific config if exists
    if 'step1' in config:
        config.update(config['step1'])
    
    # Create trainer
    trainer = Step1GaussianTrainer(
        config=config,
        patient_dir=args.patient_dir,
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
