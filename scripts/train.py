#!/usr/bin/env python3
"""
Training script for Dyna3DGR.

This script trains the Dyna3DGR model for 4D cardiac motion tracking.
Integrates ACDC data loader, Gaussian3D model, DeformationNetwork, and loss functions.
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

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.models import Gaussian3D, DeformationNetwork, initialize_gaussians_from_point_cloud
from dyna3dgr.utils.loss import Dyna3DGRLoss
from dyna3dgr.data import create_acdc_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Dyna3DGR')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to data root directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/experiment',
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
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (faster iterations, more logging)'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(output_dir):
    """Create output directories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    return output_dir


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(config, output_dir):
    """Save configuration to output directory."""
    config_file = output_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class Trainer:
    """Trainer class for Dyna3DGR."""
    
    def __init__(self, config, output_dir, device, debug=False):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory path
            device: Device to use
            debug: Enable debug mode
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.debug = debug
        
        # Setup models
        self.setup_models()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup loss
        self.setup_loss()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.epoch = 0
        self.iteration = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def setup_models(self):
        """Setup models."""
        print("Setting up models...")
        
        # 3D Gaussians
        self.gaussians = Gaussian3D(
            num_points=self.config['model']['gaussian']['num_points'],
            feature_dim=self.config['model']['gaussian']['feature_dim'],
            init_scale=self.config['model']['gaussian']['init_scale'],
            init_opacity=self.config['model']['gaussian']['init_opacity'],
        ).to(self.device)
        
        # Deformation network
        self.deformation_net = DeformationNetwork(
            spatial_freq=self.config['model']['deformation']['spatial_freq'],
            temporal_freq=self.config['model']['deformation']['temporal_freq'],
            hidden_dim=self.config['model']['deformation']['hidden_dim'],
            num_layers=self.config['model']['deformation']['num_layers'],
        ).to(self.device)
        
        # Count parameters
        gaussian_params = sum(p.numel() for p in self.gaussians.parameters())
        deformation_params = sum(p.numel() for p in self.deformation_net.parameters())
        total_params = gaussian_params + deformation_params
        
        print(f"Gaussians: {self.gaussians.num_points} points, {gaussian_params:,} parameters")
        print(f"Deformation network: {deformation_params:,} parameters")
        print(f"Total parameters: {total_params:,}")
    
    def setup_optimizer(self):
        """Setup optimizer."""
        print("Setting up optimizer...")
        
        # Combine parameters from both models
        params = [
            {
                'params': self.gaussians.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'gaussians'
            },
            {
                'params': self.deformation_net.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'deformation'
            },
        ]
        
        self.optimizer = optim.Adam(
            params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
        )
        
        # Learning rate scheduler
        scheduler_type = self.config['training']['lr_scheduler']['type']
        if scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config['training']['lr_scheduler']['gamma'],
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['lr_scheduler']['step_size'],
                gamma=self.config['training']['lr_scheduler']['gamma'],
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
            )
        else:
            self.scheduler = None
        
        print(f"Optimizer: Adam with lr={self.config['training']['learning_rate']}")
        print(f"Scheduler: {scheduler_type}")
    
    def setup_loss(self):
        """Setup loss function."""
        print("Setting up loss function...")
        
        self.criterion = Dyna3DGRLoss(
            recon_weight=self.config['training']['loss_weights']['reconstruction'],
            temporal_weight=self.config['training']['loss_weights']['temporal_consistency'],
            reg_weight=self.config['training']['loss_weights']['regularization'],
            cyclic_weight=self.config['training']['loss_weights']['cyclic_consistency'],
        ).to(self.device)
        
        print(f"Loss weights: {self.config['training']['loss_weights']}")
    
    def setup_logging(self):
        """Setup logging."""
        print("Setting up logging...")
        
        if self.config['training']['logging']['tensorboard']:
            log_dir = self.output_dir / 'logs'
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
    
    def forward_pass(self, batch):
        """
        Forward pass through the model.
        
        Args:
            batch: Batch of data
        
        Returns:
            outputs: Dictionary of outputs
            losses: Dictionary of losses
        """
        images = batch['images'].to(self.device)  # [B, T, H, W, D]
        timestamps = batch['timestamps'].to(self.device)  # [B, T]
        lengths = batch['lengths']  # [B]
        
        batch_size, max_time = images.shape[:2]
        
        # For now, we'll work with a simplified version
        # In full implementation, this would involve:
        # 1. Initialize Gaussians from first frame
        # 2. Apply deformation for each time step
        # 3. Render images
        # 4. Compute losses
        
        # Placeholder: Compute simple reconstruction loss
        # Get Gaussian positions
        gaussian_xyz = self.gaussians.xyz  # [N, 3]
        gaussian_features = self.gaussians.features  # [N, F]
        gaussian_opacity = self.gaussians.opacity  # [N, 1]
        gaussian_scale = self.gaussians.scale  # [N, 3]
        
        # Store motion sequence for temporal consistency
        motion_sequence = []
        
        # Process each time step
        for t in range(max_time):
            # Get time for this step
            t_normalized = timestamps[:, t:t+1]  # [B, 1]
            
            # Apply deformation
            deformed_xyz, deformed_alpha = self.deformation_net.apply_deformation(
                gaussian_xyz,
                gaussian_opacity,
                t_normalized[0],  # Use first batch item's time
            )
            
            motion_sequence.append(deformed_xyz.unsqueeze(0))  # [1, N, 3]
        
        # Stack motion sequence
        motion_sequence = torch.cat(motion_sequence, dim=0)  # [T, N, 3]
        
        # Placeholder reconstruction: Use simple intensity projection
        # In full implementation, this would use differentiable Gaussian splatting
        pred_images = self._placeholder_render(
            images,
            gaussian_xyz,
            gaussian_features,
            motion_sequence,
        )
        
        # Get initial and final positions for cyclic consistency
        initial_positions = motion_sequence[0]  # [N, 3]
        final_positions = motion_sequence[-1]  # [N, 3]
        
        # Compute losses
        losses = self.criterion(
            pred_images=pred_images,
            target_images=images,
            motion_sequence=motion_sequence,
            scale=gaussian_scale,
            opacity=gaussian_opacity,
            initial_positions=initial_positions,
            final_positions=final_positions,
        )
        
        outputs = {
            'pred_images': pred_images,
            'motion_sequence': motion_sequence,
            'deformed_positions': motion_sequence[-1],
        }
        
        return outputs, losses
    
    def _placeholder_render(self, target_images, gaussian_xyz, gaussian_features, motion_sequence):
        """
        Placeholder rendering function.
        
        In full implementation, this would use differentiable Gaussian splatting.
        For now, we return a simple prediction based on target images.
        
        Args:
            target_images: Target images [B, T, H, W, D]
            gaussian_xyz: Gaussian positions [N, 3]
            gaussian_features: Gaussian features [N, F]
            motion_sequence: Motion sequence [T, N, 3]
        
        Returns:
            pred_images: Predicted images [B, T, H, W, D]
        """
        # Simple placeholder: Add small noise to target images
        # This allows the loss to be computed and backpropagated
        noise = torch.randn_like(target_images) * 0.01
        pred_images = target_images + noise
        
        return pred_images
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.gaussians.train()
        self.deformation_net.train()
        
        epoch_losses = {
            'total': [],
            'recon': [],
            'temporal': [],
            'reg': [],
            'cyclic': [],
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            try:
                outputs, losses = self.forward_pass(batch)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                if self.debug:
                    raise
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.gaussians.parameters()) + list(self.deformation_net.parameters()),
                max_norm=1.0
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update iteration
            self.iteration += 1
            
            # Record losses
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key].append(losses[key].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'recon': losses['recon'].item(),
            })
            
            # Logging
            if self.writer and self.iteration % self.config['training']['logging']['log_interval'] == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}_loss', value.item(), self.iteration)
                
                # Log learning rate
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.iteration)
            
            # Densify and prune Gaussians
            if self.config['training']['densify']['enabled']:
                if (self.iteration >= self.config['training']['densify']['start_iter'] and
                    self.iteration % self.config['training']['densify']['interval'] == 0):
                    self.gaussians.densify(
                        grad_threshold=self.config['training']['densify']['grad_threshold'],
                        max_points=self.config['training']['densify']['max_points'],
                    )
            
            if self.config['training']['prune']['enabled']:
                if (self.iteration >= self.config['training']['prune']['start_iter'] and
                    self.iteration % self.config['training']['prune']['interval'] == 0):
                    self.gaussians.prune(
                        opacity_threshold=self.config['training']['prune']['opacity_threshold'],
                    )
            
            # Debug mode: only process a few batches
            if self.debug and batch_idx >= 2:
                break
        
        # Compute average losses
        avg_losses = {key: np.mean(values) if values else 0.0 
                      for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            avg_losses: Average validation losses
        """
        self.gaussians.eval()
        self.deformation_net.eval()
        
        val_losses = {
            'total': [],
            'recon': [],
            'temporal': [],
            'reg': [],
            'cyclic': [],
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    outputs, losses = self.forward_pass(batch)
                except Exception as e:
                    print(f"Error in validation: {e}")
                    if self.debug:
                        raise
                    continue
                
                # Record losses
                for key in val_losses.keys():
                    if key in losses:
                        val_losses[key].append(losses[key].item())
                
                # Debug mode: only process a few batches
                if self.debug and batch_idx >= 2:
                    break
        
        # Compute average losses
        avg_losses = {key: np.mean(values) if values else 0.0 
                      for key, values in val_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, is_best=False):
        """
        Save checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'gaussians_state_dict': self.gaussians.state_dict(),
            'deformation_net_state_dict': self.deformation_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (loss: {self.best_loss:.6f})")
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints (keep only last N)
        keep_last = self.config['training']['checkpoint']['keep_last']
        checkpoints = sorted(self.output_dir.glob('checkpoints/checkpoint_epoch_*.pth'))
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gaussians.load_state_dict(checkpoint['gaussians_state_dict'])
        self.deformation_net.load_state_dict(checkpoint['deformation_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Resumed from epoch {self.epoch}, iteration {self.iteration}")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        num_epochs = self.config['training']['num_epochs']
        start_epoch = self.epoch
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate(val_loader)
            self.val_losses.append(val_losses)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_losses['total']:.6f}")
            print(f"    - Reconstruction: {train_losses['recon']:.6f}")
            print(f"    - Temporal: {train_losses['temporal']:.6f}")
            print(f"    - Regularization: {train_losses['reg']:.6f}")
            print(f"    - Cyclic: {train_losses['cyclic']:.6f}")
            print(f"  Val Loss: {val_losses['total']:.6f}")
            print(f"  Gaussians: {self.gaussians.num_points} points")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6e}")
            
            # Logging
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_losses['total'], epoch)
                self.writer.add_scalar('epoch/val_loss', val_losses['total'], epoch)
                self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('epoch/num_gaussians', self.gaussians.num_points, epoch)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total']
            
            if (epoch + 1) % self.config['training']['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(is_best=is_best)
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_loss:.6f}")
        print("="*60)
        
        # Save final metrics
        self.save_metrics()
        
        if self.writer:
            self.writer.close()
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'final_epoch': self.epoch,
            'final_iteration': self.iteration,
        }
        
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {metrics_file}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['data']['data_root'] = args.data_root
    
    # Setup directories
    output_dir = setup_directories(args.output_dir)
    
    # Save configuration
    save_config(config, output_dir)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    
    try:
        train_loader = create_acdc_dataloader(
            data_root=config['data']['data_root'],
            split='train',
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=tuple(config['data']['image_size']),
            num_frames=config['data']['num_frames'],
            normalize=config['data']['normalize'],
            augmentation=config['data']['augmentation'],
            load_segmentation=True,
        )
        
        val_loader = create_acdc_dataloader(
            data_root=config['data']['data_root'],
            split='val',
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=tuple(config['data']['image_size']),
            num_frames=config['data']['num_frames'],
            normalize=config['data']['normalize'],
            augmentation=False,  # No augmentation for validation
            load_segmentation=True,
        )
        
        print(f"Train dataset: {len(train_loader.dataset)} patients, {len(train_loader)} batches")
        print(f"Val dataset: {len(val_loader.dataset)} patients, {len(val_loader)} batches")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("\nPlease ensure:")
        print("1. ACDC data is available at the specified path")
        print("2. Data has been preprocessed using scripts/preprocess_data.py")
        print("3. The data directory structure is correct")
        return
    
    # Create trainer
    trainer = Trainer(config, output_dir, device, debug=args.debug)
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)
        print("Checkpoint saved. You can resume training with --resume")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        if args.debug:
            raise
        print("Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)


if __name__ == '__main__':
    main()
