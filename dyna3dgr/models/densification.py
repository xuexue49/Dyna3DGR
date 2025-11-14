"""
Gaussian densification and pruning utilities.

Implements adaptive densification as described in the Dyna3DGR paper:
"Gaussian densification is performed every 500 iterations starting 
from the 500th iteration."
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


class GaussianDensificationController:
    """
    Controller for adaptive Gaussian densification and pruning.
    
    Performs:
    - Splitting: Large Gaussians with high gradients
    - Cloning: Small Gaussians with high gradients
    - Pruning: Gaussians with low opacity
    """
    
    def __init__(
        self,
        grad_threshold: float = 0.0002,
        opacity_threshold: float = 0.01,
        scale_threshold: float = 0.1,
        densify_interval: int = 500,
        densify_start_iter: int = 500,
        densify_end_iter: Optional[int] = None,
        max_gaussians: Optional[int] = None,
    ):
        """
        Initialize densification controller.
        
        Args:
            grad_threshold: Gradient threshold for densification
            opacity_threshold: Opacity threshold for pruning
            scale_threshold: Scale threshold for splitting vs cloning
            densify_interval: Interval between densification operations
            densify_start_iter: Iteration to start densification
            densify_end_iter: Iteration to stop densification (None = no limit)
            max_gaussians: Maximum number of Gaussians (None = no limit)
        """
        self.grad_threshold = grad_threshold
        self.opacity_threshold = opacity_threshold
        self.scale_threshold = scale_threshold
        self.densify_interval = densify_interval
        self.densify_start_iter = densify_start_iter
        self.densify_end_iter = densify_end_iter
        self.max_gaussians = max_gaussians
        
        # Statistics tracking
        self.grad_accum = None
        self.grad_count = 0
    
    def should_densify(self, iteration: int) -> bool:
        """
        Check if densification should be performed at this iteration.
        
        Args:
            iteration: Current training iteration
        
        Returns:
            should_densify: Whether to perform densification
        """
        if iteration < self.densify_start_iter:
            return False
        
        if self.densify_end_iter is not None and iteration >= self.densify_end_iter:
            return False
        
        if iteration % self.densify_interval != 0:
            return False
        
        return True
    
    def accumulate_gradients(self, xyz_grad: torch.Tensor):
        """
        Accumulate gradients for densification decisions.
        
        Args:
            xyz_grad: [N, 3] gradients of Gaussian positions
        """
        grad_norm = torch.norm(xyz_grad, dim=-1, keepdim=True)  # [N, 1]
        
        if self.grad_accum is None:
            self.grad_accum = grad_norm
        else:
            self.grad_accum += grad_norm
        
        self.grad_count += 1
    
    def get_average_gradients(self) -> Optional[torch.Tensor]:
        """
        Get average accumulated gradients.
        
        Returns:
            avg_grad: [N, 1] average gradient norms or None
        """
        if self.grad_accum is None or self.grad_count == 0:
            return None
        
        return self.grad_accum / self.grad_count
    
    def reset_gradients(self):
        """Reset accumulated gradients."""
        self.grad_accum = None
        self.grad_count = 0
    
    def densify_and_prune(
        self,
        gaussians: 'Gaussian3D',
    ) -> Tuple[int, int, int]:
        """
        Perform densification and pruning on Gaussians.
        
        Args:
            gaussians: Gaussian3D object to modify
        
        Returns:
            num_split: Number of Gaussians split
            num_cloned: Number of Gaussians cloned
            num_pruned: Number of Gaussians pruned
        """
        # Get average gradients
        avg_grad = self.get_average_gradients()
        
        if avg_grad is None:
            return 0, 0, 0
        
        # Identify Gaussians to densify
        high_grad_mask = avg_grad.squeeze() > self.grad_threshold
        
        # Get scale values
        scale_values = gaussians.scale  # [N, 3]
        max_scale = scale_values.max(dim=-1)[0]  # [N]
        
        # Split: high gradient + large scale
        split_mask = high_grad_mask & (max_scale > self.scale_threshold)
        
        # Clone: high gradient + small scale
        clone_mask = high_grad_mask & (max_scale <= self.scale_threshold)
        
        # Prune: low opacity
        prune_mask = gaussians.opacity.squeeze() < self.opacity_threshold
        
        # Perform operations
        num_split = self._split_gaussians(gaussians, split_mask)
        num_cloned = self._clone_gaussians(gaussians, clone_mask)
        num_pruned = self._prune_gaussians(gaussians, prune_mask)
        
        # Reset gradients
        self.reset_gradients()
        
        # Check max Gaussians limit
        if self.max_gaussians is not None and gaussians.num_points > self.max_gaussians:
            self._limit_gaussians(gaussians, self.max_gaussians)
        
        return num_split, num_cloned, num_pruned
    
    def _split_gaussians(
        self,
        gaussians: 'Gaussian3D',
        mask: torch.Tensor,
    ) -> int:
        """
        Split Gaussians into two smaller ones.
        
        Args:
            gaussians: Gaussian3D object
            mask: [N] boolean mask of Gaussians to split
        
        Returns:
            num_split: Number of Gaussians split
        """
        num_split = mask.sum().item()
        
        if num_split == 0:
            return 0
        
        # Get parameters of Gaussians to split
        split_xyz = gaussians._xyz[mask]
        split_scale = gaussians._scale[mask]
        split_rotation = gaussians._rotation[mask]
        split_opacity = gaussians._opacity[mask]
        split_features = gaussians._features[mask]
        
        # Get scale values and find max scale dimension
        scale_values = torch.exp(split_scale)  # [num_split, 3]
        max_scale_dim = torch.argmax(scale_values, dim=-1)  # [num_split]
        
        # Create two new Gaussians for each split
        # Split along the maximum scale dimension
        
        # Reduce scale
        new_scale = split_scale - torch.log(torch.tensor(1.6))  # Scale down by 1.6
        
        # Create offset along max scale dimension
        offset = torch.zeros_like(split_xyz)  # [num_split, 3]
        for i in range(num_split):
            dim = max_scale_dim[i]
            offset[i, dim] = scale_values[i, dim] * 0.5
        
        # First child: offset in positive direction
        new_xyz_1 = split_xyz + offset
        
        # Second child: offset in negative direction
        new_xyz_2 = split_xyz - offset
        
        # Reduce opacity
        new_opacity = split_opacity * 0.5
        
        # Concatenate both children
        new_xyz = torch.cat([new_xyz_1, new_xyz_2], dim=0)
        new_scale_all = torch.cat([new_scale, new_scale], dim=0)
        new_rotation = torch.cat([split_rotation, split_rotation], dim=0)
        new_opacity_all = torch.cat([new_opacity, new_opacity], dim=0)
        new_features = torch.cat([split_features, split_features], dim=0)
        
        # Remove original Gaussians and add new ones
        keep_mask = ~mask
        gaussians._xyz = nn.Parameter(
            torch.cat([gaussians._xyz[keep_mask], new_xyz], dim=0)
        )
        gaussians._scale = nn.Parameter(
            torch.cat([gaussians._scale[keep_mask], new_scale_all], dim=0)
        )
        gaussians._rotation = nn.Parameter(
            torch.cat([gaussians._rotation[keep_mask], new_rotation], dim=0)
        )
        gaussians._opacity = nn.Parameter(
            torch.cat([gaussians._opacity[keep_mask], new_opacity_all], dim=0)
        )
        gaussians._features = nn.Parameter(
            torch.cat([gaussians._features[keep_mask], new_features], dim=0)
        )
        
        gaussians.num_points = gaussians._xyz.shape[0]
        
        return num_split
    
    def _clone_gaussians(
        self,
        gaussians: 'Gaussian3D',
        mask: torch.Tensor,
    ) -> int:
        """
        Clone Gaussians.
        
        Args:
            gaussians: Gaussian3D object
            mask: [N] boolean mask of Gaussians to clone
        
        Returns:
            num_cloned: Number of Gaussians cloned
        """
        num_cloned = mask.sum().item()
        
        if num_cloned == 0:
            return 0
        
        # Get parameters of Gaussians to clone
        clone_xyz = gaussians._xyz[mask]
        clone_scale = gaussians._scale[mask]
        clone_rotation = gaussians._rotation[mask]
        clone_opacity = gaussians._opacity[mask]
        clone_features = gaussians._features[mask]
        
        # Add small perturbation to position
        clone_xyz = clone_xyz + torch.randn_like(clone_xyz) * 0.01
        
        # Concatenate with existing Gaussians
        gaussians._xyz = nn.Parameter(
            torch.cat([gaussians._xyz, clone_xyz], dim=0)
        )
        gaussians._scale = nn.Parameter(
            torch.cat([gaussians._scale, clone_scale], dim=0)
        )
        gaussians._rotation = nn.Parameter(
            torch.cat([gaussians._rotation, clone_rotation], dim=0)
        )
        gaussians._opacity = nn.Parameter(
            torch.cat([gaussians._opacity, clone_opacity], dim=0)
        )
        gaussians._features = nn.Parameter(
            torch.cat([gaussians._features, clone_features], dim=0)
        )
        
        gaussians.num_points = gaussians._xyz.shape[0]
        
        return num_cloned
    
    def _prune_gaussians(
        self,
        gaussians: 'Gaussian3D',
        mask: torch.Tensor,
    ) -> int:
        """
        Prune Gaussians with low opacity.
        
        Args:
            gaussians: Gaussian3D object
            mask: [N] boolean mask of Gaussians to prune
        
        Returns:
            num_pruned: Number of Gaussians pruned
        """
        num_pruned = mask.sum().item()
        
        if num_pruned == 0:
            return 0
        
        # Keep only Gaussians not marked for pruning
        keep_mask = ~mask
        
        if keep_mask.sum() == 0:
            # Don't prune all Gaussians
            return 0
        
        gaussians._xyz = nn.Parameter(gaussians._xyz[keep_mask])
        gaussians._scale = nn.Parameter(gaussians._scale[keep_mask])
        gaussians._rotation = nn.Parameter(gaussians._rotation[keep_mask])
        gaussians._opacity = nn.Parameter(gaussians._opacity[keep_mask])
        gaussians._features = nn.Parameter(gaussians._features[keep_mask])
        
        gaussians.num_points = gaussians._xyz.shape[0]
        
        return num_pruned
    
    def _limit_gaussians(
        self,
        gaussians: 'Gaussian3D',
        max_gaussians: int,
    ):
        """
        Limit number of Gaussians by removing those with lowest opacity.
        
        Args:
            gaussians: Gaussian3D object
            max_gaussians: Maximum number of Gaussians to keep
        """
        if gaussians.num_points <= max_gaussians:
            return
        
        # Sort by opacity and keep top-k
        opacity = gaussians.opacity.squeeze()  # [N]
        _, indices = torch.topk(opacity, k=max_gaussians, largest=True)
        
        # Sort indices to maintain order
        indices, _ = torch.sort(indices)
        
        gaussians._xyz = nn.Parameter(gaussians._xyz[indices])
        gaussians._scale = nn.Parameter(gaussians._scale[indices])
        gaussians._rotation = nn.Parameter(gaussians._rotation[indices])
        gaussians._opacity = nn.Parameter(gaussians._opacity[indices])
        gaussians._features = nn.Parameter(gaussians._features[indices])
        
        gaussians.num_points = gaussians._xyz.shape[0]
    
    def get_statistics(self) -> dict:
        """
        Get densification statistics.
        
        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            'grad_threshold': self.grad_threshold,
            'opacity_threshold': self.opacity_threshold,
            'scale_threshold': self.scale_threshold,
            'densify_interval': self.densify_interval,
            'densify_start_iter': self.densify_start_iter,
            'densify_end_iter': self.densify_end_iter,
            'max_gaussians': self.max_gaussians,
            'grad_count': self.grad_count,
        }
        
        if self.grad_accum is not None:
            stats['avg_grad_norm'] = self.grad_accum.mean().item() / max(self.grad_count, 1)
        
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GaussianDensificationController(\n"
            f"  grad_threshold={self.grad_threshold},\n"
            f"  opacity_threshold={self.opacity_threshold},\n"
            f"  scale_threshold={self.scale_threshold},\n"
            f"  densify_interval={self.densify_interval},\n"
            f"  densify_start_iter={self.densify_start_iter},\n"
            f"  densify_end_iter={self.densify_end_iter},\n"
            f"  max_gaussians={self.max_gaussians}\n"
            f")"
        )
