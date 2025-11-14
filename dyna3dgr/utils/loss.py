"""
Loss Functions

This module implements various loss functions for training Dyna3DGR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ReconstructionLoss(nn.Module):
    """
    Image reconstruction loss (L1 + SSIM).
    """
    
    def __init__(self, l1_weight: float = 0.8, ssim_weight: float = 0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred: [B, C, H, W] predicted images
            target: [B, C, H, W] target images
        
        Returns:
            loss: scalar loss value
        """
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM loss
        ssim_loss = 1 - self._ssim(pred, target)
        
        # Combined loss
        loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        return loss
    
    def _ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ) -> torch.Tensor:
        """
        Compute SSIM (Structural Similarity Index).
        
        Args:
            pred: predicted images
            target: target images
            window_size: size of Gaussian window
            C1, C2: stability constants
        
        Returns:
            ssim: SSIM value
        """
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # Create 2D Gaussian kernel
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(pred.device)
        
        # Replicate kernel for each channel
        kernel = kernel.repeat(pred.shape[1], 1, 1, 1)
        
        # Compute means
        mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=pred.shape[1])
        mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=target.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu1_mu2
        
        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for smooth motion.
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize temporal consistency loss.
        
        Args:
            order: Order of temporal derivative (1 for velocity, 2 for acceleration)
        """
        super().__init__()
        self.order = order
    
    def forward(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            motion_sequence: [T, N, 3] motion vectors over time
        
        Returns:
            loss: scalar loss value
        """
        if self.order == 1:
            # First-order: penalize velocity changes
            diff = motion_sequence[1:] - motion_sequence[:-1]
            loss = torch.norm(diff, dim=-1).mean()
        elif self.order == 2:
            # Second-order: penalize acceleration
            diff1 = motion_sequence[1:] - motion_sequence[:-1]
            diff2 = diff1[1:] - diff1[:-1]
            loss = torch.norm(diff2, dim=-1).mean()
        else:
            raise ValueError(f"Unsupported order: {self.order}")
        
        return loss


class RegularizationLoss(nn.Module):
    """
    Regularization losses for Gaussian parameters.
    """
    
    def __init__(
        self,
        scale_weight: float = 0.01,
        opacity_weight: float = 0.01,
    ):
        super().__init__()
        self.scale_weight = scale_weight
        self.opacity_weight = opacity_weight
    
    def forward(
        self,
        scale: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute regularization loss.
        
        Args:
            scale: [N, 3] Gaussian scales
            opacity: [N, 1] Gaussian opacities
        
        Returns:
            loss: scalar loss value
        """
        # Scale regularization (prevent too large scales)
        scale_loss = torch.mean(torch.abs(scale))
        
        # Opacity regularization (encourage sparsity)
        opacity_loss = torch.mean(opacity)
        
        loss = self.scale_weight * scale_loss + self.opacity_weight * opacity_loss
        
        return loss


class CyclicConsistencyLoss(nn.Module):
    """
    Cyclic consistency loss for cardiac motion.
    
    Ensures that motion returns to the initial state after one cardiac cycle.
    """
    
    def forward(
        self,
        initial_positions: torch.Tensor,
        final_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cyclic consistency loss.
        
        Args:
            initial_positions: [N, 3] positions at t=0
            final_positions: [N, 3] positions at t=T (end of cycle)
        
        Returns:
            loss: scalar loss value
        """
        loss = F.mse_loss(final_positions, initial_positions)
        return loss


class Dyna3DGRLoss(nn.Module):
    """
    Combined loss for Dyna3DGR training.
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        temporal_weight: float = 0.1,
        reg_weight: float = 0.01,
        cyclic_weight: float = 0.05,
    ):
        """
        Initialize combined loss.
        
        Args:
            recon_weight: Weight for reconstruction loss
            temporal_weight: Weight for temporal consistency loss
            reg_weight: Weight for regularization loss
            cyclic_weight: Weight for cyclic consistency loss
        """
        super().__init__()
        
        self.recon_weight = recon_weight
        self.temporal_weight = temporal_weight
        self.reg_weight = reg_weight
        self.cyclic_weight = cyclic_weight
        
        self.recon_loss = ReconstructionLoss()
        self.temporal_loss = TemporalConsistencyLoss(order=1)
        self.reg_loss = RegularizationLoss()
        self.cyclic_loss = CyclicConsistencyLoss()
    
    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        motion_sequence: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        initial_positions: Optional[torch.Tensor] = None,
        final_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            pred_images: [B, C, H, W] predicted images
            target_images: [B, C, H, W] target images
            motion_sequence: [T, N, 3] motion vectors (optional)
            scale: [N, 3] Gaussian scales (optional)
            opacity: [N, 1] Gaussian opacities (optional)
            initial_positions: [N, 3] initial positions (optional)
            final_positions: [N, 3] final positions (optional)
        
        Returns:
            losses: Dictionary of loss values
        """
        losses = {}
        
        # Reconstruction loss
        l_recon = self.recon_loss(pred_images, target_images)
        losses['recon'] = l_recon
        
        # Temporal consistency loss
        if motion_sequence is not None:
            l_temporal = self.temporal_loss(motion_sequence)
            losses['temporal'] = l_temporal
        else:
            l_temporal = torch.tensor(0.0, device=pred_images.device)
            losses['temporal'] = l_temporal
        
        # Regularization loss
        if scale is not None and opacity is not None:
            l_reg = self.reg_loss(scale, opacity)
            losses['reg'] = l_reg
        else:
            l_reg = torch.tensor(0.0, device=pred_images.device)
            losses['reg'] = l_reg
        
        # Cyclic consistency loss
        if initial_positions is not None and final_positions is not None:
            l_cyclic = self.cyclic_loss(initial_positions, final_positions)
            losses['cyclic'] = l_cyclic
        else:
            l_cyclic = torch.tensor(0.0, device=pred_images.device)
            losses['cyclic'] = l_cyclic
        
        # Total loss
        total_loss = (
            self.recon_weight * l_recon +
            self.temporal_weight * l_temporal +
            self.reg_weight * l_reg +
            self.cyclic_weight * l_cyclic
        )
        losses['total'] = total_loss
        
        return losses
