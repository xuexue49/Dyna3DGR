"""
Volume Renderer for Medical Images.

This module implements complete 3D volume rendering for medical images,
rendering all slices instead of just a single slice.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class VolumeRenderer(nn.Module):
    """
    Complete volume renderer for medical images.
    
    Renders all slices of a 3D volume using 2D Gaussian splatting.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 32),
        chunk_size: int = 1000,
        distance_threshold: float = 3.0,
    ):
        """
        Initialize volume renderer.
        
        Args:
            image_size: Volume size (H, W, D)
            chunk_size: Number of Gaussians to process at once
            distance_threshold: Distance threshold for Gaussian influence (in std devs)
        """
        super().__init__()
        
        self.image_size = image_size
        self.H, self.W, self.D = image_size
        self.chunk_size = chunk_size
        self.distance_threshold = distance_threshold
    
    def forward(
        self,
        xyz: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render complete 3D volume.
        
        Args:
            xyz: Gaussian positions [N, 3] in normalized space [0, 1]
            scale: Gaussian scales [N, 3]
            rotation: Gaussian rotations as quaternions [N, 4]
            opacity: Gaussian opacities [N, 1]
            features: Gaussian features [N, F] (e.g., intensity)
        
        Returns:
            rendered_volume: [H, W, D, F] rendered volume
        """
        N = xyz.shape[0]
        F = features.shape[1]
        
        # Initialize output volume
        rendered_volume = torch.zeros(
            self.H, self.W, self.D, F,
            dtype=features.dtype,
            device=features.device,
        )
        
        # Compute covariance matrices for all Gaussians
        covariances = self._compute_covariance_3d(scale, rotation)  # [N, 3, 3]
        
        # Render each slice
        for d in range(self.D):
            # Slice position in normalized space [0, 1]
            slice_z = d / (self.D - 1) if self.D > 1 else 0.5
            
            # Render this slice
            rendered_slice = self._render_slice(
                xyz=xyz,
                covariances=covariances,
                opacity=opacity,
                features=features,
                slice_z=slice_z,
            )  # [H, W, F]
            
            rendered_volume[:, :, d, :] = rendered_slice
        
        return rendered_volume
    
    def _compute_covariance_3d(
        self,
        scale: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 3D covariance matrices from scale and rotation.
        
        Σ = R S S^T R^T
        
        Args:
            scale: [N, 3]
            rotation: [N, 4] quaternions (w, x, y, z)
        
        Returns:
            covariances: [N, 3, 3]
        """
        N = scale.shape[0]
        
        # Build rotation matrices from quaternions
        R = self._quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]
        
        # Build scale matrices (diagonal)
        S = torch.diag_embed(scale)  # [N, 3, 3]
        
        # Compute Σ = R S S^T R^T
        RS = torch.bmm(R, S)  # [N, 3, 3]
        covariances = torch.bmm(RS, RS.transpose(-2, -1))  # [N, 3, 3]
        
        return covariances
    
    def _quaternion_to_rotation_matrix(
        self,
        quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices.
        
        Args:
            quaternion: [N, 4] (w, x, y, z)
        
        Returns:
            rotation_matrices: [N, 3, 3]
        """
        # Normalize quaternions
        quaternion = quaternion / (quaternion.norm(dim=-1, keepdim=True) + 1e-8)
        
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        
        # Build rotation matrix
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        R = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
        ], dim=-2)  # [N, 3, 3]
        
        return R
    
    def _render_slice(
        self,
        xyz: torch.Tensor,
        covariances: torch.Tensor,
        opacity: torch.Tensor,
        features: torch.Tensor,
        slice_z: float,
    ) -> torch.Tensor:
        """
        Render a single 2D slice at given z position.
        
        Uses conditional probability distribution:
        Given 3D Gaussian N(μ, Σ) and slice position z = z_slice,
        the 2D marginal distribution is:
          μ_2D = μ_xy + Σ_xy,z * Σ_zz^-1 * (z_slice - μ_z)
          Σ_2D = Σ_xy - Σ_xy,z * Σ_zz^-1 * Σ_z,xy
        
        Args:
            xyz: [N, 3] positions in [0, 1]
            covariances: [N, 3, 3] covariance matrices
            opacity: [N, 1]
            features: [N, F]
            slice_z: Slice z-position in [0, 1]
        
        Returns:
            rendered_slice: [H, W, F]
        """
        N = xyz.shape[0]
        F = features.shape[1]
        
        # Filter Gaussians close to this slice
        z_distances = torch.abs(xyz[:, 2] - slice_z)  # [N]
        
        # Estimate z-std from covariance
        z_std = torch.sqrt(covariances[:, 2, 2] + 1e-8)  # [N]
        
        # Keep Gaussians within distance_threshold standard deviations
        mask = z_distances < (self.distance_threshold * z_std)  # [N]
        
        if mask.sum() == 0:
            # No Gaussians influence this slice
            return torch.zeros(self.H, self.W, F, device=xyz.device)
        
        # Filter
        xyz_filtered = xyz[mask]  # [N', 3]
        cov_filtered = covariances[mask]  # [N', 3, 3]
        opacity_filtered = opacity[mask]  # [N', 1]
        features_filtered = features[mask]  # [N', F]
        
        # Project to 2D slice using conditional distribution
        mu_2d, cov_2d = self._project_to_slice(
            xyz_filtered,
            cov_filtered,
            slice_z,
        )  # [N', 2], [N', 2, 2]
        
        # Render 2D slice
        rendered_slice = self._render_2d_gaussians(
            mu_2d=mu_2d,
            cov_2d=cov_2d,
            opacity=opacity_filtered,
            features=features_filtered,
        )  # [H, W, F]
        
        return rendered_slice
    
    def _project_to_slice(
        self,
        xyz: torch.Tensor,
        covariances: torch.Tensor,
        slice_z: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D Gaussians to 2D slice using conditional distribution.
        
        Args:
            xyz: [N, 3]
            covariances: [N, 3, 3]
            slice_z: Scalar
        
        Returns:
            mu_2d: [N, 2]
            cov_2d: [N, 2, 2]
        """
        # Extract components
        mu_xy = xyz[:, :2]  # [N, 2]
        mu_z = xyz[:, 2:3]  # [N, 1]
        
        Sigma_xy = covariances[:, :2, :2]  # [N, 2, 2]
        Sigma_xy_z = covariances[:, :2, 2:3]  # [N, 2, 1]
        Sigma_z_xy = covariances[:, 2:3, :2]  # [N, 1, 2]
        Sigma_zz = covariances[:, 2:3, 2:3]  # [N, 1, 1]
        
        # Conditional mean: μ_2D = μ_xy + Σ_xy,z * Σ_zz^-1 * (z_slice - μ_z)
        Sigma_zz_inv = 1.0 / (Sigma_zz + 1e-8)  # [N, 1, 1]
        delta_z = slice_z - mu_z  # [N, 1]
        
        # [N, 2, 1] * [N, 1, 1] * [N, 1] -> [N, 2, 1] -> [N, 2]
        correction = torch.bmm(Sigma_xy_z, Sigma_zz_inv * delta_z.unsqueeze(-1)).squeeze(-1)
        mu_2d = mu_xy + correction  # [N, 2]
        
        # Conditional covariance: Σ_2D = Σ_xy - Σ_xy,z * Σ_zz^-1 * Σ_z,xy
        cov_2d = Sigma_xy - torch.bmm(
            Sigma_xy_z * Sigma_zz_inv,
            Sigma_z_xy,
        )  # [N, 2, 2]
        
        return mu_2d, cov_2d
    
    def _render_2d_gaussians(
        self,
        mu_2d: torch.Tensor,
        cov_2d: torch.Tensor,
        opacity: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render 2D Gaussians to image.
        
        Args:
            mu_2d: [N, 2] positions in [0, 1]
            cov_2d: [N, 2, 2] covariance matrices
            opacity: [N, 1]
            features: [N, F]
        
        Returns:
            rendered: [H, W, F]
        """
        N = mu_2d.shape[0]
        F = features.shape[1]
        
        # Create pixel grid in [0, 1]
        y_grid = torch.linspace(0, 1, self.H, device=mu_2d.device)
        x_grid = torch.linspace(0, 1, self.W, device=mu_2d.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        pixel_coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        
        # Initialize output
        rendered = torch.zeros(self.H, self.W, F, device=mu_2d.device)
        alpha_accumulated = torch.zeros(self.H, self.W, 1, device=mu_2d.device)
        
        # Render in chunks to save memory
        for chunk_start in range(0, N, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, N)
            
            mu_chunk = mu_2d[chunk_start:chunk_end]  # [C, 2]
            cov_chunk = cov_2d[chunk_start:chunk_end]  # [C, 2, 2]
            opacity_chunk = opacity[chunk_start:chunk_end]  # [C, 1]
            features_chunk = features[chunk_start:chunk_end]  # [C, F]
            
            # Compute Gaussian values at all pixels
            # [H, W, 2] - [C, 2] -> [H, W, C, 2]
            delta = pixel_coords.unsqueeze(2) - mu_chunk.unsqueeze(0).unsqueeze(0)  # [H, W, C, 2]
            
            # Compute precision matrices (inverse covariance)
            # Add small epsilon for numerical stability
            cov_chunk_stable = cov_chunk + torch.eye(2, device=cov_chunk.device).unsqueeze(0) * 1e-6
            precision = torch.inverse(cov_chunk_stable)  # [C, 2, 2]
            
            # Mahalanobis distance: d^2 = (x - μ)^T Σ^-1 (x - μ)
            # [H, W, C, 1, 2] @ [C, 2, 2] @ [H, W, C, 2, 1]
            delta_expanded = delta.unsqueeze(-2)  # [H, W, C, 1, 2]
            mahalanobis = torch.matmul(
                torch.matmul(delta_expanded, precision.unsqueeze(0).unsqueeze(0)),
                delta.unsqueeze(-1),
            ).squeeze(-1).squeeze(-1)  # [H, W, C]
            
            # Gaussian weight: exp(-0.5 * d^2)
            gaussian_weight = torch.exp(-0.5 * mahalanobis)  # [H, W, C]
            
            # Apply opacity
            alpha = opacity_chunk.squeeze(-1) * gaussian_weight  # [H, W, C]
            
            # Alpha compositing (front-to-back)
            # C_out = C_out + (1 - α_acc) * α_i * C_i
            transmittance = 1.0 - alpha_accumulated  # [H, W, 1]
            
            # Accumulate color
            weighted_features = alpha.unsqueeze(-1) * features_chunk.unsqueeze(0).unsqueeze(0)  # [H, W, C, F]
            rendered += transmittance * weighted_features.sum(dim=2)  # [H, W, F]
            
            # Accumulate alpha
            alpha_accumulated += transmittance * alpha.sum(dim=2, keepdim=True)  # [H, W, 1]
        
        return rendered


def render_volume(
    gaussians,
    image_size: Tuple[int, int, int] = (128, 128, 32),
    chunk_size: int = 1000,
) -> torch.Tensor:
    """
    Convenience function to render a complete volume.
    
    Args:
        gaussians: Gaussian3D model
        image_size: Volume size (H, W, D)
        chunk_size: Chunk size for rendering
    
    Returns:
        rendered_volume: [H, W, D, F] rendered volume
    
    Example:
        >>> from dyna3dgr.models import Gaussian3D
        >>> from dyna3dgr.rendering import render_volume
        >>> 
        >>> gaussians = Gaussian3D(num_points=5000, feature_dim=1)
        >>> volume = render_volume(gaussians, image_size=(128, 128, 32))
        >>> print(volume.shape)  # [128, 128, 32, 1]
    """
    renderer = VolumeRenderer(
        image_size=image_size,
        chunk_size=chunk_size,
    ).to(gaussians.xyz.device)
    
    return renderer(
        xyz=gaussians.xyz,
        scale=gaussians.scale,
        rotation=gaussians.rotation,
        opacity=gaussians.opacity,
        features=gaussians.features,
    )
