"""
Medical Image Renderer for 2D Slices.

This module implements Gaussian Splatting rendering specifically for
medical images, which are represented as 2D slice sequences.

Key differences from standard rendering:
- Renders 2D slices instead of 3D volumes
- Uses orthographic projection
- No antialiasing or prefiltering
- Fixed camera positions for each slice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class Medical2DSliceRenderer(nn.Module):
    """
    Renderer for medical images as 2D slice sequences.
    
    Unlike standard 3D volume rendering, this renderer:
    1. Treats medical data as consecutive 2D slices in 3D space
    2. Uses orthographic projection (no perspective distortion)
    3. Renders each slice independently
    4. Preserves exact pixel values (no antialiasing)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_slices: int = 10,
        slice_spacing: float = 1.0,
        background_value: float = 0.0,
        chunk_size: int = 500,
    ):
        """
        Initialize medical 2D slice renderer.
        
        Args:
            image_size: Size of each 2D slice (H, W)
            num_slices: Number of slices in the sequence
            slice_spacing: Physical spacing between slices
            background_value: Background intensity value
            chunk_size: Number of Gaussians to process at once
        """
        super().__init__()
        
        self.image_size = image_size
        self.num_slices = num_slices
        self.slice_spacing = slice_spacing
        self.background_value = background_value
        self.chunk_size = chunk_size
        
        # Create fixed slice positions
        self.register_buffer('slice_positions', self._create_slice_positions())
        
        # Create coordinate grids for each slice
        self.register_buffer('slice_grids', self._create_slice_grids())
    
    def _create_slice_positions(self) -> torch.Tensor:
        """
        Create fixed z-positions for each slice.
        
        Returns:
            slice_positions: Z-coordinates [num_slices]
        """
        # Center slices around z=0
        total_depth = (self.num_slices - 1) * self.slice_spacing
        z_start = -total_depth / 2
        
        positions = torch.linspace(
            z_start,
            z_start + total_depth,
            self.num_slices
        )
        
        return positions
    
    def _create_slice_grids(self) -> torch.Tensor:
        """
        Create 2D coordinate grids for each slice.
        
        Returns:
            slice_grids: Coordinate grids [num_slices, H, W, 2] (x, y)
        """
        H, W = self.image_size
        
        # Create normalized coordinates in [-1, 1]
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Stack to [H, W, 2]
        grid = torch.stack([xx, yy], dim=-1)
        
        # Repeat for all slices
        slice_grids = grid.unsqueeze(0).expand(self.num_slices, -1, -1, -1)
        
        return slice_grids
    
    def forward(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render 2D slice sequence from 3D Gaussians.
        
        Args:
            means: Gaussian centers [N, 3] (x, y, z)
            scales: Gaussian scales [N, 3]
            rotations: Gaussian rotations as quaternions [N, 4]
            opacities: Gaussian opacities [N, 1]
            features: Gaussian features [N, 1] (intensity)
        
        Returns:
            rendered_slices: Rendered 2D slices [num_slices, H, W]
        """
        N = means.shape[0]
        H, W = self.image_size
        
        # Compute covariance matrices
        covariances = self._compute_covariance(scales, rotations)  # [N, 3, 3]
        
        # Render each slice
        rendered_slices = []
        for slice_idx in range(self.num_slices):
            slice_image = self._render_single_slice(
                slice_idx,
                means,
                covariances,
                opacities,
                features,
            )
            rendered_slices.append(slice_image)
        
        # Stack to [num_slices, H, W]
        rendered_slices = torch.stack(rendered_slices, dim=0)
        
        return rendered_slices
    
    def _render_single_slice(
        self,
        slice_idx: int,
        means: torch.Tensor,
        covariances: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render a single 2D slice.
        
        Args:
            slice_idx: Index of the slice to render
            means: Gaussian centers [N, 3]
            covariances: Covariance matrices [N, 3, 3]
            opacities: Gaussian opacities [N, 1]
            features: Gaussian features [N, 1]
        
        Returns:
            slice_image: Rendered slice [H, W]
        """
        H, W = self.image_size
        N = means.shape[0]
        
        # Get slice position and grid
        slice_z = self.slice_positions[slice_idx]
        grid_2d = self.slice_grids[slice_idx]  # [H, W, 2]
        
        # Filter Gaussians near this slice (for efficiency)
        z_distances = torch.abs(means[:, 2] - slice_z)
        max_distance = 3.0 * scales[:, 2].max()  # 3 sigma
        near_mask = z_distances < max_distance
        
        if near_mask.sum() == 0:
            # No Gaussians near this slice
            return torch.full(
                (H, W),
                self.background_value,
                device=means.device,
                dtype=means.dtype
            )
        
        # Get nearby Gaussians
        means_near = means[near_mask]
        covariances_near = covariances[near_mask]
        opacities_near = opacities[near_mask]
        features_near = features[near_mask]
        
        # Project 3D Gaussians to 2D slice
        slice_image = self._project_gaussians_to_slice(
            grid_2d,
            slice_z,
            means_near,
            covariances_near,
            opacities_near,
            features_near,
        )
        
        return slice_image
    
    def _project_gaussians_to_slice(
        self,
        grid_2d: torch.Tensor,
        slice_z: float,
        means: torch.Tensor,
        covariances: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D Gaussians onto a 2D slice.
        
        This implements the marginal distribution of a 3D Gaussian
        at a fixed z-coordinate.
        
        Args:
            grid_2d: 2D coordinate grid [H, W, 2]
            slice_z: Z-coordinate of the slice
            means: Gaussian centers [M, 3]
            covariances: Covariance matrices [M, 3, 3]
            opacities: Gaussian opacities [M, 1]
            features: Gaussian features [M, 1]
        
        Returns:
            slice_image: Rendered slice [H, W]
        """
        H, W = grid_2d.shape[:2]
        M = means.shape[0]
        
        # Flatten grid for easier processing
        grid_flat = grid_2d.reshape(-1, 2)  # [H*W, 2]
        
        # Initialize output
        slice_image = torch.zeros(
            H * W,
            device=means.device,
            dtype=means.dtype
        )
        accumulated_alpha = torch.zeros(
            H * W,
            device=means.device,
            dtype=means.dtype
        )
        
        # Process Gaussians in chunks
        for start_idx in range(0, M, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, M)
            
            chunk_means = means[start_idx:end_idx]
            chunk_covs = covariances[start_idx:end_idx]
            chunk_opacities = opacities[start_idx:end_idx]
            chunk_features = features[start_idx:end_idx]
            
            # Compute contribution of this chunk
            chunk_contribution, chunk_alpha = self._compute_chunk_contribution(
                grid_flat,
                slice_z,
                chunk_means,
                chunk_covs,
                chunk_opacities,
                chunk_features,
            )
            
            # Alpha compositing (front-to-back)
            weight = (1 - accumulated_alpha).unsqueeze(-1) * chunk_alpha
            slice_image += (weight * chunk_contribution).sum(dim=-1)
            accumulated_alpha += weight.sum(dim=-1)
        
        # Add background
        slice_image += (1 - accumulated_alpha) * self.background_value
        
        # Reshape to image
        slice_image = slice_image.reshape(H, W)
        
        return slice_image
    
    def _compute_chunk_contribution(
        self,
        grid_flat: torch.Tensor,
        slice_z: float,
        means: torch.Tensor,
        covariances: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contribution of a chunk of Gaussians.
        
        Args:
            grid_flat: Flattened 2D grid [H*W, 2]
            slice_z: Z-coordinate of slice
            means: Gaussian centers [K, 3]
            covariances: Covariance matrices [K, 3, 3]
            opacities: Gaussian opacities [K, 1]
            features: Gaussian features [K, 1]
        
        Returns:
            contribution: Color contribution [H*W, K]
            alpha: Alpha values [H*W, K]
        """
        K = means.shape[0]
        num_pixels = grid_flat.shape[0]
        
        # For each Gaussian, compute 2D marginal distribution
        # Given 3D Gaussian N(μ, Σ) and fixed z, the marginal is 2D Gaussian
        
        contributions = []
        alphas = []
        
        for k in range(K):
            mu = means[k]  # [3]
            sigma = covariances[k]  # [3, 3]
            
            # Compute 2D marginal at z = slice_z
            # μ_2D = μ_xy + Σ_xy,z * Σ_zz^-1 * (z - μ_z)
            # Σ_2D = Σ_xy - Σ_xy,z * Σ_zz^-1 * Σ_z,xy
            
            mu_xy = mu[:2]  # [2]
            mu_z = mu[2]
            
            sigma_xy = sigma[:2, :2]  # [2, 2]
            sigma_xy_z = sigma[:2, 2:3]  # [2, 1]
            sigma_z_xy = sigma[2:3, :2]  # [1, 2]
            sigma_zz = sigma[2:3, 2:3]  # [1, 1]
            
            # Conditional mean
            z_offset = slice_z - mu_z
            mu_2d = mu_xy + (sigma_xy_z / (sigma_zz + 1e-6)) * z_offset
            
            # Conditional covariance
            sigma_2d = sigma_xy - torch.mm(sigma_xy_z, sigma_z_xy) / (sigma_zz + 1e-6)
            
            # Add small regularization
            sigma_2d = sigma_2d + torch.eye(2, device=sigma_2d.device) * 1e-6
            
            # Compute 2D Gaussian values
            try:
                inv_sigma_2d = torch.inverse(sigma_2d)
            except:
                # Singular matrix, skip this Gaussian
                contributions.append(torch.zeros(num_pixels, device=grid_flat.device))
                alphas.append(torch.zeros(num_pixels, device=grid_flat.device))
                continue
            
            # Compute Mahalanobis distance
            offset = grid_flat - mu_2d  # [H*W, 2]
            offset_transformed = torch.mm(offset, inv_sigma_2d)  # [H*W, 2]
            mahalanobis_dist = (offset * offset_transformed).sum(dim=1)  # [H*W]
            
            # Compute Gaussian value
            gaussian_value = torch.exp(-0.5 * mahalanobis_dist)  # [H*W]
            
            # Apply opacity
            alpha_k = opacities[k] * gaussian_value  # [H*W]
            
            # Contribution
            contribution_k = features[k] * gaussian_value  # [H*W]
            
            contributions.append(contribution_k)
            alphas.append(alpha_k)
        
        # Stack
        contribution = torch.stack(contributions, dim=-1)  # [H*W, K]
        alpha = torch.stack(alphas, dim=-1)  # [H*W, K]
        
        return contribution, alpha
    
    def _compute_covariance(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 3D covariance matrices from scales and rotations.
        
        Args:
            scales: Gaussian scales [N, 3]
            rotations: Quaternions [N, 4] (w, x, y, z)
        
        Returns:
            covariances: Covariance matrices [N, 3, 3]
        """
        N = scales.shape[0]
        
        # Build rotation matrices from quaternions
        R = self._quaternion_to_rotation_matrix(rotations)  # [N, 3, 3]
        
        # Build scale matrices
        S = torch.diag_embed(scales)  # [N, 3, 3]
        
        # Covariance: Σ = R S S^T R^T
        RS = torch.bmm(R, S)  # [N, 3, 3]
        covariances = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]
        
        return covariances
    
    def _quaternion_to_rotation_matrix(
        self,
        quaternions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices.
        
        Args:
            quaternions: Quaternions [N, 4] (w, x, y, z)
        
        Returns:
            rotation_matrices: Rotation matrices [N, 3, 3]
        """
        # Normalize quaternions
        q = F.normalize(quaternions, p=2, dim=1)
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Compute rotation matrix elements
        R = torch.stack([
            torch.stack([
                1 - 2*(y*y + z*z),
                2*(x*y - w*z),
                2*(x*z + w*y)
            ], dim=1),
            torch.stack([
                2*(x*y + w*z),
                1 - 2*(x*x + z*z),
                2*(y*z - w*x)
            ], dim=1),
            torch.stack([
                2*(x*z - w*y),
                2*(y*z + w*x),
                1 - 2*(x*x + y*y)
            ], dim=1),
        ], dim=1)
        
        return R
    
    def render_with_time(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        deformation_net: nn.Module,
    ) -> torch.Tensor:
        """
        Render slice sequence over time with deformation.
        
        Args:
            means: Initial Gaussian centers [N, 3]
            scales: Gaussian scales [N, 3]
            rotations: Gaussian rotations [N, 4]
            opacities: Gaussian opacities [N, 1]
            features: Gaussian features [N, 1]
            timestamps: Time points [T]
            deformation_net: Deformation network
        
        Returns:
            rendered_sequence: Rendered slices [T, num_slices, H, W]
        """
        T = timestamps.shape[0]
        
        rendered_sequence = []
        for t in range(T):
            # Apply deformation at this time
            deformed_means, deformed_opacities = deformation_net.apply_deformation(
                means,
                opacities,
                timestamps[t],
            )
            
            # Render slices at this time
            slices_t = self.forward(
                deformed_means,
                scales,
                rotations,
                deformed_opacities,
                features,
            )
            
            rendered_sequence.append(slices_t)
        
        # Stack to [T, num_slices, H, W]
        rendered_sequence = torch.stack(rendered_sequence, dim=0)
        
        return rendered_sequence
