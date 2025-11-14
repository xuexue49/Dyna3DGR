"""
Gaussian Splatting Renderer.

This module implements differentiable 3D Gaussian Splatting rendering
for volumetric medical images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class GaussianRenderer(nn.Module):
    """
    Differentiable 3D Gaussian Splatting renderer.
    
    Renders 3D Gaussians to 2D/3D images using splatting and alpha compositing.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (256, 256, 10),
        background_color: float = 0.0,
        use_sh: bool = False,
    ):
        """
        Initialize Gaussian renderer.
        
        Args:
            image_size: Output image size (H, W, D)
            background_color: Background color value
            use_sh: Whether to use spherical harmonics for appearance
        """
        super().__init__()
        
        self.image_size = image_size
        self.background_color = background_color
        self.use_sh = use_sh
        
        # Create coordinate grids
        self.register_buffer('coord_grid', self._create_coordinate_grid())
    
    def _create_coordinate_grid(self) -> torch.Tensor:
        """
        Create 3D coordinate grid for the output volume.
        
        Returns:
            coord_grid: Coordinate grid [H, W, D, 3]
        """
        H, W, D = self.image_size
        
        # Create normalized coordinates in [-1, 1]
        z = torch.linspace(-1, 1, H)
        y = torch.linspace(-1, 1, W)
        x = torch.linspace(-1, 1, D)
        
        # Create meshgrid
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        
        # Stack to [H, W, D, 3]
        coord_grid = torch.stack([xx, yy, zz], dim=-1)
        
        return coord_grid
    
    def forward(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
        camera_transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render Gaussians to a 3D volume.
        
        Args:
            means: Gaussian centers [N, 3]
            scales: Gaussian scales [N, 3]
            rotations: Gaussian rotations as quaternions [N, 4]
            opacities: Gaussian opacities [N, 1]
            features: Gaussian features [N, F]
            camera_transform: Optional camera transformation [4, 4]
        
        Returns:
            rendered_image: Rendered volume [H, W, D, F]
        """
        N = means.shape[0]
        H, W, D = self.image_size
        F = features.shape[1]
        
        # Apply camera transform if provided
        if camera_transform is not None:
            means = self._apply_transform(means, camera_transform)
        
        # Compute covariance matrices
        covariances = self._compute_covariance(scales, rotations)  # [N, 3, 3]
        
        # Get coordinate grid
        coords = self.coord_grid  # [H, W, D, 3]
        
        # Compute Gaussian values at each voxel
        # This is the core splatting operation
        rendered_image = self._splat_gaussians(
            coords, means, covariances, opacities, features
        )
        
        return rendered_image
    
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
    
    def _splat_gaussians(
        self,
        coords: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Splat Gaussians onto the coordinate grid.
        
        Args:
            coords: Coordinate grid [H, W, D, 3]
            means: Gaussian centers [N, 3]
            covariances: Covariance matrices [N, 3, 3]
            opacities: Gaussian opacities [N, 1]
            features: Gaussian features [N, F]
        
        Returns:
            rendered_image: Rendered volume [H, W, D, F]
        """
        H, W, D = coords.shape[:3]
        N = means.shape[0]
        F = features.shape[1]
        
        # Reshape coords for broadcasting
        coords_flat = coords.reshape(-1, 3)  # [H*W*D, 3]
        
        # Compute inverse covariances
        try:
            inv_covariances = torch.inverse(covariances)  # [N, 3, 3]
        except:
            # Add small regularization if singular
            reg = torch.eye(3, device=covariances.device) * 1e-6
            inv_covariances = torch.inverse(covariances + reg.unsqueeze(0))
        
        # Initialize output
        rendered_image = torch.zeros(
            H * W * D, F,
            device=coords.device,
            dtype=coords.dtype
        )
        
        # Initialize accumulated alpha
        accumulated_alpha = torch.zeros(
            H * W * D, 1,
            device=coords.device,
            dtype=coords.dtype
        )
        
        # Sort Gaussians by depth (optional, for better blending)
        # For now, we process in order
        
        # Process each Gaussian
        for i in range(N):
            # Compute offset from Gaussian center
            offset = coords_flat - means[i:i+1]  # [H*W*D, 3]
            
            # Compute Mahalanobis distance
            # d^2 = (x - μ)^T Σ^-1 (x - μ)
            inv_cov = inv_covariances[i]  # [3, 3]
            offset_transformed = torch.mm(offset, inv_cov)  # [H*W*D, 3]
            mahalanobis_dist = (offset * offset_transformed).sum(dim=1, keepdim=True)  # [H*W*D, 1]
            
            # Compute Gaussian value
            # G(x) = exp(-0.5 * d^2)
            gaussian_value = torch.exp(-0.5 * mahalanobis_dist)  # [H*W*D, 1]
            
            # Apply opacity
            alpha = opacities[i] * gaussian_value  # [H*W*D, 1]
            
            # Alpha compositing (front-to-back)
            # C = C + (1 - A) * α * c
            # A = A + (1 - A) * α
            weight = (1 - accumulated_alpha) * alpha  # [H*W*D, 1]
            rendered_image += weight * features[i:i+1]  # [H*W*D, F]
            accumulated_alpha += weight
            
            # Early stopping if fully opaque
            # (optional optimization)
        
        # Add background
        background = torch.ones_like(rendered_image) * self.background_color
        rendered_image += (1 - accumulated_alpha) * background
        
        # Reshape to volume
        rendered_image = rendered_image.reshape(H, W, D, F)
        
        return rendered_image
    
    def _apply_transform(
        self,
        points: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply 4x4 transformation matrix to 3D points.
        
        Args:
            points: 3D points [N, 3]
            transform: Transformation matrix [4, 4]
        
        Returns:
            transformed_points: Transformed points [N, 3]
        """
        N = points.shape[0]
        
        # Convert to homogeneous coordinates
        points_homo = torch.cat([
            points,
            torch.ones(N, 1, device=points.device)
        ], dim=1)  # [N, 4]
        
        # Apply transformation
        transformed_homo = torch.mm(points_homo, transform.T)  # [N, 4]
        
        # Convert back to 3D
        transformed_points = transformed_homo[:, :3] / transformed_homo[:, 3:4]
        
        return transformed_points


class EfficientGaussianRenderer(nn.Module):
    """
    Memory-efficient Gaussian renderer using chunked processing.
    
    This version processes Gaussians in chunks to reduce memory usage.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (256, 256, 10),
        background_color: float = 0.0,
        chunk_size: int = 1000,
        distance_threshold: float = 3.0,
    ):
        """
        Initialize efficient Gaussian renderer.
        
        Args:
            image_size: Output image size (H, W, D)
            background_color: Background color value
            chunk_size: Number of Gaussians to process at once
            distance_threshold: Distance threshold for culling (in standard deviations)
        """
        super().__init__()
        
        self.image_size = image_size
        self.background_color = background_color
        self.chunk_size = chunk_size
        self.distance_threshold = distance_threshold
        
        # Create coordinate grid
        self.register_buffer('coord_grid', self._create_coordinate_grid())
    
    def _create_coordinate_grid(self) -> torch.Tensor:
        """Create 3D coordinate grid."""
        H, W, D = self.image_size
        
        z = torch.linspace(-1, 1, H)
        y = torch.linspace(-1, 1, W)
        x = torch.linspace(-1, 1, D)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coord_grid = torch.stack([xx, yy, zz], dim=-1)
        
        return coord_grid
    
    def forward(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render Gaussians efficiently.
        
        Args:
            means: Gaussian centers [N, 3]
            scales: Gaussian scales [N, 3]
            rotations: Gaussian rotations as quaternions [N, 4]
            opacities: Gaussian opacities [N, 1]
            features: Gaussian features [N, F]
        
        Returns:
            rendered_image: Rendered volume [H, W, D, F]
        """
        N = means.shape[0]
        H, W, D = self.image_size
        F = features.shape[1]
        
        # Initialize output
        rendered_image = torch.zeros(
            H, W, D, F,
            device=means.device,
            dtype=means.dtype
        )
        
        accumulated_alpha = torch.zeros(
            H, W, D, 1,
            device=means.device,
            dtype=means.dtype
        )
        
        # Process Gaussians in chunks
        for start_idx in range(0, N, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N)
            
            # Get chunk
            chunk_means = means[start_idx:end_idx]
            chunk_scales = scales[start_idx:end_idx]
            chunk_rotations = rotations[start_idx:end_idx]
            chunk_opacities = opacities[start_idx:end_idx]
            chunk_features = features[start_idx:end_idx]
            
            # Render chunk
            chunk_contribution, chunk_alpha = self._render_chunk(
                chunk_means,
                chunk_scales,
                chunk_rotations,
                chunk_opacities,
                chunk_features,
            )
            
            # Composite
            weight = (1 - accumulated_alpha) * chunk_alpha
            rendered_image += weight * chunk_contribution
            accumulated_alpha += weight
        
        # Add background
        background = torch.ones_like(rendered_image) * self.background_color
        rendered_image += (1 - accumulated_alpha) * background
        
        return rendered_image
    
    def _render_chunk(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a chunk of Gaussians.
        
        Returns:
            contribution: Color contribution [H, W, D, F]
            alpha: Alpha values [H, W, D, 1]
        """
        H, W, D = self.image_size
        chunk_size = means.shape[0]
        F = features.shape[1]
        
        # Compute covariances
        covariances = self._compute_covariance(scales, rotations)
        inv_covariances = torch.inverse(covariances + torch.eye(3, device=covariances.device).unsqueeze(0) * 1e-6)
        
        # Get coordinates
        coords = self.coord_grid  # [H, W, D, 3]
        
        # Initialize outputs
        contribution = torch.zeros(H, W, D, F, device=means.device)
        alpha = torch.zeros(H, W, D, 1, device=means.device)
        
        # Process each Gaussian in chunk
        for i in range(chunk_size):
            # Compute offset
            offset = coords - means[i]  # [H, W, D, 3]
            
            # Compute Mahalanobis distance
            offset_flat = offset.reshape(-1, 3)  # [H*W*D, 3]
            inv_cov = inv_covariances[i]
            offset_transformed = torch.mm(offset_flat, inv_cov)
            mahalanobis_dist = (offset_flat * offset_transformed).sum(dim=1)
            mahalanobis_dist = mahalanobis_dist.reshape(H, W, D)
            
            # Cull distant Gaussians
            mask = mahalanobis_dist < (self.distance_threshold ** 2)
            
            # Compute Gaussian value
            gaussian_value = torch.exp(-0.5 * mahalanobis_dist)
            gaussian_value = gaussian_value * mask.float()
            
            # Apply opacity
            alpha_i = opacities[i] * gaussian_value.unsqueeze(-1)  # [H, W, D, 1]
            
            # Accumulate
            contribution += alpha_i * features[i]
            alpha += alpha_i
        
        return contribution, alpha
    
    def _compute_covariance(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute covariance matrices."""
        # Normalize quaternions
        q = F.normalize(rotations, p=2, dim=1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Build rotation matrices
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=1),
        ], dim=1)
        
        # Build scale matrices
        S = torch.diag_embed(scales)
        
        # Covariance
        RS = torch.bmm(R, S)
        covariances = torch.bmm(RS, RS.transpose(1, 2))
        
        return covariances
