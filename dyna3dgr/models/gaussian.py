"""
3D Gaussian Representation

This module implements the explicit 3D Gaussian representation for cardiac structures.
Each Gaussian is parameterized by position, scale, rotation, opacity, and features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion: [N, 4] quaternions (w, x, y, z)
    
    Returns:
        rotation_matrix: [N, 3, 3] rotation matrices
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # Compute rotation matrix elements
    R = torch.zeros(quaternion.shape[0], 3, 3, device=quaternion.device, dtype=quaternion.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    
    return R


def build_covariance_matrix(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Build covariance matrix from scale and rotation.
    
    Args:
        scale: [N, 3] scaling factors
        rotation: [N, 4] quaternions
    
    Returns:
        covariance: [N, 3, 3] covariance matrices
    """
    # Get rotation matrix
    R = quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]
    
    # Build scaling matrix
    S = torch.diag_embed(scale)  # [N, 3, 3]
    
    # Covariance = R * S * S^T * R^T
    RS = torch.bmm(R, S)  # [N, 3, 3]
    covariance = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]
    
    return covariance


class Gaussian3D(nn.Module):
    """
    3D Gaussian representation for cardiac structures.
    
    Each Gaussian is parameterized by:
    - xyz: position in 3D space
    - scale: scaling factors for each axis
    - rotation: rotation as quaternion
    - opacity: opacity/alpha value
    - features: appearance features (e.g., color, intensity)
    """
    
    def __init__(
        self,
        num_points: int,
        feature_dim: int = 1,
        init_scale: float = 0.01,
        init_opacity: float = 0.5,
    ):
        """
        Initialize 3D Gaussians.
        
        Args:
            num_points: Number of Gaussians
            feature_dim: Dimension of appearance features
            init_scale: Initial scale value
            init_opacity: Initial opacity value
        """
        super().__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Position [N, 3]
        self._xyz = nn.Parameter(torch.zeros(num_points, 3))
        
        # Scale [N, 3]
        self._scale = nn.Parameter(torch.ones(num_points, 3) * init_scale)
        
        # Rotation as quaternion [N, 4] (w, x, y, z)
        self._rotation = nn.Parameter(torch.zeros(num_points, 4))
        self._rotation.data[:, 0] = 1.0  # Initialize to identity rotation
        
        # Opacity [N, 1]
        self._opacity = nn.Parameter(torch.ones(num_points, 1) * init_opacity)
        
        # Features [N, F]
        self._features = nn.Parameter(torch.zeros(num_points, feature_dim))
    
    @property
    def xyz(self) -> torch.Tensor:
        """Get positions."""
        return self._xyz
    
    @property
    def scale(self) -> torch.Tensor:
        """Get scales (with activation)."""
        return torch.exp(self._scale)  # Ensure positive scales
    
    @property
    def rotation(self) -> torch.Tensor:
        """Get rotations (normalized quaternions)."""
        return self._rotation / torch.norm(self._rotation, dim=-1, keepdim=True)
    
    @property
    def opacity(self) -> torch.Tensor:
        """Get opacity (with activation)."""
        return torch.sigmoid(self._opacity)  # Ensure [0, 1]
    
    @property
    def features(self) -> torch.Tensor:
        """Get features."""
        return self._features
    
    def get_covariance(self) -> torch.Tensor:
        """
        Compute covariance matrices for all Gaussians.
        
        Returns:
            covariance: [N, 3, 3] covariance matrices
        """
        return build_covariance_matrix(self.scale, self.rotation)
    
    def compute_influence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian influence at given points.
        
        Args:
            points: [M, 3] query points
        
        Returns:
            influence: [M, N] influence values
        """
        # Get Gaussian parameters
        xyz = self.xyz  # [N, 3]
        covariance = self.get_covariance()  # [N, 3, 3]
        
        # Compute inverse covariance
        cov_inv = torch.inverse(covariance)  # [N, 3, 3]
        
        # Compute difference vectors [M, N, 3]
        diff = points.unsqueeze(1) - xyz.unsqueeze(0)  # [M, N, 3]
        
        # Compute Mahalanobis distance
        # d^2 = (x - mu)^T * Sigma^-1 * (x - mu)
        temp = torch.einsum('mni,nij->mnj', diff, cov_inv)  # [M, N, 3]
        mahal_dist = torch.sum(temp * diff, dim=-1)  # [M, N]
        
        # Compute Gaussian influence
        influence = torch.exp(-0.5 * mahal_dist)  # [M, N]
        
        return influence
    
    def densify(self, grad_threshold: float = 0.0002, max_points: int = 100000):
        """
        Densify Gaussians by splitting or cloning based on gradients.
        
        Args:
            grad_threshold: Gradient threshold for densification
            max_points: Maximum number of points
        """
        if self.num_points >= max_points:
            return
        
        # Get gradients
        if self._xyz.grad is None:
            return
        
        grad_norm = torch.norm(self._xyz.grad, dim=-1)
        
        # Find Gaussians to densify
        mask = grad_norm > grad_threshold
        
        if not mask.any():
            return
        
        # Clone selected Gaussians
        new_xyz = self._xyz[mask].detach().clone()
        new_scale = self._scale[mask].detach().clone()
        new_rotation = self._rotation[mask].detach().clone()
        new_opacity = self._opacity[mask].detach().clone()
        new_features = self._features[mask].detach().clone()
        
        # Add small perturbation
        new_xyz += torch.randn_like(new_xyz) * 0.01
        
        # Concatenate with existing Gaussians
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._scale = nn.Parameter(torch.cat([self._scale, new_scale], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._features = nn.Parameter(torch.cat([self._features, new_features], dim=0))
        
        self.num_points = self._xyz.shape[0]
    
    def prune(self, opacity_threshold: float = 0.01):
        """
        Prune Gaussians with low opacity.
        
        Args:
            opacity_threshold: Opacity threshold for pruning
        """
        # Find Gaussians to keep
        mask = self.opacity.squeeze() > opacity_threshold
        
        if mask.sum() == 0:
            return
        
        # Keep only selected Gaussians
        self._xyz = nn.Parameter(self._xyz[mask])
        self._scale = nn.Parameter(self._scale[mask])
        self._rotation = nn.Parameter(self._rotation[mask])
        self._opacity = nn.Parameter(self._opacity[mask])
        self._features = nn.Parameter(self._features[mask])
        
        self.num_points = self._xyz.shape[0]
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute volumetric values at given points.
        
        Args:
            points: [M, 3] query points
        
        Returns:
            values: [M, F] feature values
            weights: [M, N] Gaussian weights
        """
        # Compute influence
        influence = self.compute_influence(points)  # [M, N]
        
        # Weight by opacity
        weights = influence * self.opacity.T  # [M, N]
        
        # Compute weighted features
        values = torch.matmul(weights, self.features)  # [M, F]
        
        return values, weights


def initialize_gaussians_from_point_cloud(
    points: np.ndarray,
    num_gaussians: Optional[int] = None,
    feature_dim: int = 1,
    initial_features: Optional[torch.Tensor] = None,
) -> Gaussian3D:
    """
    Initialize Gaussians from a point cloud.
    
    Args:
        points: [N, 3] point cloud
        num_gaussians: Number of Gaussians (if None, use all points)
        feature_dim: Feature dimension
    
    Returns:
        gaussians: Initialized Gaussian3D object
    """
    if num_gaussians is None:
        num_gaussians = points.shape[0]
    elif num_gaussians < points.shape[0]:
        # Randomly sample points
        indices = np.random.choice(points.shape[0], num_gaussians, replace=False)
        points = points[indices]
    
    # Create Gaussians
    gaussians = Gaussian3D(num_gaussians, feature_dim)
    
    # Initialize positions
    if isinstance(points, torch.Tensor):
        gaussians._xyz.data = points.float()
        points_np = points.cpu().numpy()
    else:
        gaussians._xyz.data = torch.from_numpy(points).float()
        points_np = points
    
    # Estimate initial scale from nearest neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(points_np)
    distances, _ = tree.query(points_np, k=4)  # k=4 to exclude self
    avg_distances = distances[:, 1:].mean(axis=1)  # Exclude self
    
    init_scale = torch.from_numpy(avg_distances).float().unsqueeze(1).repeat(1, 3)
    gaussians._scale.data = torch.log(init_scale)
    
    # Initialize features if provided
    if initial_features is not None:
        if isinstance(initial_features, np.ndarray):
            initial_features = torch.from_numpy(initial_features).float()
        gaussians._features.data = initial_features
    else:
        # Default: initialize to small random values to avoid zero gradients
        gaussians._features.data = torch.randn(num_gaussians, feature_dim) * 0.1 + 0.5
    
    return gaussians
