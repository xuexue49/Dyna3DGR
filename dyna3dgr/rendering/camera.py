"""
Camera system for Gaussian Splatting rendering.

This module implements camera models and transformations for
rendering 3D Gaussians from different viewpoints.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class Camera(nn.Module):
    """
    Camera model for 3D rendering.
    
    Supports perspective and orthographic projections.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (256, 256, 10),
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
        projection_type: str = 'orthographic',
    ):
        """
        Initialize camera.
        
        Args:
            image_size: Output image size (H, W, D)
            fov: Field of view in degrees (for perspective)
            near: Near clipping plane
            far: Far clipping plane
            projection_type: 'perspective' or 'orthographic'
        """
        super().__init__()
        
        self.image_size = image_size
        self.fov = fov
        self.near = near
        self.far = far
        self.projection_type = projection_type
        
        # Camera parameters (learnable if needed)
        self.position = nn.Parameter(
            torch.tensor([0.0, 0.0, 2.0]),
            requires_grad=False
        )
        self.look_at = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0]),
            requires_grad=False
        )
        self.up = nn.Parameter(
            torch.tensor([0.0, 1.0, 0.0]),
            requires_grad=False
        )
    
    def get_view_matrix(self) -> torch.Tensor:
        """
        Compute view matrix (world to camera transform).
        
        Returns:
            view_matrix: View matrix [4, 4]
        """
        # Camera coordinate system
        forward = F.normalize(self.look_at - self.position, dim=0)
        right = F.normalize(torch.cross(forward, self.up), dim=0)
        up = torch.cross(right, forward)
        
        # Build view matrix
        view_matrix = torch.eye(4, device=self.position.device)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -forward
        view_matrix[:3, 3] = -torch.stack([
            torch.dot(right, self.position),
            torch.dot(up, self.position),
            torch.dot(forward, self.position),
        ])
        
        return view_matrix
    
    def get_projection_matrix(self) -> torch.Tensor:
        """
        Compute projection matrix.
        
        Returns:
            projection_matrix: Projection matrix [4, 4]
        """
        if self.projection_type == 'perspective':
            return self._get_perspective_matrix()
        else:
            return self._get_orthographic_matrix()
    
    def _get_perspective_matrix(self) -> torch.Tensor:
        """Compute perspective projection matrix."""
        H, W, D = self.image_size
        aspect = W / H
        
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        proj_matrix = torch.zeros(4, 4, device=self.position.device)
        proj_matrix[0, 0] = f / aspect
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj_matrix[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj_matrix[3, 2] = -1.0
        
        return proj_matrix
    
    def _get_orthographic_matrix(self) -> torch.Tensor:
        """Compute orthographic projection matrix."""
        # For medical images, orthographic projection is more appropriate
        # Maps [-1, 1]^3 to [-1, 1]^3 (identity for now)
        return torch.eye(4, device=self.position.device)
    
    def get_transform_matrix(self) -> torch.Tensor:
        """
        Get combined view-projection matrix.
        
        Returns:
            transform: Combined transformation [4, 4]
        """
        view = self.get_view_matrix()
        proj = self.get_projection_matrix()
        return torch.mm(proj, view)
    
    def set_pose(
        self,
        position: torch.Tensor,
        look_at: torch.Tensor,
        up: Optional[torch.Tensor] = None,
    ):
        """
        Set camera pose.
        
        Args:
            position: Camera position [3]
            look_at: Look-at point [3]
            up: Up vector [3]
        """
        self.position.data = position
        self.look_at.data = look_at
        if up is not None:
            self.up.data = up


class MultiViewCamera(nn.Module):
    """
    Multi-view camera system for rendering from multiple viewpoints.
    
    Useful for data augmentation and multi-view consistency.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (256, 256, 10),
        num_views: int = 1,
    ):
        """
        Initialize multi-view camera.
        
        Args:
            image_size: Output image size
            num_views: Number of views
        """
        super().__init__()
        
        self.image_size = image_size
        self.num_views = num_views
        
        # Create cameras for each view
        self.cameras = nn.ModuleList([
            Camera(image_size=image_size)
            for _ in range(num_views)
        ])
    
    def get_transform_matrices(self) -> torch.Tensor:
        """
        Get transformation matrices for all views.
        
        Returns:
            transforms: Transformation matrices [num_views, 4, 4]
        """
        transforms = torch.stack([
            camera.get_transform_matrix()
            for camera in self.cameras
        ])
        return transforms
    
    def set_circular_poses(
        self,
        radius: float = 2.0,
        height: float = 0.0,
        look_at: Optional[torch.Tensor] = None,
    ):
        """
        Set cameras in a circular arrangement around the object.
        
        Args:
            radius: Distance from center
            height: Height of cameras
            look_at: Center point to look at
        """
        if look_at is None:
            look_at = torch.zeros(3)
        
        for i, camera in enumerate(self.cameras):
            angle = 2 * math.pi * i / self.num_views
            position = torch.tensor([
                radius * math.cos(angle),
                height,
                radius * math.sin(angle),
            ])
            camera.set_pose(position, look_at)


class VolumetricCamera(nn.Module):
    """
    Camera specifically designed for volumetric medical images.
    
    Uses orthographic projection and handles 3D volumes directly.
    """
    
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (256, 256, 10),
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Initialize volumetric camera.
        
        Args:
            volume_size: Size of the volume (H, W, D)
            voxel_spacing: Spacing between voxels (sx, sy, sz)
        """
        super().__init__()
        
        self.volume_size = volume_size
        self.voxel_spacing = voxel_spacing
        
        # Compute volume bounds in world space
        self.bounds = self._compute_bounds()
    
    def _compute_bounds(self) -> torch.Tensor:
        """
        Compute volume bounds in world coordinates.
        
        Returns:
            bounds: Bounds [2, 3] (min, max)
        """
        H, W, D = self.volume_size
        sx, sy, sz = self.voxel_spacing
        
        # Compute physical size
        size_x = W * sx
        size_y = H * sy
        size_z = D * sz
        
        # Center the volume at origin
        bounds = torch.tensor([
            [-size_x/2, -size_y/2, -size_z/2],
            [size_x/2, size_y/2, size_z/2],
        ])
        
        return bounds
    
    def world_to_voxel(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to voxel indices.
        
        Args:
            points: Points in world space [N, 3]
        
        Returns:
            voxel_indices: Voxel indices [N, 3]
        """
        H, W, D = self.volume_size
        sx, sy, sz = self.voxel_spacing
        
        # Normalize to [0, 1]
        normalized = (points - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        
        # Scale to voxel indices
        voxel_indices = normalized * torch.tensor([W, H, D], device=points.device)
        
        return voxel_indices
    
    def voxel_to_world(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert voxel indices to world coordinates.
        
        Args:
            voxel_indices: Voxel indices [N, 3]
        
        Returns:
            points: Points in world space [N, 3]
        """
        H, W, D = self.volume_size
        
        # Normalize to [0, 1]
        normalized = voxel_indices / torch.tensor([W, H, D], device=voxel_indices.device)
        
        # Scale to world coordinates
        points = self.bounds[0] + normalized * (self.bounds[1] - self.bounds[0])
        
        return points
    
    def get_slice_transform(
        self,
        slice_axis: int = 2,
        slice_index: int = 0,
    ) -> torch.Tensor:
        """
        Get transformation for extracting a 2D slice.
        
        Args:
            slice_axis: Axis to slice along (0=x, 1=y, 2=z)
            slice_index: Index of the slice
        
        Returns:
            transform: Transformation matrix [4, 4]
        """
        transform = torch.eye(4)
        
        # Set the appropriate translation based on slice
        H, W, D = self.volume_size
        sizes = torch.tensor([W, H, D])
        
        # Normalize slice index to [-1, 1]
        normalized_index = (slice_index / sizes[slice_axis]) * 2 - 1
        
        # Set translation
        transform[slice_axis, 3] = normalized_index
        
        return transform


# Helper function
import torch.nn.functional as F


def normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize a vector."""
    return F.normalize(v, p=2, dim=dim)
