"""
CUDA-accelerated Gaussian Splatting renderer using diff-gaussian-rasterization.

Based on MedGS implementation.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: diff-gaussian-rasterization not available. Install it for CUDA acceleration.")


class CUDAGaussianRenderer(nn.Module):
    """
    CUDA-accelerated Gaussian Splatting renderer.
    
    Uses diff-gaussian-rasterization for fast rendering with perspective projection.
    """
    
    def __init__(
        self,
        image_size: tuple,  # (H, W)
        fov_x: float = 0.8,  # Field of view in radians
        fov_y: Optional[float] = None,
        bg_color: tuple = (0.0, 0.0, 0.0),
        scale_modifier: float = 1.0,
    ):
        super().__init__()
        
        if not CUDA_AVAILABLE:
            raise RuntimeError("diff-gaussian-rasterization not installed. Run: bash scripts/install_cuda_rasterizer.sh")
        
        self.image_height, self.image_width = image_size
        self.fov_x = fov_x
        self.fov_y = fov_y if fov_y is not None else fov_x * (self.image_height / self.image_width)
        self.scale_modifier = scale_modifier
        
        self.register_buffer('bg_color', torch.tensor(bg_color, dtype=torch.float32))
    
    def create_camera(
        self,
        camera_position: torch.Tensor,  # [3]
        look_at: torch.Tensor,  # [3]
        up: torch.Tensor = None,  # [3]
    ):
        """
        Create camera matrices for rendering.
        
        Args:
            camera_position: Camera position in world space
            look_at: Point to look at
            up: Up vector (default: [0, 1, 0])
        
        Returns:
            world_view_transform: [4, 4]
            full_proj_transform: [4, 4]
            camera_center: [3]
        """
        if up is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=camera_position.device)
        
        # View matrix (world to camera)
        z_axis = torch.nn.functional.normalize(camera_position - look_at, dim=0)
        x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis), dim=0)
        y_axis = torch.cross(z_axis, x_axis)
        
        R = torch.stack([x_axis, y_axis, z_axis], dim=0)  # [3, 3]
        t = -R @ camera_position  # [3]
        
        world_view_transform = torch.eye(4, device=camera_position.device)
        world_view_transform[:3, :3] = R
        world_view_transform[:3, 3] = t
        
        # Projection matrix
        tanfovx = math.tan(self.fov_x * 0.5)
        tanfovy = math.tan(self.fov_y * 0.5)
        
        near = 0.01
        far = 100.0
        
        proj = torch.zeros(4, 4, device=camera_position.device)
        proj[0, 0] = 1.0 / tanfovx
        proj[1, 1] = 1.0 / tanfovy
        proj[2, 2] = far / (far - near)
        proj[2, 3] = -(far * near) / (far - near)
        proj[3, 2] = 1.0
        
        full_proj_transform = world_view_transform @ proj
        
        return world_view_transform, full_proj_transform, camera_position
    
    def forward(
        self,
        means: torch.Tensor,  # [N, 3]
        scales: torch.Tensor,  # [N, 3]
        rotations: torch.Tensor,  # [N, 4] quaternions
        opacities: torch.Tensor,  # [N, 1]
        features: torch.Tensor,  # [N, F]
        camera_position: torch.Tensor,  # [3]
        look_at: torch.Tensor,  # [3]
        up: Optional[torch.Tensor] = None,  # [3]
    ):
        """
        Render Gaussians from a camera viewpoint.
        
        Args:
            means: Gaussian centers
            scales: Gaussian scales
            rotations: Gaussian rotations (quaternions)
            opacities: Gaussian opacities
            features: Gaussian features (colors)
            camera_position: Camera position
            look_at: Point to look at
            up: Up vector
        
        Returns:
            rendered_image: [F, H, W]
        """
        # Create camera matrices
        world_view_transform, full_proj_transform, camera_center = self.create_camera(
            camera_position, look_at, up
        )
        
        # Create screenspace points for gradient
        screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device=means.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # Setup rasterization
        tanfovx = math.tan(self.fov_x * 0.5)
        tanfovy = math.tan(self.fov_y * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scale_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0,  # No spherical harmonics, use precomputed colors
            campos=camera_center,
            prefiltered=False,
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Ensure features are colors [N, 3]
        if features.shape[-1] == 1:
            # Grayscale -> RGB
            colors_precomp = features.repeat(1, 3)
        elif features.shape[-1] == 3:
            colors_precomp = features
        else:
            # Use first 3 channels
            colors_precomp = features[:, :3]
        
        # Clamp colors to [0, 1]
        colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)
        
        # Rasterize
        rendered_image, radii, _ = rasterizer(
            means3D=means,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        
        return rendered_image  # [3, H, W]


def create_orthographic_camera_for_slice(
    slice_z: float,
    image_size: tuple,  # (H, W)
    spacing: tuple = (1.0, 1.0, 1.0),  # (x, y, z)
):
    """
    Create camera parameters for orthographic rendering of a 2D slice.
    
    Args:
        slice_z: Z position of the slice
        image_size: (H, W)
        spacing: Voxel spacing
    
    Returns:
        camera_position, look_at, up, fov_x, fov_y
    """
    H, W = image_size
    sx, sy, sz = spacing
    
    # Camera looks down the Z axis
    camera_position = torch.tensor([W * sx / 2, H * sy / 2, slice_z + 10.0])
    look_at = torch.tensor([W * sx / 2, H * sy / 2, slice_z])
    up = torch.tensor([0.0, 1.0, 0.0])
    
    # FOV for orthographic-like projection
    # Small FOV approximates orthographic
    fov_x = 2 * math.atan(W * sx / (2 * 10.0))
    fov_y = 2 * math.atan(H * sy / (2 * 10.0))
    
    return camera_position, look_at, up, fov_x, fov_y
