"""
Models module for Dyna3DGR.

This module contains the core model components:
- Gaussian3D: 3D Gaussian representation
- DeformationNetwork: Neural deformation network
- ControlNodeMotionField: Control node-based motion field
"""

from .gaussian import (
    Gaussian3D,
    initialize_gaussians_from_point_cloud,
    quaternion_to_rotation_matrix,
    build_covariance_matrix,
)

from .deformation_network import (
    DeformationNetwork,
    ControlNodeMotionField,
    PositionalEncoder,
    MLP,
)

from .control_nodes import (
    ControlNodes,
    initialize_control_nodes_from_gaussians,
)

from .densification import (
    GaussianDensificationController,
)

__all__ = [
    "Gaussian3D",
    "initialize_gaussians_from_point_cloud",
    "quaternion_to_rotation_matrix",
    "build_covariance_matrix",
    "DeformationNetwork",
    "ControlNodeMotionField",
    "PositionalEncoder",
    "MLP",
    "ControlNodes",
    "initialize_control_nodes_from_gaussians",
    "GaussianDensificationController",
]
