"""
Control nodes for motion field modeling.

This module implements the control-nodes-based deformation mechanism
as described in the Dyna3DGR paper.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ControlNodes(nn.Module):
    """
    Control nodes for sparse motion field representation.
    
    Uses sparse control points with RBF (Radial Basis Function) interpolation
    to model dense deformation field efficiently.
    
    Each control node has:
    - Position Ci ∈ R³: Learnable position in canonical space
    - Radius oi ∈ R+: Learnable radius of RBF kernel
    
    The dense deformation field is obtained through Linear Blend Skinning
    with RBF weights.
    """
    
    def __init__(
        self,
        num_nodes: int,
        init_positions: torch.Tensor,  # [N, 3]
        init_radius: float = 0.1,
        min_radius: float = 0.01,
        max_radius: float = 1.0,
    ):
        """
        Initialize control nodes.
        
        Args:
            num_nodes: Number of control nodes
            init_positions: Initial positions [N, 3]
            init_radius: Initial radius value
            min_radius: Minimum allowed radius
            max_radius: Maximum allowed radius
        """
        super().__init__()
        
        assert init_positions.shape[0] == num_nodes, \
            f"init_positions must have {num_nodes} nodes"
        assert init_positions.shape[1] == 3, \
            "init_positions must be [N, 3]"
        
        self.num_nodes = num_nodes
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Control node positions Ci
        self.positions = nn.Parameter(init_positions.clone())
        
        # RBF kernel radii oi (stored in log space for stability)
        self._log_radii = nn.Parameter(
            torch.ones(num_nodes, 1) * torch.log(torch.tensor(init_radius))
        )
    
    @property
    def radii(self) -> torch.Tensor:
        """
        Get radii with activation to ensure positive values.
        
        Returns:
            radii: [N, 1] positive radii
        """
        # Exponential activation with clamping
        radii = torch.exp(self._log_radii)
        radii = torch.clamp(radii, self.min_radius, self.max_radius)
        return radii
    
    def compute_rbf_weights(
        self,
        query_points: torch.Tensor,  # [M, 3]
        k_nearest_indices: torch.Tensor,  # [M, k]
    ) -> torch.Tensor:
        """
        Compute RBF weights for Linear Blend Skinning.
        
        Uses Gaussian RBF:
            ŵij = exp(-dij² / (2 * oj²))
            wj = ŵij / Σ ŵij
        
        Args:
            query_points: Query point positions [M, 3]
            k_nearest_indices: Indices of k nearest control nodes [M, k]
        
        Returns:
            weights: [M, k] normalized RBF weights
        """
        M, k = k_nearest_indices.shape
        
        # Get nearest control node positions and radii
        nearest_positions = self.positions[k_nearest_indices]  # [M, k, 3]
        nearest_radii = self.radii[k_nearest_indices]  # [M, k, 1]
        
        # Compute squared distances
        diffs = query_points.unsqueeze(1) - nearest_positions  # [M, k, 3]
        distances_sq = torch.sum(diffs ** 2, dim=-1, keepdim=True)  # [M, k, 1]
        
        # Compute RBF weights: exp(-d²/(2*o²))
        raw_weights = torch.exp(
            -distances_sq / (2 * nearest_radii ** 2 + 1e-8)
        )  # [M, k, 1]
        
        # Normalize weights
        weights_sum = raw_weights.sum(dim=1, keepdim=True) + 1e-8
        weights = raw_weights / weights_sum  # [M, k, 1]
        
        return weights.squeeze(-1)  # [M, k]
    
    def blend_transformations(
        self,
        control_transformations: Tuple[torch.Tensor, torch.Tensor],
        query_points: torch.Tensor,  # [M, 3]
        k_nearest_indices: torch.Tensor,  # [M, k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blend control node transformations to query points using Linear Blend Skinning.
        
        Formula:
            [Δxyz_i^t | α_i^t] = Σ_{j=1}^k wj · [Δxyz_j^t | α_j^t]
        
        Args:
            control_transformations: Tuple of (delta_xyz, alpha) for control nodes
                - delta_xyz: [N, 3] position offsets
                - alpha: [N, 3] scale factors
            query_points: Query point positions [M, 3]
            k_nearest_indices: Indices of k nearest control nodes [M, k]
        
        Returns:
            delta_xyz: [M, 3] blended position offsets
            alpha: [M, 3] blended scale factors
        """
        control_delta_xyz, control_alpha = control_transformations
        M, k = k_nearest_indices.shape
        
        # Compute RBF weights
        weights = self.compute_rbf_weights(
            query_points=query_points,
            k_nearest_indices=k_nearest_indices,
        )  # [M, k]
        
        # Get transformations of nearest control nodes
        nearest_delta_xyz = control_delta_xyz[k_nearest_indices]  # [M, k, 3]
        nearest_alpha = control_alpha[k_nearest_indices]  # [M, k, 3]
        
        # Weighted sum (Linear Blend Skinning)
        delta_xyz = torch.sum(
            weights.unsqueeze(-1) * nearest_delta_xyz,
            dim=1,
        )  # [M, 3]
        
        alpha = torch.sum(
            weights.unsqueeze(-1) * nearest_alpha,
            dim=1,
        )  # [M, 3]
        
        return delta_xyz, alpha
    
    def forward(
        self,
        query_points: torch.Tensor,  # [M, 3]
        k_nearest_indices: torch.Tensor,  # [M, k]
    ) -> torch.Tensor:
        """
        Compute RBF weights for query points.
        
        Args:
            query_points: Query point positions [M, 3]
            k_nearest_indices: Indices of k nearest control nodes [M, k]
        
        Returns:
            weights: [M, k] RBF weights
        """
        return self.compute_rbf_weights(query_points, k_nearest_indices)
    
    def get_state_dict_with_metadata(self) -> dict:
        """
        Get state dict with metadata for checkpointing.
        
        Returns:
            state_dict: Dictionary with parameters and metadata
        """
        return {
            'positions': self.positions.data,
            'log_radii': self._log_radii.data,
            'num_nodes': self.num_nodes,
            'min_radius': self.min_radius,
            'max_radius': self.max_radius,
        }
    
    @classmethod
    def from_state_dict_with_metadata(cls, state_dict: dict) -> 'ControlNodes':
        """
        Create ControlNodes from state dict with metadata.
        
        Args:
            state_dict: Dictionary with parameters and metadata
        
        Returns:
            control_nodes: Reconstructed ControlNodes instance
        """
        control_nodes = cls(
            num_nodes=state_dict['num_nodes'],
            init_positions=state_dict['positions'],
            init_radius=1.0,  # Will be overwritten
            min_radius=state_dict['min_radius'],
            max_radius=state_dict['max_radius'],
        )
        
        control_nodes.positions.data = state_dict['positions']
        control_nodes._log_radii.data = state_dict['log_radii']
        
        return control_nodes
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ControlNodes(\n"
            f"  num_nodes={self.num_nodes},\n"
            f"  positions_shape={tuple(self.positions.shape)},\n"
            f"  radii_range=({self.radii.min().item():.4f}, {self.radii.max().item():.4f}),\n"
            f"  min_radius={self.min_radius},\n"
            f"  max_radius={self.max_radius}\n"
            f")"
        )


def initialize_control_nodes_from_gaussians(
    gaussian_positions: torch.Tensor,  # [N, 3]
    num_control_nodes: int = None,
    init_radius: float = 0.1,
) -> ControlNodes:
    """
    Initialize control nodes from Gaussian positions.
    
    As described in the paper:
    "Control nodes are initialized from these Gaussian positions
    (with the maximum number equal to the initial number of Gaussians)"
    
    Args:
        gaussian_positions: Gaussian positions [N, 3]
        num_control_nodes: Number of control nodes (default: same as Gaussians)
        init_radius: Initial radius value
    
    Returns:
        control_nodes: Initialized ControlNodes
    """
    N = gaussian_positions.shape[0]
    
    if num_control_nodes is None:
        num_control_nodes = N
    
    if num_control_nodes < N:
        # Subsample Gaussians
        indices = torch.randperm(N)[:num_control_nodes]
        init_positions = gaussian_positions[indices]
    else:
        # Use all Gaussians
        init_positions = gaussian_positions
    
    control_nodes = ControlNodes(
        num_nodes=num_control_nodes,
        init_positions=init_positions,
        init_radius=init_radius,
    )
    
    return control_nodes
