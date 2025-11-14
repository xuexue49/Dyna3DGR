"""
Deformation Network

This module implements the implicit neural motion field for modeling
cardiac deformation over time.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PositionalEncoder(nn.Module):
    """
    Positional encoding for coordinates.
    
    Encodes input coordinates using sinusoidal functions:
    gamma(p) = (sin(2^0 * pi * p), cos(2^0 * pi * p), ..., 
                sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p))
    """
    
    def __init__(self, input_dim: int, num_frequencies: int = 10, include_input: bool = True):
        """
        Initialize positional encoder.
        
        Args:
            input_dim: Input dimension
            num_frequencies: Number of frequency bands (L)
            include_input: Whether to include original input
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Compute output dimension
        self.output_dim = input_dim * num_frequencies * 2
        if include_input:
            self.output_dim += input_dim
        
        # Frequency bands
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: [..., input_dim] input coordinates
        
        Returns:
            encoded: [..., output_dim] encoded coordinates
        """
        # Move frequency bands to same device as input
        freq_bands = self.freq_bands.to(x.device)
        
        # Compute encodings
        encodings = []
        
        if self.include_input:
            encodings.append(x)
        
        for freq in freq_bands:
            encodings.append(torch.sin(freq * np.pi * x))
            encodings.append(torch.cos(freq * np.pi * x))
        
        return torch.cat(encodings, dim=-1)


class MLP(nn.Module):
    """
    Multi-layer perceptron with skip connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: list = [4],
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            skip_connections: Layers to add skip connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim
            elif i in skip_connections:
                layer_input_dim = hidden_dim + input_dim
            else:
                layer_input_dim = hidden_dim
            
            if i == num_layers - 1:
                layer_output_dim = output_dim
            else:
                layer_output_dim = hidden_dim
            
            self.layers.append(nn.Linear(layer_input_dim, layer_output_dim))
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [..., input_dim] input
        
        Returns:
            output: [..., output_dim] output
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            # Skip connection
            if i in self.skip_connections and i > 0:
                h = torch.cat([h, x], dim=-1)
            
            h = layer(h)
            
            # Activation (except last layer)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        
        return h


class DeformationNetwork(nn.Module):
    """
    Deformation network for modeling cardiac motion.
    
    Takes spatial coordinates and time as input, outputs:
    - delta_xyz: position offset
    - delta_alpha: opacity change
    """
    
    def __init__(
        self,
        spatial_dim: int = 3,
        temporal_dim: int = 1,
        spatial_freq: int = 10,
        temporal_freq: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 8,
    ):
        """
        Initialize deformation network.
        
        Args:
            spatial_dim: Spatial coordinate dimension
            temporal_dim: Temporal coordinate dimension
            spatial_freq: Number of spatial frequency bands
            temporal_freq: Number of temporal frequency bands
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        # Positional encoders
        self.spatial_encoder = PositionalEncoder(spatial_dim, spatial_freq)
        self.temporal_encoder = PositionalEncoder(temporal_dim, temporal_freq)
        
        # Input dimension
        input_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim
        
        # MLP for deformation
        self.deformation_mlp = MLP(
            input_dim=input_dim,
            output_dim=spatial_dim + 1,  # delta_xyz + delta_alpha
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_connections=[num_layers // 2],
        )
    
    def forward(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        stop_gradient: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute deformation.
        
        Args:
            xyz: [N, 3] spatial coordinates
            t: [N, 1] or [1] temporal coordinate
            stop_gradient: Whether to stop gradient for spatial input
        
        Returns:
            delta_xyz: [N, 3] position offsets
            delta_alpha: [N, 1] opacity changes
        """
        # Stop gradient for spatial coordinates (as in paper)
        if stop_gradient:
            xyz_input = xyz.detach()
        else:
            xyz_input = xyz
        
        # Expand time if needed
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(xyz.shape[0], 1)
        
        # Encode inputs
        spatial_encoded = self.spatial_encoder(xyz_input)  # [N, D_s]
        temporal_encoded = self.temporal_encoder(t)  # [N, D_t]
        
        # Concatenate encodings
        encoded = torch.cat([spatial_encoded, temporal_encoded], dim=-1)  # [N, D_s + D_t]
        
        # Compute deformation
        output = self.deformation_mlp(encoded)  # [N, 4]
        
        # Split output
        delta_xyz = output[:, :3]  # [N, 3]
        delta_alpha = output[:, 3:4]  # [N, 1]
        
        return delta_xyz, delta_alpha
    
    def apply_deformation(
        self,
        xyz: torch.Tensor,
        alpha: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply deformation to Gaussian parameters.
        
        Args:
            xyz: [N, 3] original positions
            alpha: [N, 1] original opacities
            t: [N, 1] or [1] time
        
        Returns:
            deformed_xyz: [N, 3] deformed positions
            deformed_alpha: [N, 1] deformed opacities
        """
        # Compute deformation
        delta_xyz, delta_alpha = self.forward(xyz, t)
        
        # Apply deformation
        deformed_xyz = xyz + delta_xyz
        deformed_alpha = alpha * torch.sigmoid(delta_alpha)  # Multiplicative change
        
        return deformed_xyz, deformed_alpha


class ControlNodeMotionField(nn.Module):
    """
    Control node-based motion field.
    
    Uses a set of control nodes with learnable positions and radii,
    and interpolates motion using radial basis functions (RBF).
    """
    
    def __init__(
        self,
        num_control_nodes: int = 100,
        spatial_dim: int = 3,
        init_radius: float = 0.1,
    ):
        """
        Initialize control node motion field.
        
        Args:
            num_control_nodes: Number of control nodes
            spatial_dim: Spatial dimension
            init_radius: Initial radius for RBF kernels
        """
        super().__init__()
        
        self.num_control_nodes = num_control_nodes
        self.spatial_dim = spatial_dim
        
        # Control node positions [M, 3]
        self.control_positions = nn.Parameter(torch.randn(num_control_nodes, spatial_dim) * 0.1)
        
        # Control node radii [M, 1]
        self.control_radii = nn.Parameter(torch.ones(num_control_nodes, 1) * init_radius)
        
        # Deformation network for control nodes
        self.deformation_net = DeformationNetwork(
            spatial_dim=spatial_dim,
            hidden_dim=128,
            num_layers=4,
        )
    
    def compute_rbf_weights(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF weights for interpolation.
        
        Args:
            points: [N, 3] query points
        
        Returns:
            weights: [N, M] RBF weights
        """
        # Compute distances to control nodes
        diff = points.unsqueeze(1) - self.control_positions.unsqueeze(0)  # [N, M, 3]
        distances = torch.norm(diff, dim=-1)  # [N, M]
        
        # Compute RBF weights
        radii = torch.abs(self.control_radii.squeeze())  # [M]
        weights = torch.exp(-0.5 * (distances / radii) ** 2)  # [N, M]
        
        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights
    
    def forward(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute motion at query points.
        
        Args:
            xyz: [N, 3] query points
            t: [N, 1] or [1] time
        
        Returns:
            delta_xyz: [N, 3] motion vectors
            delta_alpha: [N, 1] opacity changes
        """
        # Compute deformation at control nodes
        control_delta_xyz, control_delta_alpha = self.deformation_net(
            self.control_positions, t
        )  # [M, 3], [M, 1]
        
        # Compute RBF weights
        weights = self.compute_rbf_weights(xyz)  # [N, M]
        
        # Interpolate motion
        delta_xyz = torch.matmul(weights, control_delta_xyz)  # [N, 3]
        delta_alpha = torch.matmul(weights, control_delta_alpha)  # [N, 1]
        
        return delta_xyz, delta_alpha
