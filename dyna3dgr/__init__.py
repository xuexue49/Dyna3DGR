"""
Dyna3DGR: 4D Cardiac Motion Tracking with Dynamic 3D Gaussian Representation

This package implements the method described in the paper:
"Dyna3DGR: 4D Cardiac Motion Tracking with Dynamic 3D Gaussian Representation"
(MICCAI 2025)

Main components:
- models: 3D Gaussian representation and deformation networks
- scene: Scene and camera management
- utils: Utility functions for training and evaluation
- data: Data loading and preprocessing
- training: Training pipeline and optimization
"""

__version__ = "0.1.0"
__author__ = "Dyna3DGR Contributors"
__license__ = "MIT"

from . import models
from . import scene
from . import utils
from . import data
from . import training

__all__ = [
    "models",
    "scene",
    "utils",
    "data",
    "training",
]
