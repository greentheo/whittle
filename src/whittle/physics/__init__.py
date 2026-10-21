"""The physics: what a candidate is scored against.

Both models here are deliberately approximate -- good enough to rank designs
against each other and to say roughly how far a design is from failing, which
is what a first-pass hobby optimization needs. Neither is a substitute for
building the part and testing it.
"""

from .drag import DragModel, DragReport
from .fea import MATERIALS, FEAResult, Load, Material, Support, VoxelFEA, material

__all__ = [
    "MATERIALS",
    "DragModel",
    "DragReport",
    "FEAResult",
    "Load",
    "Material",
    "Support",
    "VoxelFEA",
    "material",
]
