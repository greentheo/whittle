"""The physics: what a candidate is scored against.

Both models here are deliberately approximate -- good enough to rank designs
against each other and to say roughly how far a design is from failing, which
is what a first-pass hobby optimization needs. Neither is a substitute for
building the part and testing it.
"""

from .drag import DragModel, DragReport
from .fea import MATERIALS, FEAResult, FieldLoad, Load, Material, Support, VoxelFEA, material
from .frame import TUBES, Section, section, tube, tube_dimensions, tube_size

__all__ = [
    "MATERIALS",
    "TUBES",
    "DragModel",
    "DragReport",
    "FEAResult",
    "FieldLoad",
    "Load",
    "Material",
    "Section",
    "Support",
    "VoxelFEA",
    "material",
    "section",
    "tube",
    "tube_dimensions",
    "tube_size",
]
