"""whittle -- evolutionary optimization of 3D parts.

Take material away until only what carries the load is left; or carve a shape
down until the air slips past it. Bring a part, say what pushes on it or which
way the wind blows, and get back a slider of designs from conservative to
extreme.

    from whittle import LoadStudy, FlowStudy, ParametricStudy, io

See README.md for what is implemented and how far to trust it.
"""

from . import io, shapes
from .dual import DualStudy, DualSweep
from .fitness import (
    AnchorConnectivity,
    Candidate,
    Compliance,
    DragArea,
    DragCoefficient,
    KeepOut,
    MaterialFraction,
    MeshVolume,
    NoFolding,
    StayInside,
    VolumeBudget,
    WeightedSum,
)
from .geometry import DensityField, FFDLattice
from .optimize import EvolutionResult, beso, evolve
from .parametric import BuildFailed, ParametricStudy
from .physics import (
    MATERIALS,
    DragModel,
    DragReport,
    FEAResult,
    FieldLoad,
    Load,
    Material,
    Support,
    VoxelFEA,
)
from .study import FlowStudy, LoadStudy
from .sweep import Station, Sweep
from .types import Mesh
from .voxel import Grid

__version__ = "0.2.0"

__all__ = [
    "MATERIALS",
    "AnchorConnectivity",
    "BuildFailed",
    "Candidate",
    "Compliance",
    "DensityField",
    "DragArea",
    "DragCoefficient",
    "DragModel",
    "DragReport",
    "DualStudy",
    "DualSweep",
    "EvolutionResult",
    "FEAResult",
    "FFDLattice",
    "FieldLoad",
    "FlowStudy",
    "Grid",
    "KeepOut",
    "Load",
    "LoadStudy",
    "Material",
    "MaterialFraction",
    "Mesh",
    "MeshVolume",
    "NoFolding",
    "ParametricStudy",
    "StayInside",
    "Station",
    "Support",
    "Sweep",
    "VolumeBudget",
    "VoxelFEA",
    "WeightedSum",
    "__version__",
    "beso",
    "evolve",
    "io",
    "shapes",
]
