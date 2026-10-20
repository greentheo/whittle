"""whittle -- evolutionary optimization of 3D parts.

Take material away until only what carries the load is left; or push a surface
around until the flow over it costs less. One evolutionary core, two geometry
representations, OBJ and STL at both ends.

    from whittle import DensityField, MaterialFraction, evolve, io

See README.md for what is implemented and what is honestly not.
"""

from . import io
from .fitness import (
    AnchorConnectivity,
    Candidate,
    Compliance,
    DragCoefficient,
    MaterialFraction,
    MeshVolume,
    WeightedSum,
)
from .geometry import DensityField, FFDLattice
from .optimize import EvolutionResult, evolve
from .types import Mesh

__version__ = "0.1.0"

__all__ = [
    "AnchorConnectivity",
    "Candidate",
    "Compliance",
    "DensityField",
    "DragCoefficient",
    "EvolutionResult",
    "FFDLattice",
    "MaterialFraction",
    "Mesh",
    "MeshVolume",
    "WeightedSum",
    "__version__",
    "evolve",
    "io",
]
