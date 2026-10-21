"""Core value types.

A genome is a plain float vector, because that is what every off-the-shelf
evolutionary optimizer wants. Everything that makes a genome mean something --
how many parameters, what range they live in, what geometry they describe -- is
the geometry backend's business, not the optimizer's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: A point or direction in model space. Model units are millimetres wherever
#: physics is involved, because that is what STL files from CAD and slicers use.
Vector3 = tuple[float, float, float]

#: A candidate solution, as the optimizer sees it: an unlabelled float vector.
Genome = FloatArray


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh. The boundary format of this whole library."""

    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (n, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (m, 3), got {self.faces.shape}")
        if len(self.faces) and int(self.faces.max()) >= len(self.vertices):
            raise ValueError("face index out of range")

    @property
    def volume(self) -> float:
        """Signed volume via the divergence theorem.

        Meaningful only for a closed, consistently wound surface. Marching cubes
        gives us one; a mesh someone exported from CAD may not, which is why
        :func:`whittle.io.load` does not promise it.
        """
        if not len(self.faces):
            return 0.0
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    @property
    def area(self) -> float:
        if not len(self.faces):
            return 0.0
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)

    @property
    def bounds(self) -> tuple[FloatArray, FloatArray]:
        if not len(self.vertices):
            zero = np.zeros(3, dtype=np.float64)
            return zero, zero.copy()
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
