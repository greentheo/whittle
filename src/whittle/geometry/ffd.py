"""Free-form deformation: the representation that preserves topology."""

from __future__ import annotations

import numpy as np
from scipy.special import comb

from ..types import FloatArray, Genome, Mesh


def _bernstein(degree: int, t: FloatArray) -> FloatArray:
    """Bernstein basis of the given degree, evaluated at ``t``. Shape ``(len(t), degree + 1)``."""
    i = np.arange(degree + 1)
    coeff = comb(degree, i)
    return np.asarray(
        coeff * np.power.outer(t, i) * np.power.outer(1.0 - t, degree - i),
        dtype=np.float64,
    )


class FFDLattice:
    """Deforms a fixed base mesh with a trivariate Bezier control lattice.

    The genome is the displacement of every control point. Because every vertex
    moves as a smooth blend of those displacements, the result stays watertight
    and self-consistent for free -- no repair step, no risk of the optimizer
    proposing something unprintable by tearing the surface.

    The flip side is the hard limit named in :mod:`whittle.geometry.base`: this
    cannot change topology. It will tune a duct's curvature or a wing's camber.
    It will never discover that the part wanted a hole through the middle.

    The Bernstein basis is evaluated once per vertex at construction, because
    the base mesh does not move. After that a candidate costs one matrix
    multiply, which matters when the optimizer wants tens of thousands of them.
    """

    changes_topology = False

    def __init__(
        self,
        base: Mesh,
        divisions: tuple[int, int, int] = (3, 3, 3),
        *,
        max_displacement: float = 1.0,
        frozen_axes: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        if any(d < 1 for d in divisions):
            raise ValueError(f"divisions must be at least 1 in every axis, got {divisions}")
        if not len(base.vertices):
            raise ValueError("base mesh has no vertices")
        self.base = base
        self.divisions = divisions
        self.max_displacement = float(max_displacement)
        self.frozen_axes = frozen_axes
        self._basis = self._build_basis()

    def _build_basis(self) -> FloatArray:
        """Per-vertex lattice weights, flattened to ``(n_vertices, n_control_points)``."""
        lo, hi = self.base.bounds
        extent = np.where(hi - lo > 0, hi - lo, 1.0)
        local = np.clip((self.base.vertices - lo) / extent, 0.0, 1.0)
        bases = [_bernstein(d, local[:, axis]) for axis, d in enumerate(self.divisions)]
        # Outer product across the three axes, per vertex.
        weights = (
            bases[0][:, :, None, None]
            * bases[1][:, None, :, None]
            * bases[2][:, None, None, :]
        )
        result: FloatArray = weights.reshape(len(self.base.vertices), -1)
        return result

    @property
    def n_control_points(self) -> int:
        return int(np.prod([d + 1 for d in self.divisions]))

    @property
    def n_params(self) -> int:
        return self.n_control_points * 3

    @property
    def bounds(self) -> list[tuple[float, float]]:
        d = self.max_displacement
        per_point = [(0.0, 0.0) if frozen else (-d, d) for frozen in self.frozen_axes]
        return per_point * self.n_control_points

    def build(self, genome: Genome) -> Mesh:
        genome = np.asarray(genome, dtype=np.float64)
        if genome.size != self.n_params:
            raise ValueError(f"expected genome of {self.n_params} values, got {genome.size}")
        displacements = genome.reshape(self.n_control_points, 3)
        moved: FloatArray = self.base.vertices + self._basis @ displacements
        return Mesh(moved, self.base.faces)
