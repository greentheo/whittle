"""Free-form deformation: the representation that preserves topology."""

from __future__ import annotations

import numpy as np
from scipy.special import comb

from ..types import BoolArray, FloatArray, Genome, IntArray, Mesh, Vector3


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

    Three ways to cut the search down to the shapes that make sense, each of
    which also shrinks the genome:

    * ``frozen`` -- per control point and axis, ``(d0 + 1, d1 + 1, d2 + 1, 3)``,
      True where that point may not move along that axis. Freezing the z of
      the bottom layer keeps a mounting face flat and where it was, because a
      Bezier volume's boundary face depends only on its boundary layer.
    * ``mirror`` -- an axis to keep the deformation symmetric about, through
      the middle of the base mesh. Half the control points are searched and
      the other half follow.
    * ``inward`` -- control points on the outside of the lattice may only move
      into it, along each axis they are on the outside of. For carving a shape
      down rather than growing it.

    And one that does not shrink it: ``weights``, one per vertex, scales how
    far each vertex follows the lattice. Zero pins a vertex where it is, one
    lets it move freely, and a ramp between keeps the surface smooth where a
    pinned region meets a free one. This is how only chosen surfaces get
    reshaped while the rest stay exactly as they were.
    """

    changes_topology = False

    def __init__(
        self,
        base: Mesh,
        divisions: tuple[int, int, int] = (3, 3, 3),
        *,
        max_displacement: float | Vector3 = 1.0,
        frozen_axes: tuple[bool, bool, bool] = (False, False, False),
        frozen: BoolArray | None = None,
        mirror: int | None = None,
        inward: bool = False,
        weights: FloatArray | None = None,
    ) -> None:
        if any(d < 1 for d in divisions):
            raise ValueError(f"divisions must be at least 1 in every axis, got {divisions}")
        if not len(base.vertices):
            raise ValueError("base mesh has no vertices")
        if mirror is not None and mirror not in (0, 1, 2):
            raise ValueError(f"mirror must be an axis 0, 1 or 2, got {mirror}")
        self.base = base
        self.divisions = divisions
        self.max_displacement = np.broadcast_to(
            np.asarray(max_displacement, dtype=np.float64), (3,)
        ).copy()
        self.frozen_axes = frozen_axes
        self.mirror = mirror
        self.inward = inward
        self._basis = self._build_basis()
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(base.vertices),):
                raise ValueError(f"weights must be one per vertex, got {weights.shape}")
            # Scaling a vertex's rows of the basis scales its displacement, at
            # no cost per candidate: a vertex with weight 0 never moves.
            self._basis = self._basis * weights[:, None]

        dims = tuple(d + 1 for d in divisions)
        if frozen is None:
            frozen = np.zeros((*dims, 3), dtype=bool)
        frozen = np.asarray(frozen, dtype=bool)
        if frozen.shape != (*dims, 3):
            raise ValueError(f"frozen must have shape {(*dims, 3)}, got {frozen.shape}")
        index = np.indices(dims).reshape(3, -1).T
        held = frozen.reshape(-1, 3) | np.asarray(frozen_axes, dtype=bool)

        # Which searched point each control point copies, and whether it copies
        # it mirrored. Without a mirror every point is its own source.
        source = np.arange(len(index))
        flipped = np.zeros(len(index), dtype=bool)
        searched = np.ones(len(index), dtype=bool)
        if mirror is not None:
            last = dims[mirror] - 1
            upper = index[:, mirror] > last // 2
            reflected = index.copy()
            reflected[:, mirror] = last - index[:, mirror]
            source = np.where(upper, np.ravel_multi_index(tuple(reflected.T), dims), source)
            flipped = upper
            searched = ~upper
            if last % 2 == 0:
                # Points on the mirror plane cannot move off it.
                held[index[:, mirror] == last // 2, mirror] = True
        self._searched: IntArray = np.flatnonzero(searched)
        slot = np.full(len(index), -1, dtype=np.int64)
        slot[self._searched] = np.arange(len(self._searched))
        self._slot: IntArray = slot[source]
        self._flipped = flipped

        lows = np.tile(-self.max_displacement, (len(self._searched), 1))
        highs = np.tile(self.max_displacement, (len(self._searched), 1))
        if inward:
            own = index[self._searched]
            top = np.asarray(divisions)
            lows = np.where(own == 0, 0.0, lows)
            highs = np.where(own == top, 0.0, highs)
        lows = np.where(held[self._searched], 0.0, lows)
        highs = np.where(held[self._searched], 0.0, highs)
        self._bounds = list(zip(lows.ravel().tolist(), highs.ravel().tolist(), strict=True))

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
        return len(self._searched) * 3

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return list(self._bounds)

    def displacements(self, genome: Genome) -> FloatArray:
        """Every control point's displacement, ``(n_control_points, 3)``, mirror applied."""
        genome = np.asarray(genome, dtype=np.float64)
        if genome.size != self.n_params:
            raise ValueError(f"expected genome of {self.n_params} values, got {genome.size}")
        moves: FloatArray = genome.reshape(-1, 3)[self._slot].copy()
        if self.mirror is not None:
            moves[self._flipped, self.mirror] *= -1.0
        return moves

    def build(self, genome: Genome) -> Mesh:
        moved: FloatArray = self.base.vertices + self._basis @ self.displacements(genome)
        return Mesh(moved, self.base.faces)
