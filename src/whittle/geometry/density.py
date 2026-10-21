"""Density-field geometry: the representation that can change topology."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure

from ..types import FloatArray, Genome, IntArray, Mesh, Vector3


def contour(
    grid: FloatArray,
    level: float,
    *,
    spacing: float = 1.0,
    origin: Vector3 = (0.0, 0.0, 0.0),
) -> Mesh:
    """Mesh the ``level`` isosurface of a voxel grid with marching cubes.

    Voxel ``(i, j, k)`` is sampled at ``origin + spacing * (i, j, k)``, which is
    the same convention :class:`whittle.voxel.Grid` uses, so a part voxelized
    from an STL comes back out where it went in.
    """
    # Pad with empty space so the surface closes at the grid boundary
    # instead of being left open where material runs off the edge.
    padded = np.pad(grid, 1, mode="constant", constant_values=0.0)
    if padded.max() <= level or padded.min() >= level:
        return Mesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64))
    verts, faces, _, _ = measure.marching_cubes(  # type: ignore[no-untyped-call]
        padded, level=level
    )
    vertices: FloatArray = (np.asarray(verts, dtype=np.float64) - 1.0) * spacing + np.asarray(
        origin, dtype=np.float64
    )
    # marching_cubes winds its triangles so the normals point into the
    # solid. Reverse them: an STL with inward normals is read inside-out by
    # every slicer, and the signed volume comes out negative.
    wound: IntArray = np.asarray(faces, dtype=np.int64)[:, ::-1]
    return Mesh(vertices, np.ascontiguousarray(wound))


class DensityField:
    """A voxel density grid, parameterized by a coarser grid of control values.

    The genome is the coarse grid. It is upsampled to the evaluation grid and
    thresholded to produce a surface. Two reasons the genome is not simply the
    evaluation grid itself:

    * Dimension. A 48x24x24 evaluation grid is 27,648 free parameters, which no
      evolutionary method will search in reasonable time. A 12x6x6 control grid
      is 432, which is tractable.
    * Smoothness. Interpolating up imposes a length scale on the result, which
      suppresses the single-voxel checkerboarding that density methods are prone
      to, without a separate filtering step.

    ``keep`` and ``void`` are the boundary conditions of the design problem:
    where material must exist (load pads, bolt bosses, mounting faces) and where
    it must not (clearance envelopes, tool access). They are applied after
    upsampling, so the optimizer cannot erode them no matter what it proposes.
    """

    changes_topology = True

    def __init__(
        self,
        shape: tuple[int, int, int],
        control_shape: tuple[int, int, int],
        *,
        spacing: float = 1.0,
        origin: Vector3 = (0.0, 0.0, 0.0),
        keep: np.ndarray | None = None,
        void: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> None:
        if any(s < 2 for s in shape):
            raise ValueError(f"shape must be at least 2 in every axis, got {shape}")
        if any(c < 2 for c in control_shape):
            raise ValueError(f"control_shape must be at least 2 in every axis, got {control_shape}")
        if any(c > s for c, s in zip(control_shape, shape, strict=True)):
            raise ValueError(f"control_shape {control_shape} must not exceed shape {shape}")
        self.shape = shape
        self.control_shape = control_shape
        self.spacing = float(spacing)
        self.origin = origin
        self.threshold = float(threshold)
        self.keep = self._check_mask(keep, "keep")
        self.void = self._check_mask(void, "void")
        if self.keep is not None and self.void is not None and (self.keep & self.void).any():
            raise ValueError("keep and void overlap: a voxel cannot be both forced and forbidden")

    def _check_mask(self, mask: np.ndarray | None, name: str) -> np.ndarray | None:
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != self.shape:
            raise ValueError(f"{name} mask must have shape {self.shape}, got {arr.shape}")
        return arr

    @property
    def n_params(self) -> int:
        return int(np.prod(self.control_shape))

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.n_params

    def field(self, genome: Genome) -> FloatArray:
        """Upsample the control grid to the evaluation grid and apply masks."""
        genome = np.asarray(genome, dtype=np.float64)
        if genome.size != self.n_params:
            raise ValueError(f"expected genome of {self.n_params} values, got {genome.size}")
        control = genome.reshape(self.control_shape)
        zoom = tuple(s / c for s, c in zip(self.shape, self.control_shape, strict=True))
        grid: FloatArray = ndimage.zoom(control, zoom, order=3, mode="nearest")
        # zoom's rounding can be a voxel out; trim or pad to be certain.
        grid = grid[tuple(slice(0, s) for s in self.shape)]
        if grid.shape != self.shape:
            padding = [(0, s - g) for s, g in zip(self.shape, grid.shape, strict=True)]
            grid = np.pad(grid, padding, mode="edge")
        grid = np.clip(grid, 0.0, 1.0)
        if self.keep is not None:
            grid[self.keep] = 1.0
        if self.void is not None:
            grid[self.void] = 0.0
        return grid

    def build(self, genome: Genome) -> Mesh:
        """Mesh the thresholded field with marching cubes."""
        return self.mesh_field(self.field(genome))

    def mesh_field(self, grid: FloatArray) -> Mesh:
        """Mesh an already-decoded field. Useful when an objective has one in hand."""
        return contour(grid, self.threshold, spacing=self.spacing, origin=self.origin)

    def solid(self, genome: Genome) -> np.ndarray:
        """The thresholded boolean occupancy, for connectivity-style objectives."""
        return self.field(genome) >= self.threshold

    @property
    def voxel_volume(self) -> float:
        return self.spacing**3

    @property
    def design_volume(self) -> float:
        """Volume of the design box, the denominator for a material fraction."""
        return float(np.prod(self.shape)) * self.voxel_volume
