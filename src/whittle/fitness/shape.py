"""Geometric constraints, scored as penalties: zero when satisfied, growing with the violation.

A penalty that grows with *how far* a candidate is out of bounds, rather than a
flat cost for being out at all, tells the search which infeasible proposals
were nearly acceptable -- the same reasoning as
:class:`~whittle.fitness.structural.AnchorConnectivity`'s distance term.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from ..types import BoolArray, FloatArray, Mesh, Vector3
from ..voxel import Grid
from .base import Candidate


class StayInside:
    """How far the candidate's surface pokes out of a solid it must stay within.

    For carving: a shape cut down from the original part never grows past it.
    Measured as the mean, over vertices, of how far each is outside, in voxels
    of ``grid``, beyond a ``tolerance`` that absorbs the half-voxel slop of a
    surface meshed from voxels in the first place.
    """

    name = "outside"

    def __init__(self, grid: Grid, solid: BoolArray, *, tolerance: float = 0.75) -> None:
        solid = np.asarray(solid, dtype=bool)
        if solid.shape != grid.shape:
            raise ValueError(f"solid must have shape {grid.shape}, got {solid.shape}")
        # Signed distance in voxels, positive outside. Distances run between
        # voxel centres, and the surface sits half a voxel from either side.
        outside = ndimage.distance_transform_edt(~solid) - 0.5
        inside = ndimage.distance_transform_edt(solid) - 0.5
        self._distance: FloatArray = np.where(solid, -inside, outside)
        self.grid = grid
        self.tolerance = float(tolerance)

    def __call__(self, candidate: Candidate) -> float:
        v = candidate.mesh.vertices
        index = (v - np.asarray(self.grid.origin)) / self.grid.spacing
        distance = ndimage.map_coordinates(self._distance, index.T, order=1, mode="nearest")
        return float(np.mean(np.maximum(distance - self.tolerance, 0.0)))


class VolumeBudget:
    """How far past ``max_removed`` of the original volume the candidate has cut."""

    name = "over_budget"

    def __init__(self, original_volume: float, max_removed: float) -> None:
        if original_volume <= 0:
            raise ValueError("original_volume must be positive")
        self.original_volume = float(original_volume)
        self.max_removed = float(max_removed)

    def __call__(self, candidate: Candidate) -> float:
        removed = 1.0 - candidate.mesh.volume / self.original_volume
        return max(0.0, removed - self.max_removed)


class NoFolding:
    """Fraction of faces turned inside out relative to the base mesh.

    A deformation lattice pushed hard enough folds the surface through itself.
    The mesh stays topologically valid and its volume can even look right, but
    it is not a shape anyone can build. A flipped face normal is the cheap tell.
    """

    name = "folded"

    def __init__(self, base: Mesh) -> None:
        self._normals = _normals(base)

    def __call__(self, candidate: Candidate) -> float:
        dots = np.einsum("ij,ij->i", _normals(candidate.mesh), self._normals)
        return float(np.mean(dots < 0))


class KeepOut:
    """How deep the candidate's surface cuts into boxes that must stay whole.

    For the volume a part exists to enclose: the mattress in a rooftop tent,
    the cargo in a roof box. Mean penetration of vertices inside each box, as
    a fraction of that box's smallest side.
    """

    name = "intrusion"

    def __init__(self, boxes: Sequence[tuple[Vector3, Vector3]]) -> None:
        self.boxes = [
            (np.minimum(lo, hi).astype(np.float64), np.maximum(lo, hi).astype(np.float64))
            for lo, hi in (map(np.asarray, box) for box in boxes)
        ]

    def __call__(self, candidate: Candidate) -> float:
        v = candidate.mesh.vertices
        total = 0.0
        for lo, hi in self.boxes:
            depth = np.minimum(v - lo, hi - v).min(axis=1)
            total += float(np.mean(np.maximum(depth, 0.0))) / float((hi - lo).min())
        return total


def _normals(mesh: Mesh) -> FloatArray:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1, keepdims=True)
    result: FloatArray = cross / np.where(length > 0, length, 1.0)
    return result
