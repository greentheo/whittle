"""The voxel grid: where a part, its loads and its supports all get coordinates.

Every physical problem in this library is posed on one regular grid. A part
comes in as a mesh, is voxelized onto the grid, and from then on the material,
the load pads, the bolt holes and the clearance envelopes are all boolean masks
of the same shape. One coordinate system, so a load can never be applied a
voxel off from the material it is meant to push on.

Model units are millimetres by convention, because that is what every slicer
and most CAD exports assume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .geometry.density import contour
from .types import BoolArray, FloatArray, IntArray, Mesh, Vector3


@dataclass(frozen=True)
class Grid:
    """A regular voxel grid in model space.

    Voxel ``(i, j, k)`` is the cube of side ``spacing`` centred on
    ``origin + spacing * (i, j, k)``.
    """

    shape: tuple[int, int, int]
    spacing: float
    origin: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(s < 1 for s in self.shape):
            raise ValueError(f"shape must be three positive sizes, got {self.shape}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @classmethod
    def around(
        cls,
        mesh: Mesh,
        *,
        spacing: float | None = None,
        resolution: int = 48,
        pad: int = 0,
    ) -> Grid:
        """The smallest grid that covers ``mesh``.

        Give ``spacing`` in model units, or ``resolution`` -- the number of
        voxels along the part's longest side -- and let the spacing follow.
        ``pad`` adds empty voxels on every side, which the flow model wants and
        the structural one does not.
        """
        if not len(mesh.vertices):
            raise ValueError("cannot fit a grid around an empty mesh")
        lo, hi = mesh.bounds
        extent = hi - lo
        if spacing is None:
            if resolution < 2:
                raise ValueError(f"resolution must be at least 2, got {resolution}")
            spacing = float(extent.max()) / resolution
        counts = np.maximum(np.ceil(extent / spacing).astype(np.int64), 1) + 2 * pad
        # Centre the grid on the part, so a symmetric part voxelizes symmetrically.
        middle = (lo + hi) / 2.0
        origin = middle - spacing * (counts - 1) / 2.0
        return cls(
            (int(counts[0]), int(counts[1]), int(counts[2])),
            float(spacing),
            (float(origin[0]), float(origin[1]), float(origin[2])),
        )

    @property
    def voxel_volume(self) -> float:
        return self.spacing**3

    def centers(self) -> FloatArray:
        """Model-space centre of every voxel, shape ``(*shape, 3)``."""
        index = np.indices(self.shape, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        centres: FloatArray = np.moveaxis(index, 0, -1) * self.spacing + origin
        return centres

    def box(self, lo: Vector3, hi: Vector3) -> BoolArray:
        """Voxels whose centres lie in the axis-aligned box ``[lo, hi]``."""
        c = self.centers()
        low = np.minimum(lo, hi)
        high = np.maximum(lo, hi)
        mask: BoolArray = np.all((c >= low) & (c <= high), axis=-1)
        return mask

    def ball(self, center: Vector3, radius: float) -> BoolArray:
        """Voxels whose centres lie within ``radius`` of ``center``."""
        distance = np.linalg.norm(self.centers() - np.asarray(center, dtype=np.float64), axis=-1)
        mask: BoolArray = distance <= radius
        return mask

    def near(self, point: Vector3, within: BoolArray, *, radius: float | None = None) -> BoolArray:
        """The voxels of ``within`` around ``point``: how a click becomes a region.

        A point picked on a surface sits half a voxel from the nearest voxel
        centre at best, so a bare ball around it can come back empty. This
        takes the ball when it catches something and otherwise the handful of
        voxels of ``within`` nearest the point, so a region is never empty.
        """
        if not within.any():
            raise ValueError("there is no material to select from")
        radius = 1.5 * self.spacing if radius is None else float(radius)
        region = self.ball(point, radius) & within
        if region.any():
            return region
        distance = np.linalg.norm(self.centers() - np.asarray(point, dtype=np.float64), axis=-1)
        distance[~within] = np.inf
        nearest = float(distance.min())
        result: BoolArray = distance <= nearest + 0.75 * self.spacing
        return result

    def patch(
        self,
        point: Vector3,
        normal: Vector3,
        within: BoolArray,
        *,
        depth: float | None = None,
        radius: float | None = None,
    ) -> BoolArray:
        """The face under a click: material just beneath the surface at ``point``.

        ``normal`` is the outward surface normal there, as a raycast returns
        it. Selects the voxels of ``within`` whose centres lie no more than
        ``depth`` beneath the plane through ``point`` -- two voxels by default
        -- and, of those, only the connected patch the click landed on. On a
        flat face that is the whole face; on a curved one, the part of it
        that lies within ``depth`` of flat. ``radius`` caps the patch's size.
        """
        n = np.asarray(normal, dtype=np.float64)
        length = float(np.linalg.norm(n))
        if length == 0:
            raise ValueError("normal must be non-zero")
        n /= length
        depth = 2.0 * self.spacing if depth is None else float(depth)
        offset = self.centers() - np.asarray(point, dtype=np.float64)
        below = -(offset @ n)
        # Voxel centres sit about half a voxel inside a surface meshed from
        # them, so the first layer is at half a voxel deep and the second at
        # one and a half. The margin above the plane absorbs a click that
        # lands a little proud of the voxels.
        slab = within & (below >= -0.5 * self.spacing) & (below <= depth)
        if radius is not None:
            slab &= np.linalg.norm(offset, axis=-1) <= radius
        seed = self.near(point, within, radius=0.0)
        labels, _ = ndimage.label(slab)
        touched = np.unique(labels[seed & slab])
        touched = touched[touched > 0]
        if not len(touched):
            return seed
        result: BoolArray = np.isin(labels, touched)
        return result

    def voxelize(self, mesh: Mesh) -> BoolArray:
        """Solid occupancy of a closed mesh: which voxel centres are inside it.

        Ray parity along z. Every triangle stamps a toggle at the first voxel
        above the point where it crosses each (x, y) column; a running sum up
        the column then counts the crossings below each voxel, and an odd count
        means inside. The mesh must be watertight -- a hole lets a ray leave
        without being counted, and the column fills to the top.
        """
        nx, ny, nz = self.shape
        s = self.spacing
        ox, oy, oz = self.origin
        if not len(mesh.faces):
            return np.zeros(self.shape, dtype=bool)
        tri = mesh.vertices[mesh.faces]
        # Nudge the rays off the lattice by an irrational fraction of a voxel,
        # so no ray passes exactly through a vertex or along an edge -- the
        # places where parity counting double-counts.
        px = (tri[..., 0] - ox) / s - 1.234567e-4
        py = (tri[..., 1] - oy) / s - 2.345678e-4
        pz = tri[..., 2]

        i_lo = np.clip(np.ceil(px.min(axis=1)), 0, nx).astype(np.int64)
        i_hi = np.clip(np.floor(px.max(axis=1)), -1, nx - 1).astype(np.int64)
        j_lo = np.clip(np.ceil(py.min(axis=1)), 0, ny).astype(np.int64)
        j_hi = np.clip(np.floor(py.max(axis=1)), -1, ny - 1).astype(np.int64)
        width = np.maximum(i_hi - i_lo + 1, 0)
        height = np.maximum(j_hi - j_lo + 1, 0)
        counts = width * height

        toggles = np.zeros((nx, ny, nz + 1), dtype=np.int64)
        budget = 2_000_000
        start = 0
        n_tri = len(tri)
        while start < n_tri:
            # Chunk so that the per-chunk (triangle, column) pairs stay bounded.
            cumulative = np.cumsum(counts[start:])
            stop = start + max(1, int(np.searchsorted(cumulative, budget)))
            _stamp(
                toggles,
                np.arange(start, min(stop, n_tri)),
                counts,
                width,
                i_lo,
                j_lo,
                px,
                py,
                pz,
                oz,
                s,
                nz,
            )
            start = stop
        parity = np.cumsum(toggles[:, :, :nz], axis=2) % 2
        solid: BoolArray = parity == 1
        return solid

    def mesh(self, solid: BoolArray, *, smooth: float = 0.6, relax: int = 0) -> Mesh:
        """Mesh a boolean occupancy as a closed, printable surface.

        The occupancy is blurred by ``smooth`` voxels before contouring, which
        turns the staircase of a raw voxel surface into something a slicer and
        an eye will both accept. Keep it below a voxel: more than that eats
        members that are only two voxels thick.

        ``relax`` runs that many rounds of Taubin smoothing on the result. The
        blur takes the corners off each step; it does not remove the terracing
        itself, and anything that reads surface normals -- the drag model
        above all -- sees a terraced dome as a stack of flat ledges.
        """
        if solid.shape != self.shape:
            raise ValueError(f"occupancy must have shape {self.shape}, got {solid.shape}")
        field = solid.astype(np.float64)
        if smooth > 0:
            field = ndimage.gaussian_filter(field, smooth, mode="constant")
        mesh = contour(field, 0.5, spacing=self.spacing, origin=self.origin)
        return taubin(mesh, relax) if relax > 0 else mesh


def taubin(mesh: Mesh, iterations: int, *, shrink: float = 0.5, inflate: float = -0.53) -> Mesh:
    """Taubin's two-step smoothing: remove surface ripple without shrinking the part.

    Plain Laplacian smoothing pulls every vertex toward its neighbours, which
    removes ripple and also shrinks the whole shape a little every pass. Taubin
    alternates that step with a slightly larger one in the opposite direction,
    which undoes the shrinkage at long wavelengths while still killing the
    short ones -- here, the voxel terraces.
    """
    if not len(mesh.faces):
        return mesh
    f = mesh.faces
    edges = np.unique(
        np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0
    )
    n = len(mesh.vertices)
    degree = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    degree[degree == 0] = 1.0
    v = mesh.vertices.copy()
    for _ in range(iterations):
        for factor in (shrink, inflate):
            total = np.zeros_like(v)
            for axis in range(3):
                total[:, axis] = np.bincount(
                    edges[:, 0], weights=v[edges[:, 1], axis], minlength=n
                ) + np.bincount(edges[:, 1], weights=v[edges[:, 0], axis], minlength=n)
            v += factor * (total / degree[:, None] - v)
    return Mesh(v, f)


def _stamp(
    toggles: IntArray,
    chunk: IntArray,
    counts: IntArray,
    width: IntArray,
    i_lo: IntArray,
    j_lo: IntArray,
    px: FloatArray,
    py: FloatArray,
    pz: FloatArray,
    oz: float,
    spacing: float,
    nz: int,
) -> None:
    """Record where each triangle in ``chunk`` crosses the columns under it."""
    n = counts[chunk]
    if not n.sum():
        return
    tri = np.repeat(chunk, n)
    # Position of each pair within its triangle's bounding box of columns.
    local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    i = i_lo[tri] + local % width[tri]
    j = j_lo[tri] + local // width[tri]

    x0, x1, x2 = (px[tri, c] for c in range(3))
    y0, y1, y2 = (py[tri, c] for c in range(3))
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    ok = det != 0
    det = np.where(ok, det, 1.0)
    l0 = ((y1 - y2) * (i - x2) + (x2 - x1) * (j - y2)) / det
    l1 = ((y2 - y0) * (i - x2) + (x0 - x2) * (j - y2)) / det
    l2 = 1.0 - l0 - l1
    hit = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    if not hit.any():
        return
    z = l0 * pz[tri, 0] + l1 * pz[tri, 1] + l2 * pz[tri, 2]
    # The first voxel whose centre sits above the crossing.
    k = np.clip(np.ceil((z[hit] - oz) / spacing), 0, nz).astype(np.int64)
    np.add.at(toggles, (i[hit], j[hit], k), 1)
