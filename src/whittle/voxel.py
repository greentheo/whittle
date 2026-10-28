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

from collections.abc import Iterator
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

    def nearest(self, points: FloatArray, within: BoolArray) -> tuple[IntArray, IntArray, IntArray]:
        """For each point, the index of the voxel of ``within`` nearest to it.

        For points on a closed surface meshed around ``within``, that is the
        voxel just beneath each one: how a face of a surface finds the
        material it belongs to.
        """
        if not within.any():
            raise ValueError("there is no material to find")
        _, nearest = ndimage.distance_transform_edt(~within, return_indices=True)
        index = np.rint((np.asarray(points) - np.asarray(self.origin)) / self.spacing)
        index = np.clip(index.astype(np.int64), 0, np.asarray(self.shape) - 1)
        i, j, k = (nearest[axis][index[:, 0], index[:, 1], index[:, 2]] for axis in range(3))
        return i, j, k

    def voxelize(self, mesh: Mesh) -> BoolArray:
        """Solid occupancy of a closed mesh: which voxel centres are inside it.

        Ray parity along z. Every triangle stamps a toggle at the first voxel
        above the point where it crosses each (x, y) column; a running sum up
        the column then counts the crossings below each voxel, and an odd count
        means inside. The mesh must be watertight -- a hole lets a ray leave
        without being counted, and the column fills to the top.
        """
        nx, ny, nz = self.shape
        toggles = np.zeros((nx, ny, nz + 1), dtype=np.int64)
        for i, j, z in self._crossings(mesh):
            # The first voxel whose centre sits above the crossing.
            k = np.clip(np.ceil((z - self.origin[2]) / self.spacing), 0, nz).astype(np.int64)
            np.add.at(toggles, (i, j, k), 1)
        parity = np.cumsum(toggles[:, :, :nz], axis=2) % 2
        solid: BoolArray = parity == 1
        return solid

    def coverage(self, mesh: Mesh, *, samples: int = 8) -> FloatArray:
        """How much of each voxel lies inside a closed mesh, from 0 to 1.

        :meth:`voxelize` asks only whether each centre is inside, so a surface
        meshed from it lands on whichever voxel boundary comes next -- up to
        half a voxel out, and on a flat face the same way all along it. This
        measures instead: exactly along z, from where each ray really crosses
        the surface, and from ``samples`` by ``samples`` rays across each
        voxel in x and y. Meshed from it, a surface comes back within about a
        tenth of a voxel of where it was.
        """
        nx, ny, nz = self.shape
        s = self.spacing
        step = s / samples
        rays = Grid(
            (nx * samples, ny * samples, 1),
            step,
            (self.origin[0] - (s - step) / 2.0, self.origin[1] - (s - step) / 2.0, 0.0),
        )
        found = list(rays._crossings(mesh))
        if not found:
            return np.zeros(self.shape, dtype=np.float64)
        i = np.concatenate([f[0] for f in found])
        j = np.concatenate([f[1] for f in found])
        z = np.concatenate([f[2] for f in found])
        # Up each ray, crossings alternate between entering and leaving.
        ray = i * (ny * samples) + j
        order = np.lexsort((z, ray))
        i, j, z, ray = i[order], j[order], z[order], ray[order]
        first = np.r_[True, ray[1:] != ray[:-1]]
        rank = np.arange(len(ray)) - np.maximum.accumulate(np.where(first, np.arange(len(ray)), 0))
        sign = np.where(rank % 2 == 0, 1.0, -1.0) / samples**2
        # Each crossing adds (or takes away) the part of its own voxel above it
        # and every voxel above that: a step, which a running sum up the column
        # turns into how much of each voxel is inside. Averaging the rays of a
        # voxel commutes with the sum, so it happens on the voxel's own column.
        height = (z - self.origin[2]) / s + 0.5
        column = (i // samples, j // samples)
        steps = np.zeros((nx, ny, nz + 1), dtype=np.float64)
        below = height < 0
        bottom = np.zeros(int(below.sum()), dtype=np.int64)
        np.add.at(steps, (column[0][below], column[1][below], bottom), sign[below])
        within = (height >= 0) & (height < nz)
        k = np.floor(height[within]).astype(np.int64)
        part = height[within] - k
        at = (column[0][within], column[1][within])
        np.add.at(steps, (*at, k), sign[within] * (1.0 - part))
        np.add.at(steps, (*at, k + 1), sign[within] * part)
        filled: FloatArray = np.clip(np.cumsum(steps[:, :, :nz], axis=2), 0.0, 1.0)
        return filled

    def _crossings(self, mesh: Mesh) -> Iterator[tuple[IntArray, IntArray, FloatArray]]:
        """Where the mesh crosses each column of voxels: column ``i``, ``j`` and the height ``z``.

        Yielded in chunks, so the (triangle, column) pairs in flight stay bounded.
        """
        nx, ny, _ = self.shape
        s = self.spacing
        ox, oy, _ = self.origin
        if not len(mesh.faces):
            return
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

        budget = 2_000_000
        start = 0
        n_tri = len(tri)
        while start < n_tri:
            cumulative = np.cumsum(counts[start:])
            stop = start + max(1, int(np.searchsorted(cumulative, budget)))
            found = _hits(np.arange(start, min(stop, n_tri)), counts, width, i_lo, j_lo, px, py, pz)
            if found is not None:
                yield found
            start = stop

    def share(self, design: BoolArray, solid: BoolArray, coverage: FloatArray) -> FloatArray:
        """How much of each voxel is still the part, once ``design`` is all that is left of it.

        ``solid`` is the part voxelized and ``coverage`` how much of each voxel
        it filled; ``design`` is what of ``solid`` remains after material was
        taken away. A voxel of the design keeps what the part filled of it, a
        voxel outside the part keeps the sliver of surface it held, and a voxel
        removed keeps nothing. Summed, it is the design's real volume, in voxels
        -- where counting the design's voxels counts the part up to half a voxel
        bigger on every side.
        """
        kept: FloatArray = np.where(design | ~solid, coverage, 0.0)
        return kept

    def fitted(
        self, design: BoolArray, solid: BoolArray, coverage: FloatArray, *, smooth: float = 0.6
    ) -> FloatArray:
        """An occupancy to mesh ``design`` by, whose outside lands where the part's surface was.

        ``design`` is what is left of ``solid`` -- the part, voxelized -- after
        material was taken away, and ``coverage`` is :meth:`coverage` of the
        part. Its outside comes from ``coverage``, each voxel held to its own
        side of one half: a voxel of the part always meshes as material however
        thin a sliver of it the part filled, and a sliver outside never grows
        into material. What was taken away comes from ``design``, blurred by
        ``smooth`` voxels as :meth:`mesh` would blur it, so pockets get walls
        rather than steps. Mesh the result with no further smoothing.
        """
        outside = np.where(
            solid, np.maximum(coverage, 0.5 + 1e-3), np.minimum(coverage, 0.5 - 1e-3)
        )
        removed = (solid & ~design).astype(np.float64)
        if smooth > 0:
            removed = ndimage.gaussian_filter(removed, smooth, mode="constant")
        field: FloatArray = np.minimum(outside, 1.0 - removed)
        return field

    def mesh(
        self, solid: BoolArray | FloatArray, *, smooth: float = 0.6, relax: int = 0
    ) -> Mesh:
        """Mesh an occupancy as a closed, printable surface.

        ``solid`` is either which voxels are material, or how much of each is,
        as :meth:`coverage` or :meth:`fitted` give -- which puts the surface
        where it really was rather than on the nearest voxel boundary.

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
        origin = self.origin
        if solid.dtype != np.bool_:
            # A fraction filled: carry it a voxel past the surface, into a
            # margin of empty voxels so a part filling the grid gets it too.
            field = _reach(np.pad(field, 1, mode="constant", constant_values=0.0))
            origin = (origin[0] - self.spacing, origin[1] - self.spacing, origin[2] - self.spacing)
        mesh = contour(field, 0.5, spacing=self.spacing, origin=origin)
        return taubin(mesh, relax) if relax > 0 else mesh


def _reach(filled: FloatArray) -> FloatArray:
    """Carry a fraction-filled field one voxel past the surface, so a flat face meshes true.

    Marching cubes puts the surface where the field crosses one half, by
    straight-line interpolation between voxel centres. Between a voxel 70%
    filled and an empty one that lands 0.29 of a voxel out, where the face
    really is 0.2 out: an empty voxel reads 0 however far the surface is from
    it. Reading it instead as its filled neighbour less one -- -0.3 here --
    puts the crossing exactly on the face, and likewise a full voxel beside a
    sliver as one plus the sliver. Where a voxel has more than one such
    neighbour, the one nearest the surface wins.

    Exact for a face the part fills to one side of; a part thinner than a
    voxel, with both its faces in one, comes out thinner still.
    """
    padded = np.pad(filled, 1, mode="constant", constant_values=0.0)
    empty = padded <= 1e-9
    full = padded >= 1.0 - 1e-9
    outward = np.full(padded.shape, -np.inf)
    inward = np.full(padded.shape, np.inf)
    for axis in range(3):
        for shift in (1, -1):
            near = np.roll(padded, shift, axis=axis)
            material = (near > 0.5) & (near < 1.0)
            sliver = (near > 0.0) & (near < 0.5)
            outward = np.where(material, np.maximum(outward, near - 1.0), outward)
            inward = np.where(sliver, np.minimum(inward, near + 1.0), inward)
    reached = np.where(empty & np.isfinite(outward), outward, padded)
    reached = np.where(full & np.isfinite(inward), inward, reached)
    result: FloatArray = reached[1:-1, 1:-1, 1:-1]
    return result


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


def _hits(
    chunk: IntArray,
    counts: IntArray,
    width: IntArray,
    i_lo: IntArray,
    j_lo: IntArray,
    px: FloatArray,
    py: FloatArray,
    pz: FloatArray,
) -> tuple[IntArray, IntArray, FloatArray] | None:
    """Where each triangle in ``chunk`` crosses the columns under it, if anywhere."""
    n = counts[chunk]
    if not n.sum():
        return None
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
        return None
    z = l0 * pz[tri, 0] + l1 * pz[tri, 1] + l2 * pz[tri, 2]
    return i[hit], j[hit], z[hit]
