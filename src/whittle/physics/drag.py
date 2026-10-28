"""A fast engineering estimate of the drag on a closed mesh.

Not a flow solver. A flow solver worth trusting at the speeds a roof box or a
bike fairing sees -- Reynolds numbers around a million, a turbulent boundary
layer, a separated wake -- needs a mesh of millions of cells and minutes to
hours per shape, and an evolutionary search wants thousands of shapes. This is
the other end of that trade: a per-triangle model that takes about a
millisecond, built from the handful of facts that decide most of the drag of a
bluff body at high Reynolds number.

1. **Faces into the wind are pushed on.** Pressure on a windward face follows
   the Newtonian ``Cp = s^2``, where ``s`` is how squarely it faces the flow.
   A flat front is the most expensive thing a shape can have.
2. **Air accelerating round a convex shoulder pulls forward.** On a convex
   windward face the pressure drops below ambient as it turns from the flow,
   ``Cp = s^2 - (1 - s^2)``, which integrates to zero net forebody drag over a
   hemisphere -- the observation that a well-rounded nose costs almost
   nothing. A concave face does the opposite and traps the air at stagnation,
   ``Cp = 1``, which is why a dish is worse than a plate. Convexity is read
   from the dihedral angles between neighbouring triangles.
3. **The flow leaves a surface that falls away too steeply.** A leeward face
   inclined more than about 20 degrees from the flow is in separated flow and
   feels the wake's base pressure, ``Cp ~ -0.15 to -0.45`` -- stronger behind
   a short body than a long one. A gentler taper stays attached and costs
   nothing but friction. This is the boat-tail rule, and it is most of why a
   tapered tail beats a square one. A long, thin silhouette -- a roof bar, a
   strut -- sheds a two-dimensional wake whose suction is far stronger, up to
   2.6 times as strong for an endless one, which follows Hoerner's flat
   plates: drag coefficient 1.18 face-on as a square, 1.3 at ten times as
   long as it is wide, 1.98 endless. Stubby 2D sections still come out low --
   an endless square bar scores about 1.7 where the measured value is 2.1.
4. **Skin friction** scales with attached wetted area, ``Cf ~ 0.0045``.
5. **What is behind something is sheltered.** A windward face with another
   part of the body upstream of it, along the flow, sits in a wake and is
   pushed on far less. A face is judged sheltered when a leeward face lies
   upstream of it in the same column along the flow. The something can also
   be an *obstacle* that is not part of the design -- the car a hitch box
   rides behind -- which shelters what is downstream of it but is never
   scored itself.

Calibrated against textbook high-Reynolds-number drag coefficients, it gives
roughly 1.3 for a flat plate face-on, 1.1 for a cube, 0.2 for a sphere above
its drag crisis, and under 0.1 for a slender streamlined body
(tests/test_drag.py holds it to those). Rankings of shapes are what to trust;
the absolute numbers are good to perhaps 30%.

What it does not see: interaction between separated regions, vortex shedding,
lift and the drag induced by it, ground effect, Reynolds-number dependence,
crosswind, and anything a sharp edge does to the flow downstream of it beyond
the face it is on. When those dominate -- a car's underbody, a wing, a gap
between two close bodies -- this will rank shapes wrongly, and the right next
step is a real CFD run on the two or three designs this one picked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.special import expit

from ..types import BoolArray, FloatArray, IntArray, Mesh, Vector3


@dataclass(frozen=True)
class DragReport:
    """The drag on one shape, broken down by where it comes from."""

    #: Drag area ``Cd * A``, in model units squared. Force is proportional to
    #: this, not to ``cd``: a shape can lower its coefficient by getting
    #: bigger, and that does not help anyone's fuel bill.
    drag_area: float
    #: Drag coefficient on the frontal area.
    cd: float
    #: Area of the shape projected onto the plane normal to the flow.
    frontal_area: float
    #: The parts of ``drag_area``: windward pressure, wake pressure, friction.
    forebody: float
    base: float
    friction: float
    #: Each face's share of ``drag_area`` -- where to look to see what costs what.
    per_face: FloatArray
    #: The whole aerodynamic force on each face, ``(faces, 3)``, in the same
    #: units as ``drag_area`` -- force over dynamic pressure. Its component
    #: along the flow is ``per_face``; the rest is side force and lift, which
    #: a structure feels even though the fuel bill does not.
    forces: FloatArray
    #: The flow direction this was scored against.
    flow: FloatArray

    def force(
        self, speed: float, *, air_density: float = 1.225, metres_per_unit: float = 1e-3
    ) -> float:
        """Drag in newtons at ``speed`` m/s, for a mesh in millimetres by default."""
        area_m2 = self.drag_area * metres_per_unit**2
        return 0.5 * air_density * speed**2 * area_m2

    def force_vector(
        self, speed: float, *, air_density: float = 1.225, metres_per_unit: float = 1e-3
    ) -> FloatArray:
        """The total aerodynamic force in newtons -- drag, side force and lift together."""
        q = 0.5 * air_density * speed**2 * metres_per_unit**2
        total: FloatArray = q * self.forces.sum(axis=0)
        return total


class DragModel:
    """Scores meshes against a fixed flow direction.

    ``flow`` is the direction the air moves past the part, as in a wind
    tunnel. A part on a car driving towards +x sees a flow of ``(-1, 0, 0)``.

    Mesh connectivity is cached, so scoring many deformations of one mesh --
    which is what a :class:`~whittle.geometry.ffd.FFDLattice` search does --
    pays for the edge table once.

    ``aspect`` overrides the measured length-to-width ratio of each body's
    silhouette; ``float("inf")`` scores the mesh as a slice of something
    endless, like a short length of roof bar standing in for the whole bar.
    ``ends`` then names the axis the slice was cut across, and faces facing
    along it -- the cut ends, which the real bar does not have -- are ignored.

    ``obstacles`` are fixed bodies in the same frame that the scored mesh sits
    in the wake of, like the car behind which a hitch box rides. A windward
    face of the mesh downstream of an obstacle's silhouette gets no pressure
    and no friction, but still counts towards the mesh's own frontal area, so
    ``cd`` stays a coefficient on the part's own size. The wake is taken as the
    obstacle's full silhouette however far back it reaches -- right for a box a
    foot or two behind a bumper, generous for a trailer ten metres back.
    """

    def __init__(
        self,
        flow: Vector3,
        *,
        stagnation: float = 0.85,
        separation_angle: float = 20.0,
        separation_width: float = 4.0,
        skin_friction: float = 0.0045,
        aspect: float | None = None,
        ends: int | None = None,
        obstacles: Sequence[Mesh] = (),
    ) -> None:
        direction = np.asarray(flow, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length == 0:
            raise ValueError("flow direction must be non-zero")
        self.flow: FloatArray = direction / length
        self.stagnation = float(stagnation)
        self.separation_angle = float(separation_angle)
        self.separation_width = float(separation_width)
        self.skin_friction = float(skin_friction)
        self.aspect = None if aspect is None else float(aspect)
        if self.aspect is not None and self.aspect < 1.0:
            raise ValueError(f"aspect is long side over short, so at least 1; got {aspect}")
        self.ends = ends
        # Two directions spanning the plane normal to the flow, for binning
        # faces into columns along it.
        helper = np.array([1.0, 0.0, 0.0]) if abs(self.flow[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(self.flow, helper)
        e1 /= np.linalg.norm(e1)
        self._plane: FloatArray = np.stack([e1, np.cross(self.flow, e1)])
        self._faces: IntArray | None = None
        self._pairs: IntArray = np.zeros((0, 2), dtype=np.int64)
        self._edges: IntArray = np.zeros((0, 2), dtype=np.int64)
        self._degree: FloatArray = np.zeros(0, dtype=np.float64)
        self._body: IntArray = np.zeros(0, dtype=np.int64)
        self._n_bodies = 0
        self._shadow = _Shadow(obstacles, self.flow, self._plane) if obstacles else None

    def __call__(self, mesh: Mesh) -> DragReport:
        n_faces = len(mesh.faces)
        if not n_faces:
            raise ValueError("cannot score an empty mesh")
        a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
        cross = np.cross(b - a, c - a)
        twice = np.linalg.norm(cross, axis=1)
        area = 0.5 * twice
        normal = cross / np.where(twice > 0, twice, 1.0)[:, None]
        centroid = (a + b + c) / 3.0
        if self.ends is not None:
            # The cut ends of a slice are not part of the real body.
            area = np.where(np.abs(normal[:, self.ends]) > 0.9, 0.0, area)
        # +1 for a face square into the wind, -1 for one facing straight downstream.
        s = -(normal @ self.flow)

        windward = s > 0
        leeward = ~windward
        depth = centroid @ self.flow
        sheltered = windward & self._sheltered(centroid, depth, s, area)
        # What the mesh presents to the flow, whether or not an obstacle's wake
        # then hides it: that is its own size, which wake strength depends on.
        seen = windward & ~sheltered
        exposed = seen.copy()
        if self._shadow is not None:
            exposed &= ~self._shadow.hides(centroid, depth)

        # Wake strength and what counts as "curved" both depend on the size of
        # the body a face belongs to -- not of the whole mesh, which may hold
        # several separate bodies, like the bars of a roof rack.
        _, _, _, body, n_bodies = self._topology(mesh.faces)
        frontal_each = np.bincount(
            body, weights=np.where(seen, area * s, 0.0), minlength=n_bodies
        )
        frontal = float(frontal_each.sum())
        if frontal <= 0:
            raise ValueError("the mesh presents no area to the flow; is it inside out?")
        front = np.full(n_bodies, np.inf)
        back = np.full(n_bodies, -np.inf)
        np.minimum.at(front, body, depth)
        np.maximum.at(back, body, depth)
        measured = self._aspect(mesh, np.where(seen, s, 0.0), body, n_bodies)
        # The silhouette's short side: the width of a bar, the diameter of a disc.
        thickness = np.sqrt(np.maximum(frontal_each, 1e-300) / measured)
        aspect = measured if self.aspect is None else np.full(n_bodies, self.aspect)
        # 0 for a compact body, 1 for an endless one. A two-dimensional wake is
        # stronger, and it stays strong behind a longer afterbody: rectangular
        # sections only lose most of their base suction at four or more
        # thicknesses deep, where a 3D body has lost it by one.
        flatness = 1.0 - np.exp(-(aspect - 1.0) / 47.0)
        decay = (1.5 - 1.0 * flatness) * (back - front) / thickness
        base_cp = (-(0.15 + 0.30 * np.exp(-decay)) * (1.0 + 1.6 * flatness))[body]
        scale = (2.0 * thickness / np.sqrt(np.pi))[body]
        convexity = self._convexity(mesh, normal, centroid, area, scale)

        # Angle at which a leeward face falls away from the flow: 0 is parallel
        # to it, 90 is a flat back.
        fall = np.degrees(np.arcsin(np.clip(-s, 0.0, 1.0)))
        separated = np.where(
            leeward, expit((fall - self.separation_angle) / self.separation_width), 0.0
        )

        squared = np.clip(s, 0.0, 1.0) ** 2
        # Convex: shoulder suction. Concave: air trapped at full stagnation,
        # Cp = 1 -- more than the 0.85 a flat face averages, which is relieved
        # by the air escaping round its edges. That is what makes a cup worse
        # than a plate, and a carved nose that dishes inward worse than a flat one.
        convex, concave = np.clip(convexity, 0.0, 1.0), np.clip(-convexity, 0.0, 1.0)
        flat = self.stagnation * (squared - convex * (1.0 - squared))
        cp = np.zeros(n_faces, dtype=np.float64)
        cp[exposed] = (flat + concave * (1.0 - flat))[exposed]
        cp[leeward] = (base_cp * separated)[leeward]

        # Shoulder suction is real only next to the stagnation pressure that
        # drives it. Never let a forebody pull itself forward on net.
        fore = np.where(windward, area * s * cp, 0.0)
        push = float(fore[fore > 0].sum())
        pull = float(-fore[fore < 0].sum())
        if pull > push:
            cp = np.where(windward & (cp < 0), cp * (push / pull), cp)
        pressure = area * s * cp

        attached = np.where(windward, exposed.astype(np.float64), 1.0 - separated)
        sliding = np.sqrt(np.clip(1.0 - s**2, 0.0, 1.0))
        friction = self.skin_friction * area * sliding * attached

        # The full force on each face, not just its drag: pressure pushes along
        # the inward normal, friction drags along the surface in the direction
        # the air slides over it. Its component along the flow is ``per_face``.
        along = self.flow[None, :] - (normal @ self.flow)[:, None] * normal
        along /= np.where(sliding > 1e-9, sliding, 1.0)[:, None]
        forces = -(cp * area)[:, None] * normal + friction[:, None] * along

        per_face = pressure + friction
        forebody = float(pressure[windward].sum())
        base = float(pressure[leeward].sum())
        rubbing = float(friction.sum())
        drag_area = forebody + base + rubbing
        return DragReport(
            drag_area=drag_area,
            cd=drag_area / frontal,
            frontal_area=frontal,
            forebody=forebody,
            base=base,
            friction=rubbing,
            per_face=per_face,
            forces=forces,
            flow=self.flow.copy(),
        )

    def _aspect(self, mesh: Mesh, s: FloatArray, body: IntArray, n_bodies: int) -> FloatArray:
        """Length over width of each body's silhouette, from its second moments.

        The silhouette is the exposed windward faces projected along the flow,
        and its moments are integrated exactly over each projected triangle
        -- not estimated from face centres, which a coarse mesh has too few
        of. For a rectangle ``w`` by ``t`` the variances along its axes are
        ``w^2 / 12`` and ``t^2 / 12``, so the square root of their ratio is
        exactly ``w / t``.
        """
        corners = (mesh.vertices @ self._plane.T)[mesh.faces]  # (faces, 3, 2)
        u, v = corners[..., 0], corners[..., 1]
        area = 0.5 * np.abs(
            (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])
        )
        weight = np.where(s > 0, area, 0.0)
        su, sv = u.sum(axis=1), v.sum(axis=1)

        def total(values: FloatArray) -> FloatArray:
            summed = np.bincount(body, weights=weight * values, minlength=n_bodies)
            result: FloatArray = summed.astype(np.float64, copy=False)
            return result

        mass = np.maximum(total(np.ones_like(su)), 1e-300)
        mu, mv = total(su / 3.0) / mass, total(sv / 3.0) / mass
        # The second moment of a triangle about the origin is A/12 times the
        # sum of its corners' outer products plus the outer product of their sum.
        a = total(((u**2).sum(axis=1) + su**2) / 12.0) / mass - mu**2
        b = total(((v**2).sum(axis=1) + sv**2) / 12.0) / mass - mv**2
        c = total(((u * v).sum(axis=1) + su * sv) / 12.0) / mass - mu * mv
        spread = np.sqrt(((a - b) / 2.0) ** 2 + c**2)
        major, minor = (a + b) / 2.0 + spread, (a + b) / 2.0 - spread
        result: FloatArray = np.sqrt(major / np.maximum(minor, major * 1e-6 + 1e-300))
        return np.maximum(result, 1.0)

    def _convexity(
        self,
        mesh: Mesh,
        normal: FloatArray,
        centroid: FloatArray,
        area: FloatArray,
        scale: FloatArray,
    ) -> FloatArray:
        """Per face, from -1 (concave) through 0 (flat) to +1 (convex).

        Measured as curvature -- total signed turning across edges, per unit
        area, over a few rings of neighbours -- relative to the body's own
        size ``scale``, per face. Curvature rather than the angle between two triangles,
        because the angle halves when the tessellation doubles and the shape
        has not changed. Over several rings rather than one, because a surface
        meshed from voxels is terraced: a convex dome becomes alternating
        convex and concave steps, and only the net turning over a patch wider
        than a step says which way it really bends.

        Anything curving at the scale of the body or tighter counts as fully
        convex; a surface ten times flatter than that counts as barely curved.
        """
        pairs, edges, degree, _, _ = self._topology(mesh.faces)
        n = len(mesh.faces)
        if not len(pairs):
            return np.zeros(n, dtype=np.float64)
        n1, n2 = normal[pairs[:, 0]], normal[pairs[:, 1]]
        angle = np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0))
        # Normals that diverge across an edge mean a ridge; that converge, a valley.
        sign = np.sign(
            np.einsum("ij,ij->i", n1 - n2, centroid[pairs[:, 0]] - centroid[pairs[:, 1]])
        )
        length = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        share = np.repeat(0.5 * sign * angle * length, 2)
        summed = np.bincount(pairs.ravel(), weights=share, minlength=n)
        turning: FloatArray = summed.astype(np.float64, copy=False)
        patch = area.copy()
        for _ in range(4):
            turning = _spread(turning, pairs, degree)
            patch = _spread(patch, pairs, degree)
        # Turning per area is the sum of the principal curvatures: 2/R on a
        # sphere, 1/R on a cylinder.
        curvature = turning / np.where(patch > 0, patch, 1.0)
        result: FloatArray = np.clip(curvature * scale / 2.0, -1.0, 1.0)
        return result

    def _topology(
        self, faces: IntArray
    ) -> tuple[IntArray, IntArray, FloatArray, IntArray, int]:
        """Face adjacency, and which separate body each face belongs to.

        Pairs of faces sharing an edge, that edge, each face's neighbour count,
        each face's body label, and the number of bodies. Cached per faces
        array, which a deformation lattice never changes.
        """
        if self._faces is not faces:
            edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            owner = np.tile(np.arange(len(faces)), 3)
            edges.sort(axis=1)
            key = edges[:, 0] * (int(faces.max()) + 1) + edges[:, 1]
            order = np.argsort(key, kind="stable")
            key, owner, edges = key[order], owner[order], edges[order]
            same = key[1:] == key[:-1]
            self._pairs = np.stack([owner[:-1][same], owner[1:][same]], axis=1)
            self._edges = edges[:-1][same]
            degree = np.bincount(self._pairs.ravel(), minlength=len(faces)).astype(np.float64)
            self._degree = np.maximum(degree, 1.0)
            n = len(faces)
            graph = sparse.coo_matrix(
                (np.ones(len(self._pairs)), (self._pairs[:, 0], self._pairs[:, 1])), shape=(n, n)
            )
            self._n_bodies, labels = csgraph.connected_components(graph, directed=False)
            self._body = np.asarray(labels, dtype=np.int64)
            self._faces = faces
        return self._pairs, self._edges, self._degree, self._body, self._n_bodies

    def _sheltered(
        self, centroid: FloatArray, depth: FloatArray, s: FloatArray, area: FloatArray
    ) -> np.ndarray:
        """Windward faces with a leeward face upstream of them along the flow."""
        positive = area[area > 0]
        if not len(positive):
            return np.zeros(len(s), dtype=bool)
        cell = 2.0 * float(np.sqrt(np.median(positive)))
        flat = centroid @ self._plane.T
        index = np.floor((flat - flat.min(axis=0)) / cell).astype(np.int64)
        key = index[:, 0] * (int(index[:, 1].max()) + 1) + index[:, 1]
        blockers = s < -0.05
        nearest = np.full(int(key.max()) + 1, np.inf)
        np.minimum.at(nearest, key[blockers], depth[blockers])
        sheltered: np.ndarray = depth > nearest[key] + cell
        return sheltered


class _Shadow:
    """Where the wake behind some fixed obstacles starts, mapped across the flow.

    The obstacles' leeward faces are drawn onto a grid in the plane normal to
    the flow, each grid point keeping the most upstream depth at which an
    obstacle's back hangs over it. Drawn rather than binned by face, because
    an obstacle is often a few big triangles -- a slab standing in for a car
    -- whose centres alone would mark a handful of columns out of thousands.
    """

    def __init__(
        self, obstacles: Sequence[Mesh], flow: FloatArray, plane: FloatArray, *, size: int = 256
    ) -> None:
        corners, depths = [], []
        for mesh in obstacles:
            a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
            normal = np.cross(b - a, c - a)
            leeward = normal @ flow > 0.05 * np.linalg.norm(normal, axis=1)
            faces = mesh.faces[leeward]
            corners.append((mesh.vertices @ plane.T)[faces])
            depths.append((mesh.vertices @ flow)[faces].min(axis=1))
        uv = np.concatenate(corners) if corners else np.zeros((0, 3, 2))
        depth = np.concatenate(depths) if depths else np.zeros(0)
        if not len(uv):
            raise ValueError("the obstacles have no faces turned away from the flow")
        self.plane = plane
        self.origin = uv.reshape(-1, 2).min(axis=0)
        extent = uv.reshape(-1, 2).max(axis=0) - self.origin
        self.pixel = max(float(extent.max()) / size, 1e-12)
        shape = (np.ceil(extent / self.pixel).astype(np.int64) + 1).tolist()
        self.depth: FloatArray = np.full(shape, np.inf)
        for triangle, d in zip(uv, depth, strict=True):
            self._draw((triangle - self.origin) / self.pixel, float(d))

    def _draw(self, t: FloatArray, depth: float) -> None:
        lo = np.maximum(np.floor(t.min(axis=0)).astype(np.int64), 0)
        hi = np.minimum(np.ceil(t.max(axis=0)).astype(np.int64), np.array(self.depth.shape) - 1)
        i, j = np.meshgrid(
            np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij"
        )
        p = np.stack([i.ravel(), j.ravel()], axis=1).astype(np.float64)
        # Barycentric coordinates; a sliver of a triangle at least marks the
        # grid point its centre falls nearest.
        e1, e2 = t[1] - t[0], t[2] - t[0]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        inside = np.zeros(len(p), dtype=bool)
        if abs(det) > 1e-12:
            d = p - t[0]
            u = (d[:, 0] * e2[1] - d[:, 1] * e2[0]) / det
            v = (e1[0] * d[:, 1] - e1[1] * d[:, 0]) / det
            inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        centre = np.rint(t.mean(axis=0)).astype(np.int64)
        inside |= (p[:, 0] == centre[0]) & (p[:, 1] == centre[1])
        rows, cols = i.ravel()[inside], j.ravel()[inside]
        self.depth[rows, cols] = np.minimum(self.depth[rows, cols], depth)

    def hides(self, centroid: FloatArray, depth: FloatArray) -> BoolArray:
        """Which faces lie downstream of an obstacle's back."""
        index = np.rint((centroid @ self.plane.T - self.origin) / self.pixel).astype(np.int64)
        on = np.all((index >= 0) & (index < np.array(self.depth.shape)), axis=1)
        behind = np.full(len(depth), np.inf)
        behind[on] = self.depth[index[on, 0], index[on, 1]]
        result: BoolArray = depth > behind
        return result


def _spread(values: FloatArray, pairs: IntArray, degree: FloatArray) -> FloatArray:
    """Average each face's value with its edge neighbours'. One ring of smoothing."""
    n = len(values)
    total = values + np.bincount(pairs[:, 0], weights=values[pairs[:, 1]], minlength=n)
    total += np.bincount(pairs[:, 1], weights=values[pairs[:, 0]], minlength=n)
    result: FloatArray = total / (1.0 + degree)
    return result
