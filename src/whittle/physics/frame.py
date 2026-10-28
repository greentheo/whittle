"""Beams and frames: tube structures the voxel grid is too coarse to see.

A 2" square tube with a 1/8" wall is 3 mm of steel on a frame nearly two
metres across. At 48 voxels along that frame a voxel is 40 mm, so the wall
is not there at all. What carries the load in a frame is the bending and
twisting of each member as a whole, and that is what a frame model does:
each member is a beam along its centreline, with the area and stiffness of
its cross-section, joined rigidly to the others at the nodes.

Three-dimensional Timoshenko beams -- bending plus the shear that matters
in a deep or foam-cored member, and barely at all in a tube -- twelve
degrees of freedom each, solved directly -- a frame of a few dozen members is a few hundred unknowns
and a millisecond. Stress is checked at the critical corner of each section,
all along each member, and compression members against Euler buckling.

What it does not see: the welds. Joints are taken as perfectly rigid and
stress as the nominal stress in the tube, where a real welded joint
concentrates it at the toe of the weld. Under a static load that matters
little. Under road vibration it is how steel frames fail -- by fatigue
cracks starting at welds -- which is why a frame study flags any load case
that leaves less than about 3.5x.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..types import BoolArray, FloatArray, IntArray

INCH = 25.4
GRAVITY = 9.81

#: Structural tube is sold by nominal wall but may be made thinner: AISC takes
#: the design wall of welded tube as 0.93 of nominal for strength and
#: stiffness, and so does this. Mills are held to its nominal weight, though,
#: so a tube is weighed at its nominal wall.
DESIGN_WALL = 0.93


@dataclass(frozen=True)
class Section:
    """A cross-section, and everything about it a beam needs. Millimetres throughout.

    ``width`` lies along the member's local y axis and ``height`` along its
    local z, which for a horizontal member is vertical.
    """

    name: str
    width: float
    height: float
    #: The wall used for the properties below; ``None`` for a solid bar.
    wall: float | None
    area: float
    #: The area that is weighed: at the nominal wall, as tube is sold, where
    #: ``area`` and everything below are at the design wall.
    mass_area: float
    #: Second moments: ``iy`` for bending about local y (sagging in z), ``iz`` about local z.
    iy: float
    iz: float
    #: Torsion constant.
    j: float
    #: Section moduli, bending stress per moment.
    sy: float
    sz: float
    #: Torsion modulus: shear stress is torque over this.
    wt: float
    #: Areas that carry shear along local y and z.
    shear_y: float
    shear_z: float

    @property
    def solid(self) -> bool:
        return self.wall is None

    def mass_per_mm(self, density: float) -> float:
        """kg per mm of length, for a density in g/cm^3."""
        return self.mass_area * density * 1e-6


def tube(
    width: float,
    height: float | None = None,
    wall: float | None = None,
    *,
    name: str | None = None,
    design_wall: float = DESIGN_WALL,
) -> Section:
    """A rectangular tube from its outer size and nominal wall, or a solid bar if ``wall`` is None.

    Tube corners are rounded, outside radius twice the wall, as welded
    structural tube is; the wall used is ``design_wall`` of the nominal one,
    except for weighing it.
    """
    height = width if height is None else height
    if width <= 0 or height <= 0:
        raise ValueError(f"a section needs a positive size, got {width} x {height}")
    label = name or f"{width:g} x {height:g} mm" + ("" if wall is None else f" x {wall:g}")
    if wall is None or 2.0 * wall >= min(width, height):
        return _solid(label, width, height)
    if wall <= 0:
        raise ValueError(f"a tube needs a positive wall, got {wall}")
    t = wall * design_wall
    outer = _rounded(width, height, 2.0 * t)
    inner = _rounded(width - 2.0 * t, height - 2.0 * t, t)
    area, iy, iz = (a - b for a, b in zip(_moments(outer), _moments(inner), strict=True))
    sold = _rounded(width, height, 2.0 * wall)
    hollow = _rounded(width - 2.0 * wall, height - 2.0 * wall, wall)
    # Bredt's thin-walled formula, on the wall's mid-line.
    middle = _rounded(width - t, height - t, 1.5 * t)
    enclosed = _moments(middle)[0]
    perimeter = float(np.linalg.norm(np.diff(middle, axis=0, append=middle[:1]), axis=1).sum())
    return Section(
        name=label,
        width=width,
        height=height,
        wall=t,
        area=area,
        mass_area=_moments(sold)[0] - _moments(hollow)[0],
        iy=iy,
        iz=iz,
        j=4.0 * enclosed**2 * t / perimeter,
        sy=iy / (height / 2.0),
        sz=iz / (width / 2.0),
        wt=2.0 * enclosed * t,
        shear_y=2.0 * (width - 2.0 * t) * t,
        shear_z=2.0 * (height - 2.0 * t) * t,
    )


def _solid(label: str, width: float, height: float) -> Section:
    a, b = max(width, height), min(width, height)
    # Roark's approximations for a solid rectangle in torsion.
    j = a * b**3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b**4 / (12.0 * a**4)))
    wt = a * b**2 / (3.0 + 1.8 * b / a)
    area = width * height
    return Section(
        name=label,
        width=width,
        height=height,
        wall=None,
        area=area,
        mass_area=area,
        iy=width * height**3 / 12.0,
        iz=height * width**3 / 12.0,
        j=j,
        sy=width * height**2 / 6.0,
        sz=height * width**2 / 6.0,
        wt=wt,
        shear_y=area / 1.5,
        shear_z=area / 1.5,
    )


def _rounded(width: float, height: float, radius: float, per_corner: int = 12) -> FloatArray:
    """A rounded rectangle's outline, counter-clockwise, as a polygon."""
    r = min(radius, width / 2.0, height / 2.0)
    cy, cz = width / 2.0 - r, height / 2.0 - r
    points = []
    for sy, sz, start in ((1, 1, 0.0), (-1, 1, 0.5), (-1, -1, 1.0), (1, -1, 1.5)):
        angle = np.pi * (start + 0.5 * np.linspace(0.0, 1.0, per_corner))
        points.append(np.column_stack([sy * cy + r * np.cos(angle), sz * cz + r * np.sin(angle)]))
    return np.concatenate(points)


def _moments(polygon: FloatArray) -> tuple[float, float, float]:
    """Area and second moments (about the y and z axes) of a polygon centred on the origin."""
    y, z = polygon[:, 0], polygon[:, 1]
    y1, z1 = np.roll(y, -1), np.roll(z, -1)
    cross = y * z1 - y1 * z
    area = cross.sum() / 2.0
    iy = (cross * (z**2 + z * z1 + z1**2)).sum() / 12.0
    iz = (cross * (y**2 + y * y1 + y1**2)).sum() / 12.0
    return float(area), float(iy), float(iz)


_KEY = re.compile(r"^\s*([\d./]+)\s*x\s*([\d./]+)\s*x\s*([\d./]+|solid)\s*$", re.IGNORECASE)


def tube_dimensions(key: str) -> tuple[float, float, float | None]:
    """Width, height and nominal wall in mm from a key in inches: ``"2x2x1/8"``, ``"2x3x0.083"``.

    ``"2x2xsolid"`` is a solid bar, and its wall is None.
    """
    match = _KEY.match(key)
    if not match:
        raise ValueError(f"tube sizes look like '2x2x1/8' or '2x2xsolid' (inches), got {key!r}")
    width, height = (float(Fraction(part)) * INCH for part in match.groups()[:2])
    wall = None if match[3].lower() == "solid" else float(Fraction(match[3])) * INCH
    return width, height, wall


def tube_size(key: str) -> tuple[float, float]:
    """Outer size and nominal wall in mm of a square tube key, as a model would draw it.

    A solid bar's wall comes back as half its size -- a tube with no hole.
    """
    width, height, wall = tube_dimensions(key)
    if abs(width - height) > 1e-9:
        raise ValueError(f"{key!r} is not square; use tube_dimensions for its width and height")
    return width, width / 2.0 if wall is None else wall


@cache
def section(key: str) -> Section:
    """The section for a tube key in inches, as :func:`tube_dimensions` reads it."""
    width, height, wall = tube_dimensions(key)
    return tube(width, height, wall, name=key.strip())


#: Square steel tube and bar a hobby fabricator can buy, by nominal size in
#: inches. A 2" hitch shank has to be one of the 2x2 sizes to fit the receiver.
TUBES: dict[str, Section] = {
    key: section(key)
    for key in (
        "1x1x1/16",
        "1x1x0.083",
        "1x1x1/8",
        "1.25x1.25x0.083",
        "1.25x1.25x1/8",
        "1.5x1.5x0.083",
        "1.5x1.5x1/8",
        "1.5x1.5x3/16",
        "1.75x1.75x1/8",
        "2x2x0.083",
        "2x2x1/8",
        "2x2x3/16",
        "2x2x1/4",
        "2x2x5/16",
        "2x2xsolid",
        "2.5x2.5x1/8",
        "2.5x2.5x3/16",
        "3x3x1/8",
        "3x3x3/16",
    )
}

#: Degrees of freedom at each node, in order.
DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")

#: One load case for :meth:`FrameModel.solve`: nodal, distributed, and point loads.
Loads = tuple[FloatArray, FloatArray, tuple[IntArray, FloatArray, FloatArray]]


@dataclass
class CaseResult:
    """One load case: how the frame moves, what each member carries, what holds it."""

    name: str
    #: Per node: three displacements in mm, three rotations in radians.
    displacements: FloatArray
    #: Per node, what the supports push back with: three forces in N, three
    #: moments in N mm. Zero at nodes that are not supported.
    reactions: FloatArray
    #: Per member: the forces and moments on each end, in the member's own
    #: axes -- ``(members, 12)``, the near end's six then the far end's.
    end_forces: FloatArray
    #: Per member, the worst von Mises stress anywhere along it, MPa.
    stress: FloatArray
    #: Per member, the worst axial compression along it, N (0 if it is in tension).
    compression: FloatArray
    #: Per member, the Euler buckling load, N.
    buckling: FloatArray
    #: Per member: the lesser of strength over stress and buckling load over compression.
    safety: FloatArray
    #: Node positions, for :meth:`resultant`.
    nodes: FloatArray

    @property
    def deflection(self) -> FloatArray:
        """How far each node moves, mm."""
        result: FloatArray = np.linalg.norm(self.displacements[:, :3], axis=1)
        return result

    def resultant(self, about: Sequence[float]) -> tuple[FloatArray, FloatArray]:
        """The supports' total force (N) and moment (N mm) about a point.

        Whatever the supports are, this is what the structure they hold
        pushes on in return, reversed: at a hitch receiver's mouth, the
        force and moment to compare with the hitch's rating.
        """
        arm = self.nodes - np.asarray(about, dtype=np.float64)
        force = self.reactions[:, :3].sum(axis=0)
        moment = self.reactions[:, 3:].sum(axis=0) + np.cross(arm, self.reactions[:, :3]).sum(
            axis=0
        )
        return force, moment


class FrameModel:
    """A frame's geometry and supports, solved for any choice of sections and any loads.

    ``members`` pairs node indices. Each load case is ``nodal`` forces and
    moments, ``(nodes, 6)``; ``distributed`` loads along members in global
    directions, ``(members, 3)`` in N per mm; and ``points``, forces at
    points along members: member indices ``(k,)``, how far along each is as a
    fraction from its first node ``(k,)``, and global forces ``(k, 3)``.

    The material -- ``youngs_modulus``, ``shear_modulus`` and ``strength``,
    in MPa -- is one value for every member, or one per member.
    """

    def __init__(
        self,
        nodes: FloatArray,
        members: IntArray,
        fixed: BoolArray,
        *,
        youngs_modulus: float | FloatArray,
        shear_modulus: float | FloatArray,
        strength: float | FloatArray,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.members = np.asarray(members, dtype=np.int64).reshape(-1, 2)
        self.fixed = np.asarray(fixed, dtype=bool).reshape(len(self.nodes), 6)
        m = len(self.members)
        self.e, self.g, self.strength = (
            np.broadcast_to(np.asarray(value, dtype=np.float64), (m,)).copy()
            for value in (youngs_modulus, shear_modulus, strength)
        )
        if np.any(self.e <= 0) or np.any(self.g <= 0):
            raise ValueError("every member needs a positive Young's modulus and shear modulus")
        span = self.nodes[self.members[:, 1]] - self.nodes[self.members[:, 0]]
        self.length = np.linalg.norm(span, axis=1)
        if np.any(self.length <= 0):
            raise ValueError("a member joins a node to itself, or two nodes at the same place")
        self.rotation = _axes(span / self.length[:, None])
        dofs = 6 * self.members[:, :, None] + np.arange(6)[None, None, :]
        self.dofs = dofs.reshape(-1, 12)
        self.free = ~self.fixed.ravel()

    def solve(
        self, sections: Sequence[Section], cases: Mapping[str, Loads]
    ) -> dict[str, CaseResult]:
        """Displacements, member forces, stresses and reactions in every case."""
        m = len(self.members)
        if len(sections) != m:
            raise ValueError(f"need one section per member: {m}, got {len(sections)}")
        phi_y, phi_z = self._shear(sections)
        local = self._local_stiffness(sections, phi_y, phi_z)
        t = np.zeros((m, 12, 12))
        for block in range(4):
            t[:, 3 * block : 3 * block + 3, 3 * block : 3 * block + 3] = self.rotation
        k_global = np.einsum("mji,mjk,mkl->mil", t, local, t)
        n = 6 * len(self.nodes)
        rows = np.repeat(self.dofs, 12, axis=1).ravel()
        cols = np.tile(self.dofs, (1, 12)).ravel()
        stiffness = sparse.csc_matrix((k_global.ravel(), (rows, cols)), shape=(n, n))
        free = np.flatnonzero(self.free)
        try:
            # Every node held is a frame that does not move: loads along members only.
            factor = splu(stiffness[free][:, free].tocsc()) if len(free) else None
        except RuntimeError as error:
            raise ValueError(
                "the frame can move without bending anything -- a node joined by no "
                "member, or not enough supports to hold it still"
            ) from error

        results = {}
        for name, (nodal, distributed, (which, where, push)) in cases.items():
            q_local = np.einsum("mij,mj->mi", self.rotation, distributed)
            equivalent = _equivalent_loads(q_local, self.length)
            which = np.asarray(which, dtype=np.int64)
            where = np.asarray(where, dtype=np.float64) * self.length[which]
            p_local = np.einsum("kij,kj->ki", self.rotation[which], np.reshape(push, (-1, 3)))
            np.add.at(
                equivalent,
                which,
                _point_loads(p_local, where, self.length[which], phi_y[which], phi_z[which]),
            )
            loads = np.asarray(nodal, dtype=np.float64).reshape(n).copy()
            np.add.at(loads, self.dofs, np.einsum("mji,mj->mi", t, equivalent))
            d = np.zeros(n)
            if factor is not None:
                d[free] = factor.solve(loads[free])
            if not np.all(np.isfinite(d)):
                raise ValueError("the frame's stiffness is singular: check its supports")
            reactions = stiffness @ d - loads
            reactions[free] = 0.0
            ends = np.einsum("mij,mjk,mk->mi", local, t, d[self.dofs]) - equivalent
            stress, compression = _along(
                ends, q_local, self.length, sections, (which, where, p_local)
            )
            buckling = self._buckling(sections)
            with np.errstate(divide="ignore"):
                safety = np.minimum(
                    self.strength / np.maximum(stress, 1e-12),
                    np.where(compression > 0, buckling / np.maximum(compression, 1e-12), np.inf),
                )
            results[name] = CaseResult(
                name=name,
                displacements=d.reshape(-1, 6),
                reactions=reactions.reshape(-1, 6),
                end_forces=ends,
                stress=stress,
                compression=compression,
                buckling=buckling,
                safety=safety,
                nodes=self.nodes,
            )
        return results

    def _shear(self, sections: Sequence[Section]) -> tuple[FloatArray, FloatArray]:
        """How much shear adds to bending deflection, per member -- Timoshenko's phi --
        for bending in the local xy plane (about z) and in xz (about y)."""
        iy, iz, shear_y, shear_z = _properties(sections, "iy", "iz", "shear_y", "shear_z")
        scale = 12.0 * self.e / (self.g * self.length**2)
        return scale * iz / shear_y, scale * iy / shear_z

    def _buckling(self, sections: Sequence[Section]) -> FloatArray:
        """Euler's buckling load, eased by shear, each way; the lesser of the two."""
        iy, iz, shear_y, shear_z = _properties(sections, "iy", "iz", "shear_y", "shear_z")
        euler = np.pi**2 * self.e / self.length**2
        about_y, about_z = euler * iy, euler * iz
        result: FloatArray = np.minimum(
            about_y / (1.0 + about_y / (self.g * shear_z)),
            about_z / (1.0 + about_z / (self.g * shear_y)),
        )
        return result

    def _local_stiffness(
        self, sections: Sequence[Section], phi_y: FloatArray, phi_z: FloatArray
    ) -> FloatArray:
        m = len(sections)
        e, g, length = self.e, self.g, self.length
        area = np.array([s.area for s in sections])
        iy = np.array([s.iy for s in sections])
        iz = np.array([s.iz for s in sections])
        j = np.array([s.j for s in sections])
        k = np.zeros((m, 12, 12))
        axial, torsion = e * area / length, g * j / length
        for a, b in ((0, 6), (3, 9)):
            value = axial if a == 0 else torsion
            k[:, a, a] = k[:, b, b] = value
            k[:, a, b] = k[:, b, a] = -value
        # Bending in the xy plane (uy, rz) about z, and in xz (uz, ry) about y.
        for inertia, phi, v, r, sign in ((iz, phi_y, 1, 5, 1.0), (iy, phi_z, 2, 4, -1.0)):
            scale = e * inertia / (1.0 + phi)
            a = 12.0 * scale / length**3
            b = sign * 6.0 * scale / length**2
            c = (4.0 + phi) * scale / length
            d = (2.0 - phi) * scale / length
            entries = {
                (v, v): a, (v, r): b, (v, v + 6): -a, (v, r + 6): b,
                (r, r): c, (r, v + 6): -b, (r, r + 6): d,
                (v + 6, v + 6): a, (v + 6, r + 6): -b,
                (r + 6, r + 6): c,
            }  # fmt: skip
            for (p, q), value in entries.items():
                k[:, p, q] = value
                k[:, q, p] = value
        return k


def _properties(sections: Sequence[Section], *keys: str) -> list[FloatArray]:
    """One array per property named, across the members' sections."""
    return [np.array([getattr(s, key) for s in sections]) for key in keys]


def _axes(direction: FloatArray) -> FloatArray:
    """Each member's local axes as the rows of a rotation: x along it, z as near up as it can be."""
    up = np.array([0.0, 0.0, 1.0])
    reference = np.where(
        (np.abs(direction @ up) > 0.999)[:, None], np.array([1.0, 0.0, 0.0]), up
    )
    y = np.cross(reference, direction)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(direction, y)
    result: FloatArray = np.stack([direction, y, z], axis=1)
    return result


def _equivalent_loads(q: FloatArray, length: FloatArray) -> FloatArray:
    """The nodal loads equivalent to a uniform load along each member, in local axes."""
    half = q * length[:, None] / 2.0
    moment = length**2 / 12.0
    f = np.zeros((len(q), 12))
    f[:, 0:3] = half
    f[:, 6:9] = half
    f[:, 5] = q[:, 1] * moment
    f[:, 11] = -q[:, 1] * moment
    f[:, 4] = -q[:, 2] * moment
    f[:, 10] = q[:, 2] * moment
    return f


def _point_loads(
    p: FloatArray, a: FloatArray, length: FloatArray, phi_y: FloatArray, phi_z: FloatArray
) -> FloatArray:
    """The nodal loads equivalent to a force ``a`` mm along a member, in local axes.

    A fixed-ended Timoshenko beam's end forces: with no shear (phi 0) the
    familiar Pb^2(3a+b)/L^3 and Pab^2/L^2; the more shear, the more each end
    takes the share a simple lever would give it.
    """
    b = length - a
    f = np.zeros((len(p), 12))
    f[:, 0] = p[:, 0] * b / length
    f[:, 6] = p[:, 0] * a / length
    for phi, axis, spin, sign in ((phi_y, 1, 5, 1.0), (phi_z, 2, 4, -1.0)):
        share = 1.0 / (length**2 * (1.0 + phi))
        f[:, axis] = p[:, axis] * b * (b * (3.0 * a + b) + phi * length**2) * share / length
        f[:, axis + 6] = p[:, axis] * a * (a * (a + 3.0 * b) + phi * length**2) * share / length
        f[:, spin] = sign * p[:, axis] * a * b * (b + phi * length / 2.0) * share
        f[:, spin + 6] = -sign * p[:, axis] * a * b * (a + phi * length / 2.0) * share
    return f


def _along(
    ends: FloatArray,
    q: FloatArray,
    length: FloatArray,
    sections: Sequence[Section],
    points: tuple[IntArray, FloatArray, FloatArray],
    samples: int = 11,
) -> tuple[FloatArray, FloatArray]:
    """Worst von Mises stress and worst compression along each member.

    Cut each member at a point, and the near end's forces plus the load up
    to the cut are what the rest of it holds. Stresses from axial force and
    bending in both directions add at the worst corner; shear from torsion
    and transverse force adds too -- conservative, which is the point.

    A force at a point along a member puts the most bending under itself, so
    each member is also cut either side of every one of its ``points``
    (member indices, mm along, local forces).
    """
    which, where, push = points
    m = len(length)
    count = np.bincount(which, minlength=m)
    extra = 2 * int(count.max()) if len(which) else 0
    s = np.zeros((m, samples + extra))
    s[:, :samples] = np.linspace(0.0, 1.0, samples)[None, :] * length[:, None]
    # Each point's own two slots: its rank among its member's points.
    order = np.argsort(which, kind="stable")
    rank = np.empty(len(which), dtype=np.int64)
    rank[order] = np.arange(len(which)) - np.concatenate([[0], np.cumsum(count)])[which[order]]
    slot = samples + 2 * rank
    s[which, slot] = where
    s[which, slot + 1] = np.minimum(where * (1.0 + 1e-9) + 1e-9, length[which])
    s = s[:, :, None]
    near_force, near_moment = ends[:, None, 0:3], ends[:, None, 3:6]
    force = -near_force - q[:, None, :] * s
    ex = np.array([1.0, 0.0, 0.0])
    moment = (
        -near_moment
        + s * np.cross(ex, near_force)
        + (s**2 / 2.0) * np.cross(ex, q[:, None, :])
    )
    # Past a point load, the cut also holds it.
    lever = s[which, :, 0] - where[:, None]
    past = lever > 0.0
    np.add.at(force, which, np.where(past[:, :, None], -push[:, None, :], 0.0))
    np.add.at(
        moment, which, (np.where(past, lever, 0.0))[:, :, None] * np.cross(ex, push)[:, None, :]
    )
    area = np.array([sec.area for sec in sections])[:, None]
    sy = np.array([sec.sy for sec in sections])[:, None]
    sz = np.array([sec.sz for sec in sections])[:, None]
    wt = np.array([sec.wt for sec in sections])[:, None]
    shear_y = np.array([sec.shear_y for sec in sections])[:, None]
    shear_z = np.array([sec.shear_z for sec in sections])[:, None]
    normal = (
        np.abs(force[..., 0]) / area
        + np.abs(moment[..., 1]) / sy
        + np.abs(moment[..., 2]) / sz
    )
    shear = (
        np.abs(moment[..., 0]) / wt
        + np.abs(force[..., 1]) / shear_y
        + np.abs(force[..., 2]) / shear_z
    )
    stress = np.sqrt(normal**2 + 3.0 * shear**2).max(axis=1)
    compression = np.maximum(-force[..., 0], 0.0).max(axis=1)
    return stress, compression
