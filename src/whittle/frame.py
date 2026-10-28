"""Frame studies: size a welded tube structure, lightest tubes first that still hold.

::

    frame = FrameStudy(material="steel")
    mouth, inner = frame.node((76.2, 0, 228.6)), frame.node((228.6, 0, 228.6))
    tip = frame.node((-1000.0, 0, 228.6))
    frame.member(inner, mouth, "2x2x1/4", group="shank")
    frame.member(mouth, tip, "2x2x1/4", group="shank")
    frame.support(mouth, fix=("ux", "uy", "uz", "rx"))   # a receiver: a socket and a pin
    frame.support(inner, fix=("uy", "uz"))
    frame.case("driving", factor=2.5)
    frame.load(tip, force=(0, 0, -445), case="driving")
    result = frame.analyze()
    sweep = frame.sweep(groups={"shank": ["2x2x1/4", "2x2x3/16", "2x2xsolid"]})

Millimetres and newtons. Members are tubes from :data:`TUBES`, by key, or any
:class:`~whittle.physics.frame.Section`. ``group`` ties members that are cut
from the same stock -- both ladder rails, all the cross members -- so a sweep
sizes them together. A member can be made of something other than the
frame's own material -- a foam-cored wall carrying load beside the steel --
with a section and material of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from .physics.fea import Material, material
from .physics.frame import (
    DOFS,
    GRAVITY,
    TUBES,
    CaseResult,
    FrameModel,
    Loads,
    Section,
    _axes,
    section,
)
from .plate import Placement, PlateStudy, Relief
from .sweep import Column, Station, Sweep
from .types import FloatArray, Mesh, Vector3

#: Below this safety factor on a load case, a welded steel frame under road
#: vibration is in the range where fatigue cracks start at the welds.
FATIGUE_SAFETY = 3.5

_FIXES = {"all": DOFS, "pin": DOFS[:3]}

#: A bar's six faces, as quads of its eight corners: the near end's four, then the far end's.
_BOX = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (3, 2, 1, 0), (4, 5, 6, 7))


@dataclass(frozen=True)
class _Stuff:
    """What a member is made of, as a frame needs it: moduli and strength in MPa, g/cm^3."""

    youngs_modulus: float
    shear_modulus: float
    strength: float
    density: float


@dataclass
class _Case:
    factor: float = 1.0
    self_weight: bool = True
    nodal: list[tuple[int, Vector3, Vector3]] = field(default_factory=list)
    distributed: list[tuple[int, FloatArray]] = field(default_factory=list)
    points: list[tuple[int, float, Vector3]] = field(default_factory=list)


@dataclass
class FrameResult:
    """A frame with one choice of sections, under every load case."""

    cases: dict[str, CaseResult]
    sections: list[Section]
    #: kg, every member at its full centreline length.
    mass: float
    #: The least safety factor of any member in any case, and where.
    safety: float
    governing: tuple[str, int]
    #: The furthest any node moves in any case, mm, and which node.
    deflection: float
    deflects_most: tuple[str, int]
    #: Warnings worth reading before building it.
    notes: list[str]

    @property
    def stress(self) -> float:
        """The worst von Mises stress anywhere, MPa."""
        return float(max(case.stress.max() for case in self.cases.values()))


class FrameStudy:
    """Members along centrelines, supports, load cases -- and the lightest tubes that hold."""

    def __init__(
        self, material: str | Material = "steel", *, fatigue_safety: float = FATIGUE_SAFETY
    ) -> None:
        self.material = _material(material)
        self.fatigue_safety = float(fatigue_safety)
        self.stuff = _stuff(self.material)
        self.nodes: list[Vector3] = []
        self.names: dict[str, int] = {}
        self.members: list[tuple[int, int]] = []
        self.sections: list[Section] = []
        self.groups: list[str | None] = []
        #: Per member, what it is made of if not the frame's own material.
        self.materials: list[_Stuff | None] = []
        self.fixed: dict[int, set[str]] = {}
        self.cases: dict[str, _Case] = {}

    # -- building it up ------------------------------------------------------

    def node(self, xyz: Sequence[float], *, name: str | None = None) -> int:
        """Add a node; returns its index. A ``name`` lets members refer to it by name."""
        point = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self.nodes.append(point)
        index = len(self.nodes) - 1
        if name is not None:
            if name in self.names:
                raise ValueError(f"there is already a node called {name!r}")
            self.names[name] = index
        return index

    def member(
        self,
        a: int | str,
        b: int | str,
        section: str | Section | Mapping[str, Any],
        *,
        group: str | None = None,
        material: str | Material | Mapping[str, Any] | None = None,
    ) -> int:
        """Join two nodes with a member of this section; returns its index.

        ``section`` is a tube key, a :class:`Section`, or a section's numbers
        as a dict (``area``, ``iy``, ``iz``, ``j``, ``sy``, ``sz``, ``wt``,
        ``shear_y``, ``shear_z``; ``mass_area``, ``width`` and ``height`` if
        they matter). ``material`` is a name, a :class:`Material`, or a dict
        of ``youngs_modulus``, ``strength``, ``density`` and
        ``shear_modulus`` or ``poisson_ratio`` -- for a composite, its
        section transformed to one material and that material's strength.
        """
        self.members.append((self._node(a), self._node(b)))
        self.sections.append(_section(section))
        self.groups.append(group)
        self.materials.append(None if material is None else _stuff(material))
        return len(self.members) - 1

    def support(self, node: int | str, *, fix: str | Sequence[str] = "all") -> FrameStudy:
        """Hold a node. ``fix`` is ``"all"``, ``"pin"`` (no translation), or names from
        ``ux uy uz rx ry rz``."""
        names = _FIXES[fix] if isinstance(fix, str) and fix in _FIXES else fix
        names = (names,) if isinstance(names, str) else tuple(names)
        unknown = set(names) - set(DOFS)
        if unknown:
            raise ValueError(f"fix names look like {DOFS}, got {sorted(unknown)}")
        self.fixed.setdefault(self._node(node), set()).update(names)
        return self

    def case(self, name: str, *, factor: float = 1.0, self_weight: bool = True) -> FrameStudy:
        """Declare a load case. ``factor`` scales everything in it, the frame's own weight too."""
        existing = self.cases.get(name)
        self.cases[name] = _Case(
            factor=float(factor),
            self_weight=self_weight,
            nodal=[] if existing is None else existing.nodal,
            distributed=[] if existing is None else existing.distributed,
            points=[] if existing is None else existing.points,
        )
        return self

    def load(
        self,
        node: int | str,
        *,
        force: Sequence[float] = (0.0, 0.0, 0.0),
        moment: Sequence[float] = (0.0, 0.0, 0.0),
        case: str = "default",
    ) -> FrameStudy:
        """A force (N) and moment (N mm) on a node, in global directions."""
        f = (float(force[0]), float(force[1]), float(force[2]))
        mo = (float(moment[0]), float(moment[1]), float(moment[2]))
        self._case(case).nodal.append((self._node(node), f, mo))
        return self

    def spread(
        self,
        member: int,
        *,
        total: Sequence[float] | None = None,
        per_mm: Sequence[float] | None = None,
        case: str = "default",
    ) -> FrameStudy:
        """A load spread evenly along a member, in global directions: ``total`` N, or ``per_mm``."""
        if (total is None) == (per_mm is None):
            raise ValueError("give exactly one of total= or per_mm=")
        if not 0 <= member < len(self.members):
            raise ValueError(f"no member {member}")
        if total is not None:
            q = np.asarray(total, dtype=np.float64) / self._length(member)
        else:
            q = np.asarray(per_mm, dtype=np.float64)
        self._case(case).distributed.append((member, q))
        return self

    def point(
        self, member: int, at: float, *, force: Sequence[float], case: str = "default"
    ) -> FrameStudy:
        """A force (N, global directions) at a point along a member: ``at`` is how far
        along it is, from 0 at its first node to 1 at its second."""
        if not 0 <= member < len(self.members):
            raise ValueError(f"no member {member}")
        if not 0.0 <= at <= 1.0:
            raise ValueError(f"at= runs from 0 at a member's first node to 1, got {at}")
        f = (float(force[0]), float(force[1]), float(force[2]))
        self._case(case).points.append((member, float(at), f))
        return self

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> FrameStudy:
        """A frame from plain data, as another program would hand it over.

        ::

            {"material": "steel",
             "nodes": {"mouth": [76.2, 0, 228.6], ...},     # or a list: index = id
             "members": [{"a": "mouth", "b": "jog", "section": "2x2x1/4", "group": "shank"},
                         {"a": 3, "b": 4, "section": {"area": ..., "iy": ...},
                          "material": {"youngs_modulus": ..., "shear_modulus": ...}}],
             "supports": [{"node": "mouth", "fix": ["ux", "uy", "uz", "rx"]}],
             "cases": {"driving": {"factor": 2.5, "self_weight": true}},
             "loads": [{"case": "driving", "node": "tip", "force": [0, 0, -445]},
                       {"case": "driving", "member": 7, "total": [0, 0, -222]},
                       {"case": "driving", "member": 9, "at": 0.4, "force": [0, 0, -90]}]}
        """
        study = cls(spec.get("material", "steel"))
        nodes = spec["nodes"]
        if isinstance(nodes, Mapping):
            for name, xyz in nodes.items():
                study.node(xyz, name=str(name))
        else:
            for xyz in nodes:
                study.node(xyz)
        for item in spec["members"]:
            study.member(
                item["a"],
                item["b"],
                item["section"],
                group=item.get("group"),
                material=item.get("material"),
            )
        for item in spec.get("supports", []):
            study.support(item["node"], fix=item.get("fix", "all"))
        for name, case in spec.get("cases", {}).items():
            study.case(
                name, factor=case.get("factor", 1.0), self_weight=case.get("self_weight", True)
            )
        for item in spec.get("loads", []):
            case_name = item.get("case", "default")
            if "member" in item and "at" in item:
                study.point(
                    int(item["member"]), float(item["at"]), force=item["force"], case=case_name
                )
            elif "member" in item:
                study.spread(
                    int(item["member"]),
                    total=item.get("total"),
                    per_mm=item.get("per_mm"),
                    case=case_name,
                )
            else:
                study.load(
                    item["node"],
                    force=item.get("force", (0.0, 0.0, 0.0)),
                    moment=item.get("moment", (0.0, 0.0, 0.0)),
                    case=case_name,
                )
        return study

    # -- solving it ------------------------------------------------------------

    def analyze(self, sections: Sequence[Section] | None = None) -> FrameResult:
        """Solve every load case, with the members' own sections or these instead."""
        chosen = list(self.sections if sections is None else sections)
        model = self._model()
        cases = model.solve(chosen, self._loads(chosen, model))
        mass = float(sum(
            s.mass_per_mm(self._made_of(i).density) * length
            for i, (s, length) in enumerate(zip(chosen, model.length, strict=True))
        ))  # fmt: skip
        worst = min(
            ((name, int(np.argmin(c.safety)), float(c.safety.min())) for name, c in cases.items()),
            key=lambda item: item[2],
        )
        furthest = max(
            ((name, int(np.argmax(c.deflection)), float(c.deflection.max()))
             for name, c in cases.items()),
            key=lambda item: item[2],
        )  # fmt: skip
        notes = []
        for name, c in cases.items():
            if c.safety.min() < self.fatigue_safety:
                notes.append(
                    f"{name}: {c.safety.min():.1f}x at member {int(np.argmin(c.safety))} is under "
                    f"{self.fatigue_safety:g}x -- welded steel shaken on the road can crack at "
                    "its welds long before it would ever bend"
                )
        return FrameResult(
            cases=cases,
            sections=chosen,
            mass=mass,
            safety=worst[2],
            governing=(worst[0], worst[1]),
            deflection=furthest[2],
            deflects_most=(furthest[0], furthest[1]),
            notes=notes,
        )

    def sweep(
        self,
        *,
        groups: Mapping[str, Sequence[str | Section]] | None = None,
        min_safety: float = 3.0,
        max_deflection_mm: float | None = None,
    ) -> Sweep:
        """Step each group's tube up or down its list of candidates, one size at a time.

        If the frame as drawn falls short of ``min_safety`` (or deflects past
        ``max_deflection_mm``), the first stations add material where it buys
        the most safety per kilogram, until it does not fall short. The rest
        take material away where it costs the least safety per kilogram,
        until any lighter step would fall short. Each station is one size
        step from the last -- or, where no one step helps, two groups
        stepped together or one group jumped further -- and its ``design``
        names each group's tube.

        ``groups`` maps a group to the tubes it may be -- a 2" hitch shank
        only to 2x2 sizes, say. Groups not named keep their tubes, and
        without ``groups`` every group of tubes from :data:`TUBES` in the
        frame's own material may be any of them; the rest stay as they are.
        """
        names = sorted({g for g in self.groups if g is not None})
        if groups is None:
            groups = {
                name: list(TUBES)
                for name in names
                if all(
                    self.sections[i].name in TUBES and self.materials[i] is None
                    for i, g in enumerate(self.groups)
                    if g == name
                )
            }
        unknown = set(groups) - set(names)
        if unknown:
            raise ValueError(f"no members are in group(s) {sorted(unknown)}")
        ladders: dict[str, list[Section]] = {}
        current: dict[str, int] = {}
        for name, options in groups.items():
            stock = {s.name: s for s in map(_section, options)}
            members = [i for i, g in enumerate(self.groups) if g == name]
            drawn = {self.sections[i].name for i in members}
            if len(drawn) > 1:
                raise ValueError(f"group {name!r} mixes sections {sorted(drawn)}; give it one")
            start = self.sections[members[0]]
            stock.setdefault(start.name, start)
            density = self._made_of(members[0]).density
            ladders[name] = _ladder(list(stock.values()), start, density)
            current[name] = [s.name for s in ladders[name]].index(start.name)

        def design(choice: Mapping[str, int]) -> list[Section]:
            chosen = list(self.sections)
            for i, g in enumerate(self.groups):
                if g is not None and g in choice:
                    chosen[i] = ladders[g][choice[g]]
            return chosen

        def short(result: FrameResult) -> float:
            """How far, all told, the members fall short of the targets: 0 when none do."""
            weakest = np.min([case.safety for case in result.cases.values()], axis=0)
            total = float(np.maximum(min_safety - weakest, 0.0).sum()) / min_safety
            if max_deflection_mm is not None:
                total += max(result.deflection - max_deflection_mm, 0.0) / max_deflection_mm
            return total

        def fits(result: FrameResult) -> bool:
            return short(result) == 0.0

        def moves(choice: dict[str, int], up: bool) -> list[dict[str, int]]:
            steps = (-1, 1) if up else (-1,)
            return [
                {**choice, name: choice[name] + step}
                for name, ladder in ladders.items()
                for step in steps
                if 0 <= choice[name] + step < len(ladder)
            ]

        def further(choice: dict[str, int]) -> list[dict[str, int]]:
            """When no one step helps: two groups stepped together, or one jumped further.

            A heavier X fixes the X but loads the shank past its limit, and
            only both at once get anywhere; or the next tube up is no better
            where it counts, and only the one after it is.
            """
            jumps = [
                {**choice, name: index}
                for name, ladder in ladders.items()
                for index in range(len(ladder))
                if abs(index - choice[name]) > 1
            ]
            pairs = [
                {**choice, a: choice[a] + sa, b: choice[b] + sb}
                for a, b in combinations(ladders, 2)
                for sa in (-1, 1)
                for sb in (-1, 1)
                if 0 <= choice[a] + sa < len(ladders[a]) and 0 <= choice[b] + sb < len(ladders[b])
            ]
            return pairs + jumps

        def strengthen(
            choices: list[dict[str, int]], here: FrameResult
        ) -> tuple[dict[str, int], FrameResult] | None:
            found = []
            for choice in choices:
                result = self.analyze(design(choice))
                gain, cost = short(here) - short(result), result.mass - here.mass
                if gain > 1e-9:
                    rank = (cost <= 0, gain if cost <= 0 else gain / cost)
                    found.append((rank, choice, result))
            if not found:
                return None
            _, choice, result = max(found, key=lambda step: step[0])
            return choice, result

        path = [(dict(current), self.analyze(design(current)))]
        # Short of the target: the move that closes the most of the shortfall
        # per kilogram -- summed over every member short of it, not just the
        # weakest, so that two members tied for weakest do not stall it. A
        # lighter tube that closes some -- by easing the load the rest of the
        # frame carries -- costs nothing, and comes first.
        while not fits(path[-1][1]):
            choice, here = path[-1]
            step = strengthen(moves(choice, up=True), here) or strengthen(further(choice), here)
            if step is None:
                break
            path.append(step)
        # On target: the lighter tube that costs the least safety per kilogram saved.
        while fits(path[-1][1]):
            here = path[-1][1]
            found = []
            for choice in moves(path[-1][0], up=False):
                result = self.analyze(design(choice))
                if fits(result):
                    saved, lost = here.mass - result.mass, here.safety - result.safety
                    rank = (lost <= 0, saved if lost <= 0 else saved / lost)
                    found.append((rank, choice, result))
            if not found:
                break
            _, choice, result = max(found, key=lambda step: step[0])
            path.append((choice, result))

        original = path[0][1].mass
        stations = []
        for choice, result in path:
            named = {name: ladders[name][index].name for name, index in choice.items()}
            stations.append(
                Station(
                    1.0 - result.mass / original,
                    {
                        "mass_kg": result.mass,
                        "safety_factor": result.safety,
                        "peak_stress_mpa": result.stress,
                        "deflection_mm": result.deflection,
                    },
                    _frame_mesh(self, result.sections),
                    design=named,
                    analysis=result,
                )
            )
        notes = []
        if not fits(path[-1][1]):
            notes.append(
                f"nothing in the lists reaches {min_safety:g}x"
                + ("" if max_deflection_mm is None else f" within {max_deflection_mm:g} mm")
                + f": the best is {path[-1][1].safety:.1f}x"
            )
        weak = [i for i, (_, result) in enumerate(path) if result.safety < self.fatigue_safety]
        if weak:
            notes.append(
                f"{len(weak)} of {len(path)} stations are under {self.fatigue_safety:g}x on some "
                "load case: fine for a load that is put on once, not for one shaken on the road"
            )
        return Sweep(
            "frame",
            stations,
            [
                Column("mass_kg", "mass kg", "{:.2f}"),
                Column("safety_factor", "safety", "{:.1f}x"),
                Column("peak_stress_mpa", "stress MPa", "{:.0f}"),
                Column("deflection_mm", "deflect mm", "{:.1f}"),
                *(Column(name, name, "{}", design=True) for name in ladders),
            ],
            note="\n".join(notes),
        )

    def mesh(self, sections: Sequence[Section] | None = None) -> Mesh:
        """Every member as a straight bar of its outer section, for a look or a print."""
        return _frame_mesh(self, list(self.sections if sections is None else sections))()

    # -- plates ------------------------------------------------------------------

    def gusset(
        self,
        node: int | str,
        first: int,
        second: int,
        *,
        thickness: float,
        plates: int = 2,
        reach: float | None = None,
        case: str | None = None,
        sections: Sequence[Section] | None = None,
        material: str | Material | None = None,
        spacing: float | None = None,
        resolution: int = 80,
    ) -> PlateStudy:
        """Gusset plates at a joint, loaded with what the frame says the joint carries.

        Cheek plates, ``plates`` of them -- one each side of the tubes --
        lying in the plane ``first`` and ``second`` span where they meet at
        ``node``. Each is welded over a band of each tube running ``reach``
        mm from the joint (four times the tube's depth by default, and no
        further than the member goes), and its outline is the hull of the
        two bands.

        The second member's band is held and the first member's carries what
        that member pushes into the joint -- in the case with the largest
        moment in the plates' plane, or ``case`` -- shared out over the
        plates as if they carried all of it. They do not: the tube-to-tube
        weld carries some too, and each station of the sweep says how much
        of the joint's moment the plates take off that weld.

        Plate coordinates run from the node, ``u`` along the first member and
        ``v`` across it towards the second; :attr:`PlateStudy.placement`
        puts them back in the frame.
        """
        n = self._node(node)
        for m in (first, second):
            if not 0 <= m < len(self.members) or n not in self.members[m]:
                raise ValueError(f"member {m} does not meet node {node!r}")
        if first == second:
            raise ValueError("a gusset joins two different members")
        chosen = list(self.sections if sections is None else sections)
        model = self._model()
        result = self.analyze(chosen)
        origin = np.asarray(self.nodes[n])

        def away(m: int) -> tuple[FloatArray, float]:
            a, b = self.members[m]
            span = np.asarray(self.nodes[b if a == n else a]) - origin
            length = float(np.linalg.norm(span))
            return span / length, length

        (along_a, length_a), (along_b, length_b) = away(first), away(second)
        normal = np.cross(along_a, along_b)
        if np.linalg.norm(normal) < 1e-6:
            raise ValueError("the members are in line: a gusset needs a corner")
        normal /= np.linalg.norm(normal)
        u, v = along_a, np.cross(normal, along_a)

        def across(m: int, direction: FloatArray) -> float:
            _, y, z = model.rotation[m]
            section = chosen[m]
            return float(abs(direction @ y) * section.width + abs(direction @ z) * section.height)

        depth_a = across(first, np.cross(normal, along_a))
        depth_b = across(second, np.cross(normal, along_b))
        reach_a = reach if reach is not None else min(4.0 * depth_a, length_a)
        reach_b = reach if reach is not None else min(4.0 * depth_b, length_b)

        def band(along: FloatArray, depth: float, start: float, far: float) -> FloatArray:
            side = np.cross(normal, along)
            half = depth / 2.0
            ends = ((start, -half), (start, half), (far, half), (far, -half))
            corners = [s * along + w * side for s, w in ends]
            return np.array([[c @ u, c @ v] for c in corners])

        band_a = band(along_a, depth_a, -depth_b / 2.0, reach_a)
        band_b = band(along_b, depth_b, -depth_a / 2.0, reach_b)
        corners = np.vstack([band_a, band_b])
        outline = corners[ConvexHull(corners).vertices]
        plate = PlateStudy(
            outline,
            thickness=thickness,
            material=self.material if material is None else material,
            spacing=spacing,
            resolution=resolution,
        )
        held = plate.mask(band_b)
        loaded = plate.mask(band_a) & ~held

        near = self.members[first][0] == n
        turn = model.rotation[first]

        def pushed(c: CaseResult) -> tuple[FloatArray, FloatArray]:
            """What the first member pushes into the joint, in frame axes: force and moment."""
            ends = c.end_forces[first, :6] if near else c.end_forces[first, 6:]
            return -(turn.T @ ends[:3]), -(turn.T @ ends[3:])

        if case is None:
            case = max(result.cases, key=lambda k: abs(pushed(result.cases[k])[1] @ normal))
        elif case not in result.cases:
            raise ValueError(f"no load case {case!r}; there are {sorted(result.cases)}")
        force, moment = pushed(result.cases[case])
        plate.fix(held)
        plate.load(
            loaded,
            force=(float(force @ u) / plates, float(force @ v) / plates),
            moment=float(moment @ normal) / plates,
            about=(0.0, 0.0),
        )
        plate.placement = Placement(*(_vector(a) for a in (origin, u, v, normal)))

        names = {i: k for k, i in self.names.items()}
        label = (
            f"the {self.groups[first] or f'member {first}'}-to-"
            f"{self.groups[second] or f'member {second}'} weld at {names.get(n, f'node {n}')}"
        )
        bend = []
        for m, start, far in ((first, -depth_b / 2.0, reach_a), (second, -depth_a / 2.0, reach_b)):
            _, y, z = model.rotation[m]
            inertia = (normal @ y) ** 2 * chosen[m].iy + (normal @ z) ** 2 * chosen[m].iz
            bend.append((start + far) / 2.0 / (self._made_of(m).youngs_modulus * inertia))
        plate.relief = Relief(
            weld=label,
            moment=abs(float(moment @ normal)),
            case=case,
            tube_stiffness=1.0 / sum(bend),
            plates=plates,
            region=loaded,
        )

        sideways = abs(float(force @ normal))
        twisting = float(np.hypot(moment @ u, moment @ v))
        if sideways > 1.0 or twisting > 0.01 * abs(float(moment @ normal)):
            plate.notes.append(
                f"{case}: {sideways:.0f} N across the plates and {twisting / 1000:.0f} N m out "
                "of their plane go through the tubes and their own weld, not the plates"
            )
        for m, far, length in ((first, reach_a, length_a), (second, reach_b, length_b)):
            if far > 0.5 * length:
                plate.notes.append(
                    f"the plates run {far / length:.0%} of the way along "
                    f"{self.groups[m] or f'member {m}'}: if the joint at its other end has "
                    "plates too, cut each side as one plate over both"
                )
        thick_a = across(first, normal)
        thick_b = across(second, normal)
        if abs(thick_a - thick_b) > 0.5:
            narrow = (first, thick_a) if thick_a < thick_b else (second, thick_b)
            plate.notes.append(
                f"the tubes are {thick_a:.1f} and {thick_b:.1f} mm across: a flat plate on "
                f"each side needs a {abs(thick_a - thick_b) / 2:.1f} mm spacer on "
                f"{self.groups[narrow[0]] or f'member {narrow[0]}'}, or tubes the same width"
            )
        plate.offset = (max(thick_a, thick_b) + thickness) / 2.0
        return plate

    # -- internals ---------------------------------------------------------------

    def _node(self, ref: int | str) -> int:
        if isinstance(ref, str):
            if ref not in self.names:
                raise ValueError(f"no node called {ref!r}")
            return self.names[ref]
        if not 0 <= int(ref) < len(self.nodes):
            raise ValueError(f"no node {ref}")
        return int(ref)

    def _made_of(self, member: int) -> _Stuff:
        return self.materials[member] or self.stuff

    def _case(self, name: str) -> _Case:
        if name not in self.cases:
            self.cases[name] = _Case()
        return self.cases[name]

    def _length(self, member: int) -> float:
        a, b = self.members[member]
        return float(np.linalg.norm(np.subtract(self.nodes[b], self.nodes[a])))

    def _model(self) -> FrameModel:
        if not self.members:
            raise ValueError("add some members first")
        if not self.fixed:
            raise ValueError("support the frame somewhere first")
        fixed = np.zeros((len(self.nodes), 6), dtype=bool)
        for node, names in self.fixed.items():
            for name in names:
                fixed[node, DOFS.index(name)] = True
        made_of = [self._made_of(i) for i in range(len(self.members))]
        return FrameModel(
            np.array(self.nodes),
            np.array(self.members),
            fixed,
            youngs_modulus=np.array([m.youngs_modulus for m in made_of]),
            shear_modulus=np.array([m.shear_modulus for m in made_of]),
            strength=np.array([m.strength for m in made_of]),
        )

    def _loads(
        self, sections: Sequence[Section], model: FrameModel
    ) -> dict[str, Loads]:
        cases = self.cases or {"default": _Case()}
        weight = np.array([
            s.mass_per_mm(self._made_of(i).density) * GRAVITY for i, s in enumerate(sections)
        ])  # fmt: skip
        loads = {}
        for name, case in cases.items():
            nodal = np.zeros((len(self.nodes), 6))
            for node, force, moment in case.nodal:
                nodal[node, :3] += force
                nodal[node, 3:] += moment
            spread = np.zeros((len(self.members), 3))
            for member, q in case.distributed:
                spread[member] += q
            if case.self_weight:
                spread[:, 2] -= weight
            points = (
                np.array([member for member, _, _ in case.points], dtype=np.int64),
                np.array([at for _, at, _ in case.points], dtype=np.float64),
                np.array([f for _, _, f in case.points], dtype=np.float64).reshape(-1, 3)
                * case.factor,
            )
            loads[name] = (nodal * case.factor, spread * case.factor, points)
        return loads


def _vector(a: FloatArray) -> Vector3:
    return (float(a[0]), float(a[1]), float(a[2]))


def _material(spec: str | Material) -> Material:
    # FrameStudy's ``material`` argument shadows the module function inside it.
    return material(spec)


def _ladder(options: list[Section], drawn: Section, density: float) -> list[Section]:
    """A group's tubes lightest first, less any another tube beats on every count.

    Mass alone does not order tubes by strength: a 2.5" tube with a thin wall
    is lighter than a 2" one with a thick wall, and stiffer and stronger too.
    Dropping every tube that some other is no heavier than and no weaker
    than, in bending both ways and in torsion, leaves a list in which each
    step up in mass is a step up in strength. The tube as drawn always stays.
    """
    def beats(a: Section, b: Section) -> bool:
        lighter = a.mass_per_mm(density) <= b.mass_per_mm(density)
        no_weaker = all(getattr(a, k) >= getattr(b, k) for k in _STRENGTHS)
        better = a.mass_per_mm(density) < b.mass_per_mm(density) or any(
            getattr(a, k) > getattr(b, k) for k in _STRENGTHS
        )
        return lighter and no_weaker and better

    kept = [s for s in options if s.name == drawn.name or not any(beats(o, s) for o in options)]
    return sorted(kept, key=lambda s: s.mass_per_mm(density))


_STRENGTHS = ("sy", "sz", "iy", "iz", "j")


def _stuff(spec: str | Material | Mapping[str, Any]) -> _Stuff:
    if not isinstance(spec, Mapping):
        m = material(spec)
        return _Stuff(
            m.youngs_modulus,
            m.youngs_modulus / (2.0 * (1.0 + m.poisson_ratio)),
            m.strength,
            m.density,
        )
    e = float(spec["youngs_modulus"])
    if "shear_modulus" in spec:
        g = float(spec["shear_modulus"])
    else:
        g = e / (2.0 * (1.0 + float(spec.get("poisson_ratio", 0.3))))
    stuff = _Stuff(e, g, float(spec["strength"]), float(spec["density"]))
    if min(stuff.youngs_modulus, stuff.shear_modulus, stuff.strength) <= 0:
        raise ValueError(f"a material needs positive moduli and strength, got {dict(spec)}")
    return stuff


_PROPERTIES = ("area", "iy", "iz", "j", "sy", "sz", "wt", "shear_y", "shear_z")


def _section(spec: str | Section | Mapping[str, Any]) -> Section:
    if isinstance(spec, Section):
        return spec
    if isinstance(spec, Mapping):
        missing = [key for key in _PROPERTIES if key not in spec]
        if missing:
            raise ValueError(f"a section given as numbers needs {missing} too")
        values = {key: float(spec[key]) for key in _PROPERTIES}
        if min(values.values()) <= 0:
            raise ValueError(f"a section's numbers must all be positive, got {values}")
        side = float(np.sqrt(values["area"]))
        return Section(
            name=str(spec.get("name", "custom")),
            width=float(spec.get("width", side)),
            height=float(spec.get("height", side)),
            wall=None,
            mass_area=float(spec.get("mass_area", values["area"])),
            **values,
        )
    return TUBES[spec] if spec in TUBES else section(spec)


def _frame_mesh(study: FrameStudy, sections: list[Section]) -> Callable[[], Mesh]:
    nodes = np.array(study.nodes)
    members = np.array(study.members)

    def build() -> Mesh:
        span = nodes[members[:, 1]] - nodes[members[:, 0]]
        length = np.linalg.norm(span, axis=1)
        axes = _axes(span / length[:, None])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
        vertices, faces = [], []
        for i, (a, sec) in enumerate(zip(members[:, 0], sections, strict=True)):
            _, y, z = axes[i]
            ring = corners[:, :1] * (sec.width / 2.0) * y + corners[:, 1:] * (sec.height / 2.0) * z
            near, far = nodes[a] + ring, nodes[a] + span[i] + ring
            base = 8 * i
            vertices.append(np.concatenate([near, far]))
            for p, q, r, w in _BOX:
                faces += [[base + p, base + q, base + r], [base + p, base + r, base + w]]
        return Mesh(np.concatenate(vertices), np.array(faces, dtype=np.int64))

    return build
