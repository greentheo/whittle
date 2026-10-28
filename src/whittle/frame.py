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
sizes them together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .physics.fea import Material, material
from .physics.frame import (
    DOFS,
    GRAVITY,
    TUBES,
    CaseResult,
    FrameModel,
    Section,
    _axes,
    section,
)
from .sweep import Column, Station, Sweep
from .types import FloatArray, Mesh, Vector3

#: Below this safety factor on a load case, a welded steel frame under road
#: vibration is in the range where fatigue cracks start at the welds.
FATIGUE_SAFETY = 3.5

_FIXES = {"all": DOFS, "pin": DOFS[:3]}

#: A bar's six faces, as quads of its eight corners: the near end's four, then the far end's.
_BOX = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (3, 2, 1, 0), (4, 5, 6, 7))


@dataclass
class _Case:
    factor: float = 1.0
    self_weight: bool = True
    nodal: list[tuple[int, Vector3, Vector3]] = field(default_factory=list)
    distributed: list[tuple[int, FloatArray]] = field(default_factory=list)


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
        self.nodes: list[Vector3] = []
        self.names: dict[str, int] = {}
        self.members: list[tuple[int, int]] = []
        self.sections: list[Section] = []
        self.groups: list[str | None] = []
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
        self, a: int | str, b: int | str, section: str | Section, *, group: str | None = None
    ) -> int:
        """Join two nodes with a member of this section; returns its index."""
        self.members.append((self._node(a), self._node(b)))
        self.sections.append(_section(section))
        self.groups.append(group)
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

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> FrameStudy:
        """A frame from plain data, as another program would hand it over.

        ::

            {"material": "steel",
             "nodes": {"mouth": [76.2, 0, 228.6], ...},     # or a list: index = id
             "members": [{"a": "mouth", "b": "jog", "section": "2x2x1/4", "group": "shank"}],
             "supports": [{"node": "mouth", "fix": ["ux", "uy", "uz", "rx"]}],
             "cases": {"driving": {"factor": 2.5, "self_weight": true}},
             "loads": [{"case": "driving", "node": "tip", "force": [0, 0, -445]},
                       {"case": "driving", "member": 7, "total": [0, 0, -222]}]}
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
            study.member(item["a"], item["b"], item["section"], group=item.get("group"))
        for item in spec.get("supports", []):
            study.support(item["node"], fix=item.get("fix", "all"))
        for name, case in spec.get("cases", {}).items():
            study.case(
                name, factor=case.get("factor", 1.0), self_weight=case.get("self_weight", True)
            )
        for item in spec.get("loads", []):
            case_name = item.get("case", "default")
            if "member" in item:
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
        density = self.material.density
        mass = float(sum(s.mass_per_mm(density) * length for s, length in zip(
            chosen, model.length, strict=True
        )))
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
        step from the last, and its ``design`` names each group's tube.

        ``groups`` maps a group to the tubes it may be -- a 2" hitch shank
        only to 2x2 sizes, say. Groups not named keep their tubes, and
        without ``groups`` every group may be any tube in :data:`TUBES`.
        """
        names = sorted({g for g in self.groups if g is not None})
        if groups is None:
            groups = {name: list(TUBES) for name in names}
        unknown = set(groups) - set(names)
        if unknown:
            raise ValueError(f"no members are in group(s) {sorted(unknown)}")
        density = self.material.density
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

        path = [(dict(current), self.analyze(design(current)))]
        # Short of the target: the move that closes the most of the shortfall
        # per kilogram -- summed over every member short of it, not just the
        # weakest, so that two members tied for weakest do not stall it. A
        # lighter tube that closes some -- by easing the load the rest of the
        # frame carries -- costs nothing, and comes first.
        while not fits(path[-1][1]):
            here = path[-1][1]
            found = []
            for choice in moves(path[-1][0], up=True):
                result = self.analyze(design(choice))
                gain, cost = short(here) - short(result), result.mass - here.mass
                if gain > 1e-9:
                    rank = (cost <= 0, gain if cost <= 0 else gain / cost)
                    found.append((rank, choice, result))
            if not found:
                break
            _, choice, result = max(found, key=lambda step: step[0])
            path.append((choice, result))
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

    # -- internals ---------------------------------------------------------------

    def _node(self, ref: int | str) -> int:
        if isinstance(ref, str):
            if ref not in self.names:
                raise ValueError(f"no node called {ref!r}")
            return self.names[ref]
        if not 0 <= int(ref) < len(self.nodes):
            raise ValueError(f"no node {ref}")
        return int(ref)

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
        return FrameModel(
            np.array(self.nodes),
            np.array(self.members),
            fixed,
            youngs_modulus=self.material.youngs_modulus,
            poisson_ratio=self.material.poisson_ratio,
            strength=self.material.strength,
        )

    def _loads(
        self, sections: Sequence[Section], model: FrameModel
    ) -> dict[str, tuple[FloatArray, FloatArray]]:
        cases = self.cases or {"default": _Case()}
        weight = np.array([s.mass_per_mm(self.material.density) * GRAVITY for s in sections])
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
            loads[name] = (nodal * case.factor, spread * case.factor)
        return loads


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


def _section(spec: str | Section) -> Section:
    if isinstance(spec, Section):
        return spec
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
