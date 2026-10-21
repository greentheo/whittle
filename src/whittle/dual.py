"""Shape a part for the wind and lighten it for its loads, as one map.

The two studies couple in one direction. A flow study carves the outside of a
part; a load study removes material, and as long as it keeps the outer skin --
which a roof box, a fairing or a housing has to -- it never changes the
outside, so it never changes the drag. What does flow from one to the other
is the envelope (which material exists to be lightened at all) and the wind
load (a slipperier shape is pushed on less).

So rather than alternate between them, which would settle on one compromise
whose position depends on step sizes, a :class:`DualStudy` nests them:

1. **Carve** the outside to a handful of drag budgets, as a flow study does.
2. **Lighten** inside each carved shape, keeping its skin, under the user's
   loads plus that shape's own wind loads -- once straight into the wind,
   once in a crosswind.
3. **Refine** along the carving axis wherever neighbouring shapes differ
   most, coarse first and filling in.

Each lightening run produces its whole column of designs at once (BESO's
history is its sweep), so a map of ``n`` shapes by ``m`` weights costs ``n``
carves and ``n`` lightening runs, not ``n * m`` of anything.

The crosswind case uses the crossflow principle: a crosswind load is the
drag at driving speed along the direction of travel plus the broadside drag
at crosswind speed across it. Both are head-on evaluations, which is where
the drag model is calibrated; its own answers for a yawed flow are not to be
trusted, because its suction terms were fitted to air arriving from the front.

z is up, as in STL files: cargo rests on the floor below its box, and the
crosswind blows horizontally.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .optimize.beso import beso
from .physics.drag import DragModel, DragReport
from .physics.fea import FieldLoad, Load, Material, Support, VoxelFEA, material
from .study import Box, Carver, FlowStudy, select
from .sweep import Column, Station
from .types import BoolArray, FloatArray, Mesh, Vector3
from .voxel import Grid

G = 9.81
UP = np.array([0.0, 0.0, 1.0])


def wind_forces(
    mesh: Mesh,
    report: DragReport,
    grid: Grid,
    solid: BoolArray,
    *,
    speed: float,
    air_density: float = 1.225,
    metres_per_unit: float = 1e-3,
) -> FloatArray:
    """The wind on ``mesh`` as forces on the voxels of ``solid``, in newtons.

    Each face's aerodynamic force (:attr:`DragReport.forces`) goes to the
    solid voxel nearest its centre, which for a closed surface meshed around
    the solid is the voxel just beneath it. The total is preserved exactly.
    """
    out = np.zeros((*grid.shape, 3), dtype=np.float64)
    if not solid.any() or not len(mesh.faces):
        return out
    _, nearest = ndimage.distance_transform_edt(~solid, return_indices=True)
    centroid = mesh.vertices[mesh.faces].mean(axis=1)
    index = np.rint((centroid - np.asarray(grid.origin)) / grid.spacing).astype(np.int64)
    index = np.clip(index, 0, np.asarray(grid.shape) - 1)
    target = tuple(nearest[axis][index[:, 0], index[:, 1], index[:, 2]] for axis in range(3))
    q = 0.5 * air_density * speed**2 * metres_per_unit**2
    for axis in range(3):
        np.add.at(out[..., axis], target, q * report.forces[:, axis])
    return out


@dataclass(frozen=True)
class Placement:
    """Where a support or load goes, kept as the user said it and re-applied to each shape."""

    at: Vector3 | None = None
    box: Box | None = None
    radius: float | None = None
    normal: Vector3 | None = None
    depth: float | None = None

    def region(self, grid: Grid, within: BoolArray) -> BoolArray:
        return select(
            grid,
            within,
            at=self.at,
            box=self.box,
            radius=self.radius,
            normal=self.normal,
            depth=self.depth,
        )


@dataclass
class DualRow:
    """One carved shape, and the lighter designs found inside it."""

    #: Fraction of the original volume the carving removed.
    removed: float
    #: The removal it was allowed: ``removed`` can fall short of it, if no
    #: lower-drag shape using more of it was found.
    budget: float
    #: The carved outside, as the flow study scored it.
    shape: Mesh = field(repr=False)
    genome: FloatArray = field(repr=False)
    report: DragReport = field(repr=False)
    #: Lighter and lighter designs inside ``shape``, index 0 unlightened.
    #: ``Station.removed`` is the fraction of this shape's own material removed.
    stations: list[Station] = field(default_factory=list)
    #: Why lightening stopped short, if it did.
    note: str = ""

    @property
    def drag(self) -> float:
        return float(self.stations[0].metrics["drag"]) if self.stations else 1.0


@dataclass
class DualSweep:
    """A map of designs: carved for drag down the rows, lightened for weight along each."""

    rows: list[DualRow]
    columns: list[Column]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a dual sweep needs at least one row")

    def at(self, aero: float, weight: float) -> Station:
        """The design nearest ``aero`` of the volume carved and ``weight`` of the rest removed."""
        row = min(self.rows, key=lambda r: abs(r.removed - aero))
        return min(row.stations, key=lambda s: abs(s.removed - weight))

    def designs(self) -> list[tuple[int, int, Station]]:
        """Every design, with its row and column."""
        return [(i, j, s) for i, row in enumerate(self.rows) for j, s in enumerate(row.stations)]

    def front(self, *, min_safety: float | None = None) -> list[tuple[int, int]]:
        """The designs no other beats on both drag and mass -- the trade-off worth looking at.

        With ``min_safety``, only designs at least that safe are considered:
        a light, slippery part that breaks is not on anyone's front.
        """
        candidates = [
            (i, j, s.metrics["drag"], s.metrics["mass_g"])
            for i, j, s in self.designs()
            if min_safety is None or s.metrics["safety_factor"] >= min_safety
        ]
        front = [
            (i, j)
            for i, j, d, m in candidates
            if not any(
                (d2 <= d and m2 <= m) and (d2 < d or m2 < m) for _, _, d2, m2 in candidates
            )
        ]
        return sorted(front, key=lambda ij: self.rows[ij[0]].stations[ij[1]].metrics["mass_g"])

    def table(self) -> str:
        lines = []
        for row in self.rows:
            lines.append(f"carved {row.removed:.0%}: drag {row.drag:.0%} of the original")
            for station in row.stations:
                cells = [f"  lightened {station.removed:4.0%}"]
                for column in self.columns:
                    value = station.metrics.get(column.key)
                    cells.append(f"{column.label} {column.format.format(value)}")
                lines.append("  ".join(cells))
            if row.note:
                lines.append(f"  ({row.note})")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


class DualStudy:
    """Shape a part for the wind and lighten it for its loads -- both, as one map.

    ::

        study = DualStudy(io.load("roof_box.stl"), flow=(-1, 0, 0), mirror="y", flat=("-z",))
        study.support(box=((-450, -460, -10), (-350, 460, 30)))   # front roof bar
        study.support(box=((350, -460, -10), (450, 460, 30)))     # rear roof bar
        study.cargo(((-750, -350, 40), (750, 350, 300)), kg=50)
        sweep = study.sweep()
        print(sweep)
        io.save(sweep.at(aero=0.15, weight=0.4).mesh, "roof_box.stl")

    ``cargo`` is a box that must stay empty -- carving cannot cut into it,
    lightening cannot put material in it -- with its weight resting on the
    floor beneath. ``support`` and ``load`` place things as a
    :class:`~whittle.study.LoadStudy` does, and are re-applied to every shape.
    Wind loads come from each shape's own drag at ``speed_kmh``, plus a
    crosswind case at ``crosswind_kmh``.

    ``skin`` is how many voxels of the outside lightening must leave alone,
    which is what keeps the carved shape -- and its drag -- intact.
    """

    def __init__(
        self,
        part: Mesh,
        *,
        flow: Vector3,
        material: str | Material = "pla",
        speed_kmh: float = 110.0,
        crosswind_kmh: float = 40.0,
        resolution: int = 40,
        structure_resolution: int = 48,
        detail: int | tuple[int, int, int] = 2,
        mirror: str | int | None = None,
        flat: Sequence[str] = (),
        skin: int = 1,
    ) -> None:
        if skin < 1:
            raise ValueError("keep at least one voxel of skin, or lightening can change the shape")
        self.flow = FlowStudy(
            part,
            flow=flow,
            resolution=resolution,
            detail=detail,
            mirror=mirror,
            flat=flat,
            speed_kmh=speed_kmh,
        )
        self.grid = Grid.around(part, resolution=structure_resolution)
        self.material = _material(material)
        self.speed_kmh = float(speed_kmh)
        self.crosswind_kmh = float(crosswind_kmh)
        self.skin = int(skin)
        self.supports: list[tuple[Placement, tuple[bool, bool, bool]]] = []
        self.loads: list[tuple[Vector3, Placement]] = []
        self.cargo_boxes: list[tuple[Box, float]] = []
        # Broadside: horizontal and square to the direction of travel.
        side = np.cross(UP, self.flow.model.flow)
        if np.linalg.norm(side) < 1e-9:
            raise ValueError("the flow is vertical; there is no crosswind direction")
        self.side: FloatArray = side / np.linalg.norm(side)
        self.broadside = DragModel((float(self.side[0]), float(self.side[1]), float(self.side[2])))

    def support(
        self,
        *,
        at: Vector3 | None = None,
        box: Box | None = None,
        radius: float | None = None,
        normal: Vector3 | None = None,
        depth: float | None = None,
        axes: tuple[bool, bool, bool] = (True, True, True),
    ) -> DualStudy:
        """Hold the part here -- where it sits on its bars, where it is bolted."""
        self.supports.append((Placement(at, box, radius, normal, depth), axes))
        return self

    def load(
        self,
        force: Vector3,
        *,
        at: Vector3 | None = None,
        box: Box | None = None,
        radius: float | None = None,
        normal: Vector3 | None = None,
        depth: float | None = None,
    ) -> DualStudy:
        """A force that acts in every case, in newtons, on top of the wind."""
        self.loads.append((force, Placement(at, box, radius, normal, depth)))
        return self

    def cargo(self, box: Box, *, kg: float = 0.0) -> DualStudy:
        """A space that stays empty, carrying ``kg`` on the floor beneath it."""
        self.cargo_boxes.append((box, float(kg)))
        self.flow.keep(box)
        return self

    def sweep(
        self,
        *,
        remove_up_to: float = 0.25,
        aero_stations: int = 5,
        lighten_up_to: float = 0.6,
        weight_stations: int = 5,
        effort: float = 1.0,
        seed: int | None = 0,
        min_safety: float = 2.0,
        progress: Callable[[float], None] | None = None,
    ) -> DualSweep:
        """Carve, lighten inside each carving, and fill in the carving axis where it matters.

        Starts with the original shape, the most carved one and one between,
        then adds shapes halfway between whichever neighbours differ most --
        in drag, and in how light a design can get while staying at least
        ``min_safety`` safe -- until there are ``aero_stations`` of them.
        ``progress`` gets the fraction done; raising from it abandons the run.
        """
        if not self.supports:
            raise ValueError("place at least one support first")
        if aero_stations < 2:
            raise ValueError("need at least two shapes: the original and one carved")
        carver = Carver(self.flow, effort=effort, seed=seed)
        removals = list(np.linspace(0.0, lighten_up_to, weight_stations))
        total = 2 * aero_stations
        done = 0

        def report(fraction: float) -> None:
            if progress is not None:
                progress(min(1.0, (done + fraction) / total))

        rows: list[DualRow] = []

        def add(budget: float, start: DualRow | None) -> None:
            nonlocal done
            genome = np.zeros(carver.lattice.n_params) if start is None else start.genome
            drag = 1.0 if start is None else start.drag
            if budget > 0:
                wide = start is None or start.removed == 0
                genome, drag = carver.carve(budget, genome, drag, wide=wide, progress=report)
            done += 1
            shape = self.flow.base if budget == 0 else carver.lattice.build(genome)
            row = self._lighten(shape, genome, carver.reference, carver.volume, removals, report)
            row.budget = budget
            done += 1
            rows.append(row)
            rows.sort(key=lambda r: r.removed)

        first = [0.0, remove_up_to] if aero_stations == 2 else [0.0, remove_up_to / 2, remove_up_to]
        for budget in first:
            add(budget, rows[-1] if rows else None)
        scale = max(rows[0].stations[0].metrics["mass_g"], 1e-9)

        def gap(pair: tuple[DualRow, DualRow]) -> float:
            lo, hi = pair
            light = abs(_lightest(lo, min_safety) - _lightest(hi, min_safety)) / scale
            return abs(lo.drag - hi.drag) + light

        while len(rows) < aero_stations:
            lo, hi = max(zip(rows, rows[1:], strict=False), key=gap)
            if hi.budget - lo.budget < 0.01:
                break  # nothing meaningful left between neighbours to fill in
            add((lo.budget + hi.budget) / 2.0, lo)
        return DualSweep(
            rows,
            [
                Column("mass_g", "mass g", "{:.0f}"),
                Column("drag", "drag", "{:.0%}"),
                Column("safety_factor", "safety", "{:.1f}x"),
                Column("deflection_mm", "deflect mm", "{:.2f}"),
            ],
        )

    def _lighten(
        self,
        shape: Mesh,
        genome: FloatArray,
        reference: float,
        volume: float,
        removals: list[float],
        progress: Callable[[float], None],
    ) -> DualRow:
        grid = self.grid
        occupancy = grid.voxelize(shape)
        void = np.zeros(grid.shape, dtype=bool)
        for box, _ in self.cargo_boxes:
            void |= grid.box(*box)
        domain = occupancy & ~void
        if not domain.any():
            raise ValueError("nothing is left to lighten once the cargo space is emptied")
        # The skin is every voxel with any neighbour outside, edges and corners
        # included. Face neighbours alone leave a sloping surface as a
        # staircase of voxels touching only at their edges, which falls apart
        # -- structurally -- the moment the material behind it is removed.
        cube = np.ones((3, 3, 3), dtype=bool)
        skin = occupancy & ~ndimage.binary_erosion(occupancy, structure=cube, iterations=self.skin)
        keep = skin & domain

        supports = [Support(p.region(grid, domain), axes) for p, axes in self.supports]
        fixed = [(force, p.region(grid, domain)) for force, p in self.loads]
        for box, kg in self.cargo_boxes:
            if kg <= 0:
                continue
            (x0, y0, z0), (x1, y1, _) = (np.minimum(*box), np.maximum(*box))
            floor = grid.box((x0, y0, z0 - 2.0 * grid.spacing), (x1, y1, z0)) & domain
            if not floor.any():
                raise ValueError(f"nothing under the cargo box {box} for it to rest on")
            fixed.append(((0.0, 0.0, -kg * G), floor))

        report = self.flow.model(shape)
        broadside = self.broadside(shape)
        speed, crosswind = self.speed_kmh / 3.6, self.crosswind_kmh / 3.6
        straight = wind_forces(shape, report, grid, domain, speed=speed)
        loads: list[Load | FieldLoad] = [FieldLoad(straight, case=0)]
        cases = [0]
        if crosswind > 0:
            across = wind_forces(shape, broadside, grid, domain, speed=crosswind)
            loads.append(FieldLoad(straight + across, case=1))
            cases.append(1)
        for force, region in fixed:
            loads.extend(Load(region, force, case=case) for case in cases)

        fea = VoxelFEA(
            grid.shape, spacing=grid.spacing, material=self.material, loads=loads, supports=supports
        )
        furthest = max(removals) or 1.0
        run = beso(
            fea,
            domain,
            removals=removals,
            keep=keep,
            callback=lambda _, removed, __: progress(min(1.0, removed / furthest)),
        )
        aero = {
            "drag": report.drag_area / reference,
            "cd": report.cd,
            "force_n": report.force(speed),
            # By the crossflow principle, the side force is the broadside drag.
            "side_n": broadside.force(crosswind),
        }
        base = run.stations[0].result.compliance
        voxel_cm3 = grid.voxel_volume / 1000.0
        stations = []
        for s in run.stations:
            result = s.result
            metrics = {
                **aero,
                "mass_g": float(s.solid.sum()) * voxel_cm3 * self.material.density,
                "stiffness": base / result.compliance,
                "deflection_mm": result.max_displacement,
                "peak_stress_mpa": result.peak_stress,
                "safety_factor": self.material.strength / result.peak_stress,
            }
            stations.append(
                Station(s.removed, metrics, _mesher(grid, s.solid), design=s.solid, analysis=result)
            )
        return DualRow(1.0 - shape.volume / volume, 0.0, shape, genome, report, stations, run.note)


def _lightest(row: DualRow, min_safety: float) -> float:
    safe = [s.metrics["mass_g"] for s in row.stations if s.metrics["safety_factor"] >= min_safety]
    return min(safe) if safe else row.stations[0].metrics["mass_g"]


def _material(spec: str | Material) -> Material:
    # DualStudy's ``material`` argument shadows the module function inside it.
    return material(spec)


def _mesher(grid: Grid, solid: BoolArray) -> Callable[[], Mesh]:
    return lambda: grid.mesh(solid, relax=8)
