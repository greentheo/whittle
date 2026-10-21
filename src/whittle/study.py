"""Studies: a part, what is done to it, and the slider that comes out.

This is the layer a person -- or a GUI -- talks to. Both studies have the same
shape:

1. Bring a part, as a mesh in millimetres.
2. Say what happens to it. For a :class:`LoadStudy`, where it is held and what
   pushes on it. For a :class:`FlowStudy`, which way the air comes from.
3. Call ``sweep()`` and get back a :class:`~whittle.sweep.Sweep`: the part with
   more and more material removed, each station scored, from the original to
   the most aggressive design the physics will allow.

Every method that places something takes either ``at=`` (a point, as a click
on the surface would give) or ``box=`` (two corners), in model coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .fitness.aero import DragArea
from .fitness.base import Objective, WeightedSum
from .fitness.shape import KeepOut, NoFolding, StayInside, VolumeBudget
from .geometry.ffd import FFDLattice
from .optimize.beso import beso
from .optimize.evolve import evolve
from .physics.drag import DragModel, DragReport
from .physics.fea import FEAResult, Load, Material, Support, VoxelFEA, material
from .sweep import Column, Station, Sweep
from .types import BoolArray, Mesh, Vector3
from .voxel import Grid

Box = tuple[Vector3, Vector3]


class LoadStudy:
    """Lighten a part without losing the strength it needs.

    ::

        study = LoadStudy(io.load("bracket.stl"), material="petg")
        study.support(box=((0, 0, 0), (5, 40, 60)))        # bolted to the wall
        study.load((0, 0, -100), at=(80, 20, 55))         # ~10 kg on the tip
        sweep = study.sweep(remove_up_to=0.6)
        print(sweep)
        io.save(sweep.at(0.4).mesh, "bracket-40.stl")

    Voxel size sets both the cost and the finest member the result can have.
    ``resolution`` voxels along the part's longest side is the default way to
    pick it; 40-60 is a few seconds to a minute per sweep on a laptop.
    """

    def __init__(
        self,
        part: Mesh,
        *,
        material: str | Material = "pla",
        spacing: float | None = None,
        resolution: int = 48,
    ) -> None:
        grid = Grid.around(part, spacing=spacing, resolution=resolution)
        solid = grid.voxelize(part)
        if not solid.any():
            raise ValueError("the part voxelized to nothing; is the mesh closed?")
        self._setup(grid, solid, material)
        self.part: Mesh | None = part

    @classmethod
    def from_voxels(
        cls, grid: Grid, solid: BoolArray, *, material: str | Material = "pla"
    ) -> LoadStudy:
        """A study of a part that is already a voxel occupancy on ``grid``."""
        study = cls.__new__(cls)
        study._setup(grid, np.asarray(solid, dtype=bool), material)
        study.part = None
        return study

    def _setup(self, grid: Grid, solid: BoolArray, spec: str | Material) -> None:
        if solid.shape != grid.shape:
            raise ValueError(f"occupancy must have shape {grid.shape}, got {solid.shape}")
        self.grid = grid
        self.solid = solid
        self.material = material(spec)
        self.loads: list[Load] = []
        self.supports: list[Support] = []
        self.protected = np.zeros(grid.shape, dtype=bool)

    def region(
        self, *, at: Vector3 | None = None, box: Box | None = None, radius: float | None = None
    ) -> BoolArray:
        """The part's voxels at a point or in a box."""
        if (at is None) == (box is None):
            raise ValueError("give exactly one of at= or box=")
        if box is not None:
            mask = self.grid.box(*box) & self.solid
            if not mask.any():
                raise ValueError(f"box {box} contains none of the part")
            return mask
        assert at is not None
        return self.grid.near(at, self.solid, radius=radius)

    def support(
        self,
        *,
        at: Vector3 | None = None,
        box: Box | None = None,
        radius: float | None = None,
        axes: tuple[bool, bool, bool] = (True, True, True),
    ) -> LoadStudy:
        """Hold part of the part in place: a bolt, a clamp, a face glued down."""
        self.supports.append(Support(self.region(at=at, box=box, radius=radius), axes))
        return self

    def load(
        self,
        force: Vector3,
        *,
        at: Vector3 | None = None,
        box: Box | None = None,
        radius: float | None = None,
    ) -> LoadStudy:
        """Push on the part. ``force`` is the total, in newtons (1 kg is about 9.8 N)."""
        self.loads.append(Load(self.region(at=at, box=box, radius=radius), force))
        return self

    def keep(
        self, *, at: Vector3 | None = None, box: Box | None = None, radius: float | None = None
    ) -> LoadStudy:
        """Material that must survive whatever the physics thinks: a bolt boss, a mating face."""
        self.protected |= self.region(at=at, box=box, radius=radius)
        return self

    def solver(self) -> VoxelFEA:
        if not self.loads or not self.supports:
            raise ValueError("place at least one support and one load first")
        return VoxelFEA(
            self.grid.shape,
            spacing=self.grid.spacing,
            material=self.material,
            loads=self.loads,
            supports=self.supports,
        )

    def analyze(self) -> FEAResult:
        """Solve the original part, untouched."""
        return self.solver().solve(self.solid)

    def sweep(
        self,
        *,
        remove_up_to: float = 0.6,
        stations: int | Sequence[float] = 7,
        evolution_rate: float = 0.02,
        filter_radius: float = 1.5,
        smooth: float = 0.6,
        relax: int = 8,
        callback: Callable[[int, float, float], None] | None = None,
    ) -> Sweep:
        """Whittle the part down in steps, recording a design at each.

        ``stations`` is either a count, spread evenly from nothing removed to
        ``remove_up_to``, or an explicit list of removal fractions.

        ``smooth`` and ``relax`` only shape the exported meshes, not what was
        analyzed: see :meth:`~whittle.voxel.Grid.mesh`. The defaults move the
        surface by a percent or two of the volume and make members look like
        members rather than stacks of voxels.
        """
        if isinstance(stations, int):
            if stations < 2:
                raise ValueError("need at least two stations: the original and one lighter")
            removals = list(np.linspace(0.0, remove_up_to, stations))
        else:
            removals = [0.0, *stations]
        fea = self.solver()
        run = beso(
            fea,
            self.solid,
            removals=removals,
            keep=self.protected,
            evolution_rate=evolution_rate,
            filter_radius=filter_radius,
            callback=callback,
        )
        baseline = run.stations[0].result.compliance
        voxel_cm3 = self.grid.voxel_volume / 1000.0
        result_stations = []
        for s in run.stations:
            result = s.result
            metrics = {
                "mass_g": float(s.solid.sum()) * voxel_cm3 * self.material.density,
                "stiffness": baseline / result.compliance,
                "deflection_mm": result.max_displacement,
                "peak_stress_mpa": result.peak_stress,
                "safety_factor": self.material.strength / result.peak_stress,
            }
            result_stations.append(
                Station(
                    s.removed, metrics, _mesher(self.grid, s.solid, smooth, relax), design=s.solid
                )
            )
        return Sweep(
            "load",
            result_stations,
            [
                Column("mass_g", "mass g", "{:.1f}"),
                Column("stiffness", "stiffness", "{:.0%}"),
                Column("deflection_mm", "deflect mm", "{:.3f}"),
                Column("peak_stress_mpa", "stress MPa", "{:.1f}"),
                Column("safety_factor", "safety", "{:.1f}x"),
            ],
            note=run.note,
        )


_AXES = {"x": 0, "y": 1, "z": 2}


class FlowStudy:
    """Carve a part down to cut its drag, without ever growing it.

    ::

        study = FlowStudy(io.load("roof_box.stl"), flow=(-1, 0, 0), mirror="y", flat=("-z",))
        sweep = study.sweep(remove_up_to=0.25)
        print(sweep)
        io.save(sweep.at(0.10).mesh, "roof_box-10.stl")

    ``flow`` is the direction the air moves past the part, as in a wind
    tunnel: a box on a car driving towards +x sees ``(-1, 0, 0)``.

    The part is re-meshed from voxels first, ``resolution`` along its longest
    side, so any closed STL works however it was tessellated. Then a coarse
    deformation lattice -- ``detail`` divisions per axis -- pushes its surface
    inward, and each station of the sweep is the lowest-drag shape found that
    removes at most that much of the original volume. Coarse is deliberate: two
    divisions can taper a tail, slope a nose and round the corners, which is
    most of what drag wants, and a coarse lattice cannot make the wiggles that
    a fine one would use to game the model.

    ``mirror`` keeps the result symmetric about the part's middle on that
    axis. ``flat`` lists sides of the part's bounding box -- ``"-z"`` is the
    bottom -- that must stay flat and where they are, like a face that sits on
    roof bars. :meth:`keep` protects a box of the interior from being cut into.

    ``extruded`` is for long parts with a constant cross-section -- a roof bar,
    a strut, a fairing strip. Model a short slice of it, name the axis it runs
    along, and the slice is scored as a piece of something endless: its cut
    ends are ignored, its wake is the strong two-dimensional kind, and the
    carving keeps the cross-section the same all along it.
    """

    def __init__(
        self,
        part: Mesh,
        *,
        flow: Vector3,
        resolution: int = 40,
        detail: int | tuple[int, int, int] = 2,
        mirror: str | int | None = None,
        flat: Sequence[str] = (),
        extruded: str | int | None = None,
        speed_kmh: float = 100.0,
        relax: int = 15,
    ) -> None:
        self.part = part
        self.extruded = _AXES[extruded] if isinstance(extruded, str) else extruded
        if self.extruded is None:
            self.model = DragModel(flow)
        else:
            self.model = DragModel(flow, aspect=float("inf"), ends=self.extruded)
        self.grid = Grid.around(part, resolution=resolution, pad=2)
        self.solid = self.grid.voxelize(part)
        if not self.solid.any():
            raise ValueError("the part voxelized to nothing; is the mesh closed?")
        self.base = self.grid.mesh(self.solid, relax=relax)
        self.detail = (detail, detail, detail) if isinstance(detail, int) else detail
        self.mirror = _AXES[mirror] if isinstance(mirror, str) else mirror
        if self.extruded is not None:
            if self.mirror is not None and self.mirror != self.extruded:
                raise ValueError("an extruded part is uniform along its length; drop mirror")
            # Two mirrored layers along the length move the cross-section as one.
            self.mirror = self.extruded
            layers = list(self.detail)
            layers[self.extruded] = 1
            self.detail = (layers[0], layers[1], layers[2])
        self.flat = tuple(flat)
        for side in self.flat:
            if len(side) != 2 or side[0] not in "+-" or side[1] not in _AXES:
                raise ValueError(f"sides look like '-z' or '+x', got {side!r}")
        self.speed_kmh = float(speed_kmh)
        self.keep_boxes: list[Box] = []

    def keep(self, box: Box) -> FlowStudy:
        """Protect a box inside the part from being carved into."""
        self.keep_boxes.append(box)
        return self

    def analyze(self, mesh: Mesh | None = None) -> DragReport:
        """Score a mesh -- by default the part itself, as re-meshed."""
        return self.model(self.base if mesh is None else mesh)

    def lattice(self) -> FFDLattice:
        dims = tuple(d + 1 for d in self.detail)
        frozen = np.zeros((*dims, 3), dtype=bool)
        for side in self.flat:
            axis = _AXES[side[1]]
            layer: list[int | slice] = [slice(None), slice(None), slice(None)]
            layer[axis] = 0 if side[0] == "-" else dims[axis] - 1
            frozen[tuple(layer) + (axis,)] = True
        if self.extruded is not None:
            # A slice may not get longer or shorter, only change its section.
            frozen[..., self.extruded] = True
        lo, hi = self.base.bounds
        return FFDLattice(
            self.base,
            self.detail,
            max_displacement=tuple(0.5 * (hi - lo)),
            frozen=frozen,
            mirror=self.mirror,
            inward=True,
        )

    def sweep(
        self,
        *,
        remove_up_to: float = 0.25,
        stations: int | Sequence[float] = 6,
        effort: float = 1.0,
        seed: int | None = 0,
        callback: Callable[[int, float, float], None] | None = None,
    ) -> Sweep:
        """Carve the part to each removal budget in turn, each from the last.

        ``effort`` scales the search budget per station. ``callback(station,
        removed, drag)`` is called as each station is finished.
        """
        if isinstance(stations, int):
            if stations < 2:
                raise ValueError("need at least two stations: the original and one carved")
            budgets = list(np.linspace(0.0, remove_up_to, stations))[1:]
        else:
            budgets = sorted(float(b) for b in stations if b > 0)
        lattice = self.lattice()
        volume = self.base.volume
        reference = self.model(self.base).drag_area
        constraints: dict[Objective, float] = {
            StayInside(self.grid, self.solid): 5.0,
            NoFolding(self.base): 10.0,
        }
        if self.keep_boxes:
            constraints[KeepOut(self.keep_boxes)] = 20.0
        drag = DragArea(self.model, reference=reference)

        genome = np.zeros(lattice.n_params)
        designs = [(0.0, genome, self.base)]
        best = 1.0
        for number, budget in enumerate(budgets, start=1):
            objective = WeightedSum({drag: 1.0, VolumeBudget(volume, budget): 20.0, **constraints})
            result = evolve(
                lattice,
                objective,
                seed=seed,
                maxiter=max(5, int(round(30 * effort))),
                popsize=6,
                x0=genome,
                spread=0.15 if number == 1 else 0.08,
            )
            # Every penalty is normalized -- a fraction of the volume, of a voxel,
            # of the faces, of a keep box -- so a thousandth of one is invisible.
            feasible = all(v <= 1e-3 for k, v in result.terms.items() if k != "drag")
            # Never let the slider get worse as it goes. A larger budget can
            # always fall back on the smaller one's design, which still fits.
            if feasible and result.terms["drag"] < best:
                genome, best = result.genome, result.terms["drag"]
            mesh = lattice.build(genome)
            designs.append((budget, genome, mesh))
            if callback is not None:
                callback(number, 1.0 - mesh.volume / volume, best)

        speed = self.speed_kmh / 3.6
        result_stations = []
        for _, design, mesh in designs:
            report = self.model(mesh)
            metrics = {
                "drag": report.drag_area / reference,
                "cd": report.cd,
                "cda_m2": report.drag_area * 1e-6,
                "force_n": report.force(speed),
                "frontal_m2": report.frontal_area * 1e-6,
            }
            result_stations.append(
                Station(1.0 - mesh.volume / volume, metrics, _constant(mesh), design=design)
            )
        return Sweep(
            "flow",
            result_stations,
            [
                Column("drag", "drag", "{:.0%}"),
                Column("cd", "Cd", "{:.3f}"),
                Column("cda_m2", "CdA m2", "{:.4f}"),
                Column("force_n", f"N @{self.speed_kmh:g}km/h", "{:.1f}"),
            ],
        )


def _mesher(grid: Grid, solid: BoolArray, smooth: float, relax: int) -> Callable[[], Mesh]:
    return lambda: grid.mesh(solid, smooth=smooth, relax=relax)


def _constant(mesh: Mesh) -> Callable[[], Mesh]:
    return lambda: mesh
