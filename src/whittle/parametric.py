"""Tuning a parametric model's own dimensions against a flow.

A :class:`~whittle.study.FlowStudy` takes any closed STL, re-meshes it from
voxels and pushes the surface around with a lattice. That works on anything,
but what comes back is a mesh -- smooth only as far as the lattice and the
remesh allow, and with nothing left of the design intent that made it.

When the part already exists as a parametric model -- a CadQuery script, an
OpenSCAD file, a few lines of numpy -- it is better to search its own
dimensions instead: tail length, nose radius, how far the top falls away at
the back. Every candidate is then a real design, as smooth as the model makes
it, and the answer is a set of numbers to type back into the model rather
than a mesh to reverse-engineer. Fewer numbers, too, which a search settles
far faster than the eighty-odd moves of a lattice.

The builder is called once per candidate, so it sets the pace: a few
thousand calls per sweep. Keep the searched parameters to a handful.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .fitness.aero import DragArea
from .fitness.base import Candidate, Objective, WeightedSum
from .fitness.shape import KeepOut, VolumeBudget
from .optimize.evolve import evolve
from .physics.drag import DragModel, DragReport
from .study import _AXES, Box, _constant
from .surface import open_edges, split_long_edges, weld
from .sweep import Column, Station, Sweep
from .types import FloatArray, Genome, Mesh, Vector3

#: What a builder may return: a mesh, or ``(vertices, faces)`` as anything
#: numpy can read -- which is what CadQuery's ``shape.tessellate()`` gives,
#: once its vertices are turned into tuples.
Built = Mesh | tuple[Any, Any]

#: Metric names a sweep's stations already use, which a parameter may not take.
_RESERVED = {"removed", "drag", "cd", "cda_m2", "force_n", "frontal_m2"}


class BuildFailed(Exception):
    """The builder could not make a shape from these parameters."""


class ParametricStudy:
    """Tune a parametric model's dimensions for drag; a slider of designs, each a set of numbers.

    ::

        def build(p):
            body = cq.Workplane("XY").box(p["length"], p["width"], p["height"])
            body = body.edges(">Z and |Y").fillet(p["radius"])
            vertices, triangles = body.val().tessellate(0.5, 0.2)
            return [v.toTuple() for v in vertices], triangles

        study = ParametricStudy(
            build,
            {"radius": (5, 200)},
            start={"length": 990, "width": 1880, "height": 1040, "radius": 10},
            flow=(-1, 0, 0),
        )
        sweep = study.sweep(remove_up_to=0.2)
        print(sweep)
        sweep.at(0.10).design  # {"length": 990.0, ..., "radius": 163.2}

    ``build`` takes a dict of every parameter and returns the shape, in
    millimetres. ``bounds`` names the parameters to search and the range of
    each; ``start`` is the design as it stands, and gives every parameter
    ``build`` needs -- those not in ``bounds`` are passed through unchanged.
    A build that raises is a candidate the search skips, so a fillet too big
    for its edge costs nothing but the attempt.

    Each station of the sweep is the lowest-drag design found that gives up at
    most that much of the starting volume, as with a flow study. Unlike one,
    a parametric design can also grow if its bounds let it -- a longer tail
    -- and then its ``removed`` is negative. The bounds are the place to say
    what may not change.

    Shapes are welded and their long edges split before scoring, so no edge
    is longer than ``1 / resolution`` of the starting design's longest side --
    about the size of a flow study's re-meshed triangles, which is what the
    drag model was calibrated on. CAD exports need both: they repeat every
    vertex along the seams between faces, and pair big flat triangles with
    thin strips along fillets. The geometry is untouched; it just gets more,
    evener triangles, which is what the drag model reads curvature and
    shelter from. Every candidate is evened to the same size, so they are
    ranked on the same footing.

    ``obstacles`` are fixed bodies the part rides in the wake of -- the car
    in front of a hitch box. They shelter the part but are never scored and
    never change. ``extruded`` scores the part as a slice of something
    endless, as for a flow study.
    """

    def __init__(
        self,
        build: Callable[[dict[str, float]], Built],
        bounds: Mapping[str, tuple[float, float]],
        *,
        start: Mapping[str, float],
        flow: Vector3,
        resolution: int = 20,
        extruded: str | int | None = None,
        speed_kmh: float = 100.0,
        obstacles: Sequence[Mesh] = (),
    ) -> None:
        if not bounds:
            raise ValueError("name at least one parameter to search")
        self.builder = build
        self.start = {name: float(value) for name, value in start.items()}
        self.bounds: dict[str, tuple[float, float]] = {}
        for name, (low, high) in bounds.items():
            if name in _RESERVED:
                raise ValueError(f"{name!r} is a metric name; call the parameter something else")
            if name not in self.start:
                raise ValueError(f"start= gives no value for {name!r}")
            if not low <= self.start[name] <= high:
                raise ValueError(
                    f"{name}={self.start[name]:g} is outside its bounds ({low}, {high})"
                )
            self.bounds[name] = (float(low), float(high))
        axis = _AXES[extruded] if isinstance(extruded, str) else extruded
        if axis is None:
            self.model = DragModel(flow, obstacles=obstacles)
        else:
            self.model = DragModel(flow, aspect=float("inf"), ends=axis, obstacles=obstacles)
        self.speed_kmh = float(speed_kmh)
        self.keep_boxes: list[Box] = []

        try:
            first = _as_mesh(build(dict(self.start)))
            lo, hi = first.bounds
            self.max_edge = float((hi - lo).max()) / resolution
            self.tolerance = 1e-5 * float(np.linalg.norm(hi - lo))
            self.base = self._prepare(first)
        except Exception as error:
            raise ValueError(f"the starting design does not build: {error}") from error
        holes = open_edges(self.base)
        if holes:
            raise ValueError(
                f"the starting design has {holes} open edges after welding; "
                "the drag model needs a closed solid"
            )

    def keep(self, box: Box) -> ParametricStudy:
        """Keep the part's surface out of this box: the cargo it exists to hold."""
        self.keep_boxes.append(box)
        return self

    def shape(self, params: Mapping[str, float] | None = None) -> Mesh:
        """The mesh for these parameters, ready to score. Unnamed ones come from ``start``.

        Raises :class:`BuildFailed` if the builder raises or returns nothing
        solid.
        """
        full = {**self.start, **({} if params is None else params)}
        try:
            built = _as_mesh(self.builder(full))
        except Exception as error:
            raise BuildFailed(f"{type(error).__name__}: {error}") from error
        return self._prepare(built)

    def _prepare(self, built: Mesh) -> Mesh:
        """Weld, turn outward and even out a built mesh, or raise :class:`BuildFailed`."""
        mesh = weld(built, self.tolerance)
        volume = mesh.volume if len(mesh.faces) else 0.0
        if volume < 0:
            mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1].copy())
        lo, hi = mesh.bounds
        if abs(volume) <= 1e-9 * float(np.prod(np.maximum(hi - lo, 1e-12))):
            raise BuildFailed("the builder returned no closed solid")
        return split_long_edges(mesh, self.max_edge)

    def score(self, params: Mapping[str, float] | None = None) -> DragReport:
        """The drag on one design -- by default the starting one."""
        return self.model(self.shape(params))

    def sweep(
        self,
        *,
        remove_up_to: float = 0.25,
        stations: int | Sequence[float] = 6,
        effort: float = 1.0,
        seed: int | None = 0,
        callback: Callable[[int, float, float], None] | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> Sweep:
        """Tune the parameters for each volume budget in turn, each from the last.

        The same slider as a flow study's, and the same hooks: ``effort``
        scales the search per station, ``callback(station, removed, drag)``
        follows each station, ``progress`` the fraction of the run done.
        Raising from either abandons the run.
        """
        if isinstance(stations, int):
            if stations < 2:
                raise ValueError("need at least two stations: the original and one tuned")
            budgets = list(np.linspace(0.0, remove_up_to, stations))[1:]
        else:
            budgets = sorted(float(b) for b in stations if b > 0)
        shape = _Shape(self)
        volume = self.base.volume
        reference = self.model(self.base).drag_area
        drag = DragArea(self.model, reference=reference)
        constraints: dict[Objective, float] = {}
        if self.keep_boxes:
            constraints[KeepOut(self.keep_boxes)] = 20.0
        generations = max(5, int(round(30 * effort)))

        genome = shape.genome(self.start)
        designs = [(genome, self.base)]
        best = 1.0
        for number, budget in enumerate(budgets, start=1):

            def step(generation: int, _: float, done: int = number - 1) -> None:
                if progress is not None:
                    fraction = min(generation, generations) / generations
                    progress((done + fraction) / len(budgets))

            objective = _Forgiving(
                WeightedSum({drag: 1.0, VolumeBudget(volume, budget): 20.0, **constraints})
            )
            result = evolve(
                shape,
                objective,
                seed=seed,
                maxiter=generations,
                popsize=6,
                x0=genome,
                spread=0.3 if number == 1 else 0.15,
                callback=step,
            )
            # As for a flow study: penalties are normalized, a thousandth of
            # one is invisible, and the budget gets a percent of slack.
            slack = {"over_budget": 0.01}
            terms = result.terms
            feasible = "drag" in terms and all(
                v <= slack.get(k, 1e-3) for k, v in terms.items() if k != "drag"
            )
            if feasible and terms["drag"] < best:
                genome, best = result.genome, float(terms["drag"])
            mesh = shape.build(genome)
            designs.append((genome, mesh))
            if callback is not None:
                callback(number, 1.0 - mesh.volume / volume, best)

        speed = self.speed_kmh / 3.6
        result_stations = []
        for design, mesh in designs:
            report = self.model(mesh)
            params = shape.params(design)
            metrics = {
                "drag": report.drag_area / reference,
                "cd": report.cd,
                "cda_m2": report.drag_area * 1e-6,
                "force_n": report.force(speed),
                "frontal_m2": report.frontal_area * 1e-6,
                **{name: params[name] for name in self.bounds},
            }
            result_stations.append(
                Station(
                    1.0 - mesh.volume / volume,
                    metrics,
                    _constant(mesh),
                    design=params,
                    analysis=report,
                )
            )
        notes = []
        reached = result_stations[-1].removed
        if budgets and reached < budgets[-1] - 0.02:
            notes.append(
                f"stopped short at {reached:.0%} removed: nothing found that gave up more"
                " volume was any slipperier -- the bounds, or the gains, ran out"
            )
        if shape.failed:
            tried = shape.failed + shape.built
            notes.append(
                f"{shape.failed} of {tried} candidate designs did not build and were skipped"
                f" -- the last said: {shape.last_error}"
            )
        note = "\n".join(notes)
        return Sweep(
            "flow",
            result_stations,
            [
                Column("drag", "drag", "{:.0%}"),
                Column("cd", "Cd", "{:.3f}"),
                Column("cda_m2", "CdA m2", "{:.4f}"),
                Column("force_n", f"N @{self.speed_kmh:g}km/h", "{:.1f}"),
                *(Column(name, name, "{:.3g}") for name in self.bounds),
            ],
            note=note,
        )


class _Shape:
    """A parametric study's builder, seen as a geometry backend: a genome in, a mesh out.

    Remembers its last few hundred meshes, because the search re-scores its
    best candidate every generation and a builder call is the expensive part.
    """

    changes_topology = True

    def __init__(self, study: ParametricStudy, *, remember: int = 256) -> None:
        self.study = study
        self.names = list(study.bounds)
        self.remember = remember
        self.built = 0
        self.failed = 0
        self.last_error = ""
        self._meshes: dict[bytes, Mesh] = {}

    @property
    def n_params(self) -> int:
        return len(self.names)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [self.study.bounds[name] for name in self.names]

    def params(self, genome: Genome) -> dict[str, float]:
        """Every parameter of the design: the searched ones from the genome, the rest from start."""
        searched = dict(zip(self.names, (float(g) for g in genome), strict=True))
        return {**self.study.start, **searched}

    def genome(self, params: Mapping[str, float]) -> FloatArray:
        return np.array([params[name] for name in self.names], dtype=np.float64)

    def build(self, genome: Genome) -> Mesh:
        key = np.asarray(genome, dtype=np.float64).tobytes()
        mesh = self._meshes.get(key)
        if mesh is None:
            try:
                mesh = self.study.shape(self.params(genome))
            except BuildFailed as error:
                self.failed += 1
                self.last_error = str(error)
                raise
            self.built += 1
            if len(self._meshes) >= self.remember:
                del self._meshes[next(iter(self._meshes))]
            self._meshes[key] = mesh
        return mesh


class _Forgiving:
    """An objective that scores a design that would not build as hopeless, instead of failing."""

    def __init__(self, inner: WeightedSum, *, penalty: float = 1e3) -> None:
        self.inner = inner
        self.penalty = penalty

    @property
    def name(self) -> str:
        return self.inner.name

    def terms(self, candidate: Candidate) -> dict[str, float]:
        try:
            return self.inner.terms(candidate)
        except BuildFailed:
            return {"failed": 1.0}

    def __call__(self, candidate: Candidate) -> float:
        try:
            return self.inner(candidate)
        except BuildFailed:
            return self.penalty


def _as_mesh(built: Built) -> Mesh:
    if isinstance(built, Mesh):
        return built
    vertices, faces = built
    return Mesh(
        np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )
