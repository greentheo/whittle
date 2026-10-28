"""Plates: gussets and brackets cut from flat stock, lightened, and drawn for the plasma table.

::

    plate = PlateStudy([(0, 0), (250, 0), (250, 60), (60, 250), (0, 250)], thickness=4.7625)
    plate.fix([(0, 0), (60, 0), (60, 250), (0, 250)])            # welded to the riser
    plate.load([(60, 0), (250, 0), (250, 60), (60, 60)], force=(0, -2000), moment=-4e5)
    sweep = plate.sweep(remove_up_to=0.5)
    sweep[3].design.save("gusset.dxf")

A plate cut from sheet is loaded in its own plane, so it is analysed as one
layer of voxels: the voxel can be any size, and the loads are scaled by its
thickness over the plate's, which leaves the stresses those of the real plate.
Material can only come out all the way through, so lightening makes holes --
and holes a plasma table can cut: none narrower than ``min_hole``, no web
between them narrower than ``min_web``, a solid rim round the outline, and a
margin of solid plate along every weld.

:meth:`whittle.frame.FrameStudy.gusset` sets one up at a frame's joint, with
the loads the frame says the joint carries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import draw, measure

from .optimize.beso import beso
from .physics.fea import FEAResult, FieldLoad, Material, Support, VoxelFEA, material
from .sweep import Column, Station, Sweep
from .types import BoolArray, FloatArray, Mesh, Vector3
from .voxel import Grid

#: Below this a welded plate shaken on the road is in fatigue territory, as for frames.
FATIGUE_SAFETY = 3.5

Polygon = Sequence[Sequence[float]] | FloatArray


@dataclass(frozen=True)
class Placement:
    """Where a plate sits in a frame: its corner of coordinates and its axes, all in mm."""

    origin: Vector3
    u: Vector3
    v: Vector3
    normal: Vector3

    def place(self, uv: FloatArray, offset: float = 0.0) -> FloatArray:
        """Plate coordinates ``(n, 2)`` to frame coordinates, ``offset`` along the normal."""
        uv = np.asarray(uv, dtype=np.float64)
        result: FloatArray = (
            np.asarray(self.origin)
            + uv[:, :1] * np.asarray(self.u)
            + uv[:, 1:2] * np.asarray(self.v)
            + offset * np.asarray(self.normal)
        )
        return result


@dataclass(frozen=True)
class Cut:
    """A plate as it is cut: an outline and the holes in it, plate coordinates in mm."""

    #: Counter-clockwise, ``(n, 2)``.
    outline: FloatArray
    #: Each clockwise, ``(n, 2)``.
    holes: list[FloatArray]
    thickness: float

    @property
    def area(self) -> float:
        """mm^2 of plate left."""
        return _area(self.outline) - sum(-_area(h) for h in self.holes)

    def loops(self) -> list[FloatArray]:
        return [self.outline, *self.holes]

    def svg(self) -> str:
        """The cut as an SVG drawing, true size in mm, v up."""
        lo = self.outline.min(axis=0)
        hi = self.outline.max(axis=0)
        width, height = hi - lo
        paths = " ".join(
            "M " + " L ".join(f"{u - lo[0]:.3f} {hi[1] - v:.3f}" for u, v in loop) + " Z"
            for loop in self.loops()
        )
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width:.3f}mm" height="{height:.3f}mm" '
            f'viewBox="0 0 {width:.3f} {height:.3f}">\n'
            f'  <path d="{paths}" fill="#9aa4ad" fill-rule="evenodd" '
            'stroke="#000" stroke-width="0.3"/>\n</svg>\n'
        )

    def dxf(self) -> str:
        """The cut as an R12 DXF: one closed polyline per loop, in mm, on layer CUT.

        R12 because every plasma and CAM program reads it.
        """
        lines = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
                 "9", "$INSUNITS", "70", "4", "9", "$MEASUREMENT", "70", "1", "0", "ENDSEC",
                 "0", "SECTION", "2", "ENTITIES"]  # fmt: skip
        for loop in self.loops():
            lines += ["0", "POLYLINE", "8", "CUT", "66", "1", "70", "1",
                      "10", "0.0", "20", "0.0", "30", "0.0"]  # fmt: skip
            for u, v in loop:
                lines += ["0", "VERTEX", "8", "CUT"]
                lines += ["10", f"{u:.4f}", "20", f"{v:.4f}", "30", "0.0"]
            lines += ["0", "SEQEND", "8", "CUT"]
        lines += ["0", "ENDSEC", "0", "EOF"]
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> Path:
        """Write a ``.dxf`` or ``.svg``, by the file's extension."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in (".dxf", ".svg"):
            raise ValueError(f"a cut saves as .dxf or .svg, not {suffix or 'no extension'}")
        path.write_text(self.dxf() if suffix == ".dxf" else self.svg())
        return path


@dataclass(frozen=True)
class Relief:
    """How much of a joint's moment its plates take off the weld between the members.

    The plates and the tube-to-tube weld hold the joint together side by
    side, and share its moment by stiffness: the plates' measured from the
    plate itself, the tubes' from their bending between the plates' welds.
    """

    #: The weld relieved, in words: "the shank-to-riser weld at corner_low".
    weld: str
    #: The joint's moment in the plates' plane, N mm, and the load case it is from.
    moment: float
    case: str
    #: N mm per radian, of the tubes' path from one plate weld to the other.
    tube_stiffness: float
    plates: int
    #: Where the loaded member's weld is: the moment's handle on the plate.
    region: BoolArray


@dataclass
class _Push:
    region: BoolArray
    force: FloatArray
    moment: float
    about: FloatArray | None


class PlateStudy:
    """A flat plate: an outline, welds that hold it, welds that load it -- and lighter versions.

    Coordinates ``(u, v)`` are in mm in the plate's own plane. Regions --
    for :meth:`fix`, :meth:`load` and :meth:`keep` -- are polygons in the
    same coordinates, clipped to the outline.
    """

    def __init__(
        self,
        outline: Polygon,
        *,
        thickness: float,
        material: str | Material = "steel",
        spacing: float | None = None,
        resolution: int = 80,
    ) -> None:
        poly = np.asarray(outline, dtype=np.float64).reshape(-1, 2)
        if len(poly) < 3:
            raise ValueError("an outline needs at least three corners")
        if _area(poly) < 0:
            poly = poly[::-1]
        if thickness <= 0:
            raise ValueError(f"a plate needs a positive thickness, got {thickness}")
        self.outline = poly
        self.thickness = float(thickness)
        self.material = material_of(material)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        self.spacing = float(spacing or (hi - lo).max() / resolution)
        pad = 2
        cells = np.ceil((hi - lo) / self.spacing).astype(int) + 2 * pad
        #: The centre of cell (0, 0), in plate coordinates.
        self.corner = lo + self.spacing / 2.0 - pad * self.spacing
        self.grid = Grid((int(cells[0]), int(cells[1]), 1), self.spacing,
                         (float(self.corner[0]), float(self.corner[1]), 0.0))  # fmt: skip
        self.domain = self.mask(poly)
        if self.domain.sum() < 20:
            raise ValueError("the plate is only a few voxels across; give a finer spacing")
        self.held = np.zeros(self.domain.shape, dtype=bool)
        self.pushes: list[_Push] = []
        self.protected = np.zeros(self.domain.shape, dtype=bool)
        #: Where the plate sits in a frame, if :meth:`FrameStudy.gusset` made it,
        #: and how far each cheek plate's mid-plane is from it, either side.
        self.placement: Placement | None = None
        self.offset = 0.0
        #: Things the loads leave out, worth reading.
        self.notes: list[str] = []
        #: The joint weld these plates relieve, if :meth:`FrameStudy.gusset` made them.
        self.relief: Relief | None = None

    # -- setting it up ---------------------------------------------------------

    def mask(self, polygon: Polygon) -> BoolArray:
        """The plate's cells whose centres fall inside ``polygon``."""
        poly = (np.asarray(polygon, dtype=np.float64).reshape(-1, 2) - self.corner) / self.spacing
        inside: BoolArray = draw.polygon2mask(  # type: ignore[no-untyped-call]
            self.grid.shape[:2], poly
        )
        if hasattr(self, "domain"):
            inside &= self.domain
        return inside

    def fix(self, region: Polygon | BoolArray) -> PlateStudy:
        """Hold the plate here: welded to the member that carries the load away."""
        self.held |= self._region(region)
        return self

    def load(
        self,
        region: Polygon | BoolArray,
        *,
        force: Sequence[float] = (0.0, 0.0),
        moment: float = 0.0,
        about: Sequence[float] | None = None,
    ) -> PlateStudy:
        """Push on the plate here: a force (N) in its plane and a moment (N mm) about its normal.

        Spread over the region so that the force and the moment about
        ``about`` -- the region's centroid if not given -- come out exactly:
        a member welded along it, pushing as a whole.
        """
        cells = self._region(region)
        f = np.asarray(force, dtype=np.float64).reshape(2)
        point = None if about is None else np.asarray(about, dtype=np.float64).reshape(2)
        self.pushes.append(_Push(cells, f, float(moment), point))
        return self

    def keep(self, region: Polygon | BoolArray) -> PlateStudy:
        """Plate that must stay whatever the physics thinks: a bolt hole's surround, a tab."""
        self.protected |= self._region(region)
        return self

    # -- solving it --------------------------------------------------------------

    def solver(self) -> VoxelFEA:
        if not self.pushes or not self.held.any():
            raise ValueError("fix the plate somewhere and load it somewhere first")
        return VoxelFEA(
            self.grid.shape,
            spacing=self.spacing,
            material=self.material,
            loads=[FieldLoad(self._forces())],
            supports=[Support(self.held[:, :, None])],
        )

    def analyze(self) -> FEAResult:
        """Solve the plate whole, with no holes."""
        return self.solver().solve(self.domain[:, :, None])

    def sweep(
        self,
        *,
        remove_up_to: float = 0.5,
        stations: int = 6,
        min_safety: float = FATIGUE_SAFETY,
        min_hole: float | None = None,
        min_web: float | None = None,
        rim: float | None = None,
        margin: float | None = None,
        evolution_rate: float = 0.02,
        filter_radius: float = 1.5,
        progress: Callable[[float], None] | None = None,
    ) -> Sweep:
        """Lighten the plate in steps, each station a plate ready to cut.

        Holes come from strain energy (BESO), then each station is made
        cuttable -- holes no narrower than ``min_hole``, webs no narrower
        than ``min_web``, a ``rim`` of solid plate round the outline and a
        ``margin`` beside every weld -- and solved again as it will be cut.
        Its ``design`` is the :class:`Cut`, ready for :meth:`Cut.save`.
        """
        t = self.thickness
        min_hole = max(12.7, 1.5 * t) if min_hole is None else float(min_hole)
        min_web = max(8.0, 2.0 * t) if min_web is None else float(min_web)
        rim = min_web if rim is None else float(rim)
        margin = max(6.0, t) if margin is None else float(margin)
        if stations < 2:
            raise ValueError("need at least two stations: the plate whole and one lighter")
        fea = self.solver()
        welds = self.held | np.any(self._forces() != 0.0, axis=-1)[:, :, 0]
        keep = (
            self.protected
            | (self.domain & ~_erode(self.domain, rim / self.spacing))
            | (self.domain & _dilate(welds, margin / self.spacing))
        )
        removals = list(np.linspace(0.0, remove_up_to, stations))

        def report(_: int, removed: float, __: float) -> None:
            if progress is not None:
                progress(min(1.0, removed / (remove_up_to or 1.0)))

        run = beso(
            fea,
            self.domain[:, :, None],
            removals=removals,
            keep=keep[:, :, None],
            evolution_rate=evolution_rate,
            filter_radius=filter_radius,
            callback=report,
        )
        whole = _area(self.outline)
        density = self.material.density
        result_stations: list[Station] = []
        for s in run.stations:
            plate = self._cuttable(s.solid[:, :, 0], keep, min_hole, min_web)
            result = fea.solve(plate[:, :, None])
            cut = self._cut(plate, min_hole)
            metrics = {
                "mass_kg": cut.area * t * density * 1e-6,
                "safety_factor": self.material.strength / result.peak_stress,
                "peak_stress_mpa": result.peak_stress,
                "deflection_mm": result.max_displacement,
                "holes": float(len(cut.holes)),
            }
            if result_stations and cut.area == result_stations[-1].design.area:  # type: ignore[attr-defined]
                continue  # cleaned back to the same plate as the station before
            note = ""
            if self.relief is not None:
                share, left = self._relieved(plate, self.relief)
                metrics["plate_share"] = share
                metrics["weld_moment_nmm"] = left
                note = _relief_note(self.relief, share, left)
            result_stations.append(
                Station(1.0 - cut.area / whole, metrics, self._slab(plate), design=cut,
                        analysis=result, note=note)  # fmt: skip
            )
        notes = []
        if run.note:
            # The plate's own reason, not the generic one: the welds hold most of a gusset.
            bands = float((self.domain & welds).sum()) / float(self.domain.sum())
            held = float(keep.sum()) / float(self.domain.sum())
            notes.append(
                f"stopped at {run.stations[-1].removed:.0%} removed of the {remove_up_to:.0%} "
                f"asked: {held:.0%} of the plate has to stay -- {bands:.0%} is welded over the "
                "tubes, the rest the margin beside the welds and the rim round the outline. "
                "Longer legs leave more plate between the tubes to take; a narrower margin or "
                "rim a little more"
            )
        notes += self.notes
        if self.relief is not None:
            notes += list(dict.fromkeys([result_stations[0].note, result_stations[-1].note]))
        weak = [s for s in result_stations if s.metrics["safety_factor"] < min_safety]
        if weak:
            notes.append(
                f"{len(weak)} of {len(result_stations)} plates are under {min_safety:g}x: "
                "thicker plate, longer legs, or fewer holes"
            )
        return Sweep(
            "plate",
            result_stations,
            [
                Column("mass_kg", "kg/plate", "{:.3f}"),
                Column("safety_factor", "safety", "{:.1f}x"),
                Column("peak_stress_mpa", "stress MPa", "{:.0f}"),
                Column("deflection_mm", "deflect mm", "{:.3f}"),
                Column("holes", "holes", "{:.0f}"),
            ],
            note="\n".join(notes),
        )

    # -- internals ---------------------------------------------------------------

    def _region(self, region: Polygon | BoolArray) -> BoolArray:
        if isinstance(region, np.ndarray) and region.dtype == bool:
            if region.shape != self.domain.shape:
                raise ValueError(f"a region mask must have shape {self.domain.shape}")
            cells: BoolArray = np.logical_and(region, self.domain)
        else:
            cells = self.mask(region)  # type: ignore[arg-type]
        if not cells.any():
            raise ValueError("that region misses the plate")
        return cells

    def _centres(self, cells: BoolArray) -> FloatArray:
        result: FloatArray = self.corner + self.spacing * np.argwhere(cells)
        return result

    def _forces(self, pushes: Sequence[_Push] | None = None) -> FloatArray:
        """Newtons per cell, for one layer of cells: each push spread statically equivalent.

        Scaled by the cells' thickness over the plate's, so that a layer of
        cells strains as the plate does.
        """
        forces = np.zeros((*self.domain.shape, 1, 3))
        scale = self.spacing / self.thickness
        for push in self.pushes if pushes is None else pushes:
            index = np.argwhere(push.region)
            r = self._centres(push.region)
            c = r.mean(axis=0)
            about = c if push.about is None else push.about
            arm = about - c
            torque = push.moment + arm[0] * push.force[1] - arm[1] * push.force[0]
            d = r - c
            polar = float((d**2).sum())
            if polar == 0.0 and torque != 0.0:
                raise ValueError("a moment needs a region more than one cell across")
            spin = np.column_stack([-d[:, 1], d[:, 0]]) * (torque / polar if polar else 0.0)
            each = push.force / len(r) + spin
            forces[index[:, 0], index[:, 1], 0, :2] += each * scale
        return forces

    def _relieved(self, plate: BoolArray, relief: Relief) -> tuple[float, float]:
        """The plates' share of the joint's moment, and what is left on the weld, N mm."""
        twist = 1e5
        push = _Push(relief.region, np.zeros(2), twist, None)
        fea = VoxelFEA(
            self.grid.shape,
            spacing=self.spacing,
            material=self.material,
            loads=[FieldLoad(self._forces([push]))],
            supports=[Support(self.held[:, :, None])],
        )
        moved = fea.solve(plate[:, :, None]).displacement
        cell = 0.25 * (moved[:-1, :-1] + moved[1:, :-1] + moved[:-1, 1:] + moved[1:, 1:])
        cell = cell.mean(axis=2)[..., :2]
        index = np.argwhere(relief.region)
        d = self._centres(relief.region)
        d -= d.mean(axis=0)
        w = cell[index[:, 0], index[:, 1]]
        turned = float((d[:, 0] * w[:, 1] - d[:, 1] * w[:, 0]).sum() / (d**2).sum())
        stiffness = relief.plates * twist / turned if turned > 0 else float("inf")
        share = stiffness / (stiffness + relief.tube_stiffness)
        return share, relief.moment * (1.0 - share)

    def _cuttable(
        self, solid: BoolArray, keep: BoolArray, min_hole: float, min_web: float
    ) -> BoolArray:
        """Make a lightened plate one a plasma table can cut: open the slivers, widen the webs."""
        holes = self.domain & ~solid
        holes = _erode(holes, min_web / 2.0 / self.spacing)
        holes = ndimage.binary_opening(holes, structure=_disk(min_hole / 2.0 / self.spacing))
        plate: BoolArray = (self.domain & ~holes) | keep
        return plate

    def _cut(self, plate: BoolArray, min_hole: float) -> Cut:
        holes = np.pad(self.domain & ~plate, 1).astype(np.float64)
        smooth = ndimage.gaussian_filter(holes, sigma=1.0)
        loops = []
        smallest = 0.25 * math.pi * (min_hole / 2.0) ** 2
        for contour in measure.find_contours(smooth, 0.5):  # type: ignore[no-untyped-call]
            if len(contour) < 4 or not np.allclose(contour[0], contour[-1]):
                continue
            # Simplify away the voxels' staircase, then round what is left:
            # rounding cuts corners, so a hole only ever gets a little smaller.
            loop = measure.approximate_polygon(  # type: ignore[no-untyped-call]
                contour, tolerance=0.35
            )[:-1]
            loop = _rounded(_rounded(loop))
            loop = self.corner + self.spacing * (loop - 1.0)
            if abs(_area(loop)) < smallest:
                continue
            loops.append(loop if _area(loop) < 0 else loop[::-1])
        return Cut(self.outline.copy(), loops, self.thickness)

    def _slab(self, plate: BoolArray) -> Callable[[], Mesh]:
        grid, thickness, placement = self.grid, self.thickness, self.placement

        def build() -> Mesh:
            mesh = grid.mesh(plate[:, :, None], smooth=0.0)
            v = mesh.vertices.copy()
            v[:, 2] *= thickness / grid.spacing
            if placement is not None:
                v = placement.place(v[:, :2], 0.0) + v[:, 2:3] * np.asarray(placement.normal)
            return Mesh(v, mesh.faces)

        return build


def _relief_note(relief: Relief, share: float, left: float) -> str:
    def both(nmm: float) -> str:
        return f"{nmm / 1000.0:,.0f} N m ({nmm / 112.985:,.0f} lb in)"

    return (
        f"{relief.weld}: {both(relief.moment)} in {relief.case} with no plates, about "
        f"{both(left)} with these {relief.plates} -- they take {share:.0%} of it, by "
        "stiffness, an estimate"
    )


def material_of(spec: str | Material) -> Material:
    # PlateStudy's ``material`` argument shadows the module function inside it.
    return material(spec)


def _rounded(loop: FloatArray) -> FloatArray:
    """Chaikin's corner cutting, once round a closed loop."""
    after = np.roll(loop, -1, axis=0)
    result: FloatArray = np.empty((2 * len(loop), 2))
    result[0::2] = 0.75 * loop + 0.25 * after
    result[1::2] = 0.25 * loop + 0.75 * after
    return result


def _area(loop: FloatArray) -> float:
    """Signed area of a closed loop: positive counter-clockwise."""
    u, v = loop[:, 0], loop[:, 1]
    return float(0.5 * (u * np.roll(v, -1) - np.roll(u, -1) * v).sum())


def _disk(radius: float) -> BoolArray:
    reach = max(1, int(math.ceil(radius)))
    i, j = np.ogrid[-reach : reach + 1, -reach : reach + 1]
    disk: BoolArray = i**2 + j**2 <= max(radius, 1.0) ** 2
    return disk


def _erode(mask: BoolArray, radius: float) -> BoolArray:
    if radius < 0.5:
        return mask
    eroded: BoolArray = ndimage.binary_erosion(mask, structure=_disk(radius))
    return eroded


def _dilate(mask: BoolArray, radius: float) -> BoolArray:
    if radius < 0.5:
        return mask
    grown: BoolArray = ndimage.binary_dilation(mask, structure=_disk(radius))
    return grown
