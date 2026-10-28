"""A local web app around the studies: load a part, click to place, slide.

Standard library only -- ``http.server`` and one worker thread -- so the GUI
adds nothing to a library that otherwise needs three dependencies. It binds
to localhost and keeps everything in memory: a tool for one person on their
own machine, not a service. The page itself loads three.js from a CDN, so
the browser needs to be online the first time.

The HTTP layer is thin on purpose. :class:`App` holds the state and does the
work, and is what the tests drive; the handler only translates requests.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import traceback
import uuid
import webbrowser
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .. import io
from ..dual import DualStudy, DualSweep
from ..physics.drag import DragReport
from ..physics.fea import MATERIALS, FEAResult
from ..study import Box, FlowStudy, LoadStudy
from ..sweep import Sweep
from ..types import BoolArray, FloatArray, Mesh, Vector3
from ..voxel import Grid
from .presets import PRESETS, preset

STATIC = Path(__file__).parent / "static"
UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
MAX_UPLOAD = 200 * 1024 * 1024


class Cancelled(Exception):
    """Raised from a progress hook to abandon a running study."""


class BadRequest(ValueError):
    """Something the page sent that cannot be acted on. Shown to the user as is."""


@dataclass
class Part:
    id: str
    name: str
    mesh: Mesh
    _voxels: dict[int, tuple[Grid, BoolArray]] = field(default_factory=dict)
    _coverage: dict[int, FloatArray] = field(default_factory=dict)

    def voxels(self, resolution: int) -> tuple[Grid, BoolArray]:
        """The part on a grid ``resolution`` voxels along its longest side. Cached."""
        if resolution not in self._voxels:
            grid = Grid.around(self.mesh, resolution=resolution)
            solid = grid.voxelize(self.mesh)
            if not solid.any():
                raise BadRequest("the part voxelized to nothing -- is the mesh closed?")
            self._voxels[resolution] = (grid, solid)
        return self._voxels[resolution]

    def coverage(self, resolution: int) -> FloatArray:
        """How much of each voxel the part fills, on the grid :meth:`voxels` gives. Cached."""
        if resolution not in self._coverage:
            grid, _ = self.voxels(resolution)
            self._coverage[resolution] = grid.coverage(self.mesh)
        return self._coverage[resolution]


@dataclass
class Job:
    id: str
    part: Part
    settings: dict[str, Any]
    state: str = "queued"
    progress: float = 0.0
    error: str = ""
    started: float = 0.0
    finished: float = 0.0
    study: LoadStudy | FlowStudy | DualStudy | None = None
    sweep: Sweep | None = None
    dual: DualSweep | None = None
    #: Every station, in order: a sweep's stations, or a dual map's row by row.
    stations: list[Any] = field(default_factory=list)
    heat_max: float = 1.0
    cancel: threading.Event = field(default_factory=threading.Event)
    meshes: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return str(self.settings.get("kind", ""))


class App:
    """Everything the page can ask for. One study runs at a time; more queue."""

    def __init__(self) -> None:
        self.parts: dict[str, Part] = {}
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whittle-study")

    def info(self) -> dict[str, Any]:
        return {
            "materials": {
                key: {
                    "name": m.name,
                    "modulus": m.youngs_modulus,
                    "strength": m.strength,
                    "density": m.density,
                }
                for key, m in MATERIALS.items()
            },
            "examples": [
                {"id": p.id, "name": p.name, "description": p.description, "kind": p.setup["kind"]}
                for p in PRESETS
            ],
            "units": list(UNITS),
        }

    # -- parts ---------------------------------------------------------------

    def upload(self, data: bytes, name: str, units: str = "mm") -> dict[str, Any]:
        suffix = Path(name).suffix.lower()
        if suffix not in (".stl", ".obj"):
            raise BadRequest("upload an .stl or .obj file")
        if units not in UNITS:
            raise BadRequest(f"units must be one of {', '.join(UNITS)}")
        try:
            mesh = io.parse(data, suffix)
        except Exception as error:  # a malformed file can fail in many ways
            raise BadRequest(f"could not read {name}: {error}") from error
        if not len(mesh.faces):
            raise BadRequest(f"{name} has no triangles")
        mesh = Mesh(mesh.vertices * UNITS[units], mesh.faces)
        return self._add(name, mesh)

    def example(self, preset_id: str) -> dict[str, Any]:
        try:
            chosen = preset(preset_id)
        except KeyError:
            raise BadRequest(f"no example called {preset_id!r}") from None
        payload = self._add(chosen.name, chosen.build())
        payload["setup"] = chosen.setup
        return payload

    def _add(self, name: str, mesh: Mesh) -> dict[str, Any]:
        part = Part(uuid.uuid4().hex[:12], name, mesh)
        with self._lock:
            self.parts[part.id] = part
        lo, hi = mesh.bounds
        return {
            "id": part.id,
            "name": name,
            "mesh": mesh_payload(mesh),
            "min": lo.tolist(),
            "max": hi.tolist(),
            "triangles": len(mesh.faces),
            "closed": is_closed(mesh),
        }

    def _part(self, part_id: object) -> Part:
        part = self.parts.get(str(part_id))
        if part is None:
            raise BadRequest("that part is no longer loaded -- load it again")
        return part

    # -- placing things --------------------------------------------------------

    def preview(self, body: dict[str, Any]) -> dict[str, Any]:
        """The voxels a placed support, load or keep would select, to draw them."""
        part = self._part(body.get("part"))
        grid, solid = part.voxels(_int(body, "resolution", 48, 8, 128))
        study = LoadStudy.from_voxels(grid, solid)
        region = study.region(**_region(body))
        centres = grid.centers()[region].astype(np.float32)
        return {"points": _b64(centres), "count": int(region.sum()), "spacing": grid.spacing}

    # -- studies -----------------------------------------------------------------

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        part = self._part(body.get("part"))
        if body.get("kind") not in ("load", "flow", "dual"):
            raise BadRequest("kind must be 'load', 'flow' or 'dual'")
        if body["kind"] == "load" and (not body.get("supports") or not body.get("loads")):
            raise BadRequest("place at least one support and one load first")
        if body["kind"] == "dual" and not body.get("supports"):
            raise BadRequest("place at least one support first -- where the part is held")
        job = Job(uuid.uuid4().hex[:12], part, body)
        with self._lock:
            self.jobs[job.id] = job
        self._worker.submit(self._run, job)
        return self.status(job.id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        job.cancel.set()
        return self.status(job_id)

    def _run(self, job: Job) -> None:
        if job.cancel.is_set():
            job.state = "cancelled"
            return
        job.state, job.started = "running", time.time()

        def progress(fraction: float) -> None:
            if job.cancel.is_set():
                raise Cancelled
            job.progress = max(job.progress, float(fraction))

        try:
            if job.kind == "load":
                self._run_load(job, progress)
            elif job.kind == "flow":
                self._run_flow(job, progress)
            else:
                self._run_dual(job, progress)
            job.state, job.progress = "done", 1.0
        except Cancelled:
            job.state = "cancelled"
        except BadRequest as error:
            job.state, job.error = "failed", str(error)
        except Exception as error:  # surface anything else to the page, not just the log
            job.state, job.error = "failed", f"{type(error).__name__}: {error}"
            traceback.print_exc()
        finally:
            job.finished = time.time()

    def _run_load(self, job: Job, progress: Callable[[float], None]) -> None:
        s = job.settings
        resolution = _int(s, "resolution", 48, 8, 128)
        grid, solid = job.part.voxels(resolution)
        material = str(s.get("material", "pla"))
        if material not in MATERIALS:
            raise BadRequest(f"unknown material {material!r}")
        study = LoadStudy.from_voxels(
            grid, solid, material=material, coverage=job.part.coverage(resolution)
        )
        try:
            for item in s.get("supports", []):
                study.support(**_region(item))
            for item in s.get("loads", []):
                study.load(_vector(item, "force"), **_region(item))
            for item in s.get("keeps", []):
                study.keep(**_region(item))
            for item in s.get("surfaces", []):
                study.whittle(**_region(item))
        except ValueError as error:
            raise BadRequest(str(error)) from error
        job.study = study
        job.sweep = study.sweep(
            remove_up_to=_float(s, "removeUpTo", 0.6, 0.05, 0.95),
            stations=_int(s, "stations", 7, 2, 16),
            progress=progress,
        )
        job.stations = list(job.sweep)
        job.heat_max = _peak(job.stations)

    def _run_flow(self, job: Job, progress: Callable[[float], None]) -> None:
        s = job.settings
        try:
            study = FlowStudy(
                job.part.mesh,
                flow=_vector(s, "flow"),
                resolution=_int(s, "resolution", 40, 12, 96),
                detail=_int(s, "detail", 2, 1, 4),
                mirror=_axis(s.get("mirror")),
                flat=[str(side) for side in s.get("flat", [])],
                extruded=_axis(s.get("extruded")),
                speed_kmh=_float(s, "speedKmh", 100.0, 1.0, 400.0),
            )
            for box in s.get("keepBoxes", []):
                study.keep(_box(box))
            for item in s.get("surfaces", []):
                study.whittle(**_region(item))
        except ValueError as error:
            raise BadRequest(str(error)) from error
        job.study = study
        job.sweep = study.sweep(
            remove_up_to=_float(s, "removeUpTo", 0.25, 0.02, 0.8),
            stations=_int(s, "stations", 6, 2, 12),
            effort=_float(s, "effort", 1.0, 0.1, 4.0),
            progress=progress,
        )
        job.stations = list(job.sweep)
        original = job.sweep.original
        heat = _drag_heat(original.mesh, original.analysis)
        job.heat_max = float(np.percentile(np.abs(heat), 99)) or 1.0

    def _run_dual(self, job: Job, progress: Callable[[float], None]) -> None:
        s = job.settings
        material = str(s.get("material", "pla"))
        if material not in MATERIALS:
            raise BadRequest(f"unknown material {material!r}")
        try:
            study = DualStudy(
                job.part.mesh,
                flow=_vector(s, "flow"),
                material=material,
                speed_kmh=_float(s, "speedKmh", 110.0, 1.0, 400.0),
                crosswind_kmh=_float(s, "crosswindKmh", 40.0, 0.0, 200.0),
                resolution=_int(s, "resolution", 40, 12, 96),
                structure_resolution=_int(s, "structureResolution", 48, 8, 128),
                detail=_int(s, "detail", 2, 1, 4),
                mirror=_axis(s.get("mirror")),
                flat=[str(side) for side in s.get("flat", [])],
                skin=_int(s, "skin", 1, 1, 4),
            )
            for item in s.get("supports", []):
                study.support(**_region(item))
            for item in s.get("loads", []):
                study.load(_vector(item, "force"), **_region(item))
            for cargo in s.get("cargo", []):
                study.cargo(_box(cargo.get("box")), kg=_float(cargo, "kg", 0.0, 0.0, 1e5))
            for item in s.get("surfaces", []):
                study.whittle(**_region(item))
        except ValueError as error:
            raise BadRequest(str(error)) from error
        job.study = study
        job.dual = study.sweep(
            remove_up_to=_float(s, "removeUpTo", 0.25, 0.02, 0.8),
            aero_stations=_int(s, "aeroStations", 5, 2, 8),
            lighten_up_to=_float(s, "lightenUpTo", 0.6, 0.05, 0.95),
            weight_stations=_int(s, "weightStations", 5, 2, 10),
            effort=_float(s, "effort", 1.0, 0.1, 4.0),
            progress=progress,
        )
        job.stations = [station for _, _, station in job.dual.designs()]
        job.heat_max = _peak(job.stations)

    def status(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        elapsed = (job.finished or time.time()) - job.started if job.started else 0.0
        payload: dict[str, Any] = {
            "id": job.id,
            "kind": job.kind,
            "state": job.state,
            "progress": job.progress,
            "error": job.error,
            "elapsed": elapsed,
        }
        if job.state == "done" and job.dual is not None:
            payload["result"] = self._dual_result(job, job.dual)
        elif job.state == "done" and job.sweep is not None:
            payload["result"] = self._result(job, job.sweep)
        return payload

    def _dual_result(self, job: Job, dual: DualSweep) -> dict[str, Any]:
        assert isinstance(job.study, DualStudy)
        rows: list[dict[str, Any]] = []
        stations: list[dict[str, Any]] = []
        for i, row in enumerate(dual.rows):
            rows.append(
                {"removed": row.removed, "drag": row.drag, "note": row.note, "first": len(stations)}
            )
            for j, station in enumerate(row.stations):
                stations.append(
                    {
                        "row": i,
                        "column": j,
                        "removed": station.removed,
                        "metrics": _finite(station.metrics),
                    }
                )
        return {
            "dual": True,
            "columns": [
                {"key": c.key, "label": c.label, "format": c.format} for c in dual.columns
            ],
            "rows": rows,
            "stations": stations,
            "note": "",
            "heatMax": job.heat_max,
            "heat": {"label": "von Mises stress", "unit": "MPa", "signed": False},
            "strength": job.study.material.strength,
        }

    def _result(self, job: Job, sweep: Sweep) -> dict[str, Any]:
        result: dict[str, Any] = {
            "columns": [
                {"key": c.key, "label": c.label, "format": c.format} for c in sweep.columns
            ],
            "stations": [
                {"removed": st.removed, "metrics": _finite(st.metrics)} for st in sweep.stations
            ],
            "note": sweep.note,
            "heatMax": job.heat_max,
        }
        if isinstance(job.study, LoadStudy):
            result["heat"] = {"label": "von Mises stress", "unit": "MPa", "signed": False}
            result["strength"] = job.study.material.strength
        else:
            result["heat"] = {"label": "drag contribution", "unit": "Cp", "signed": True}
        return result

    def station_mesh(self, job_id: str, index: int) -> dict[str, Any]:
        job = self._job(job_id)
        station = self._station(job, index)
        if index not in job.meshes:
            mesh = station.mesh
            if isinstance(station.analysis, FEAResult):
                assert isinstance(job.study, LoadStudy | DualStudy)
                heat = _stress_heat(mesh, station.analysis, job.study.grid)
            elif isinstance(station.analysis, DragReport):
                heat = _drag_heat(mesh, station.analysis)
            else:
                heat = None
            job.meshes[index] = mesh_payload(mesh, heat)
        return job.meshes[index]

    def shape_mesh(self, job_id: str, row: int) -> dict[str, Any]:
        """The carved outside of one row of a dual map, with its drag heatmap."""
        job = self._job(job_id)
        if job.dual is None:
            raise BadRequest("only a dual study has carved shapes")
        if not 0 <= row < len(job.dual.rows):
            raise BadRequest("no such shape")
        shape = job.dual.rows[row]
        return mesh_payload(shape.shape, _drag_heat(shape.shape, shape.report))

    def station_stl(self, job_id: str, index: int) -> tuple[bytes, str]:
        job = self._job(job_id)
        station = self._station(job, index)
        stem = Path(job.part.name).stem.replace(" ", "_") or "part"
        return io.stl_bytes(station.mesh), f"{stem}-{round(station.removed * 100):02d}pct.stl"

    def _job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise BadRequest("no such study")
        return job

    def _station(self, job: Job, index: int) -> Any:
        if not job.stations:
            raise BadRequest("that study has not finished")
        if not 0 <= index < len(job.stations):
            raise BadRequest("no such station")
        return job.stations[index]


# -- payload helpers -------------------------------------------------------------


def mesh_payload(mesh: Mesh, heat: FloatArray | None = None) -> dict[str, Any]:
    """A mesh for the page: base64 typed arrays, which three.js takes as they are."""
    return {
        "positions": _b64(mesh.vertices.astype(np.float32)),
        "indices": _b64(mesh.faces.astype(np.uint32)),
        "heat": None if heat is None else _b64(np.asarray(heat, dtype=np.float32)),
    }


def is_closed(mesh: Mesh) -> bool:
    """Whether every edge has exactly two faces -- what voxelizing needs."""
    f = mesh.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _stress_heat(mesh: Mesh, result: FEAResult, grid: Grid) -> FloatArray:
    """Per vertex, the worst stress among the voxels around it."""
    index = (mesh.vertices - np.asarray(grid.origin)) / grid.spacing
    low = np.floor(index).astype(np.int64)
    shape = np.asarray(grid.shape)
    heat = np.zeros(len(mesh.vertices), dtype=np.float64)
    for corner in np.ndindex(2, 2, 2):
        cell = low + np.asarray(corner)
        inside = np.all((cell >= 0) & (cell < shape), axis=1)
        values = np.zeros(len(cell))
        c = cell[inside]
        values[inside] = result.von_mises[c[:, 0], c[:, 1], c[:, 2]]
        heat = np.maximum(heat, values)
    return heat


def _drag_heat(mesh: Mesh, report: object) -> FloatArray:
    """Per vertex, the drag per unit area of the faces around it."""
    if not isinstance(report, DragReport):
        return np.zeros(len(mesh.vertices))
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    density = report.per_face / np.where(area > 0, area, 1.0)
    n = len(mesh.vertices)
    flat = mesh.faces.ravel()
    weight = np.repeat(area, 3)
    total = np.bincount(flat, weights=np.repeat(density, 3) * weight, minlength=n)
    covered = np.bincount(flat, weights=weight, minlength=n)
    result: FloatArray = total / np.where(covered > 0, covered, 1.0)
    return result


def _peak(stations: list[Any]) -> float:
    """The worst peak stress in a run: the top of every heatmap's scale, so slides compare."""
    peaks = [st.metrics["peak_stress_mpa"] for st in stations]
    finite = [p for p in peaks if np.isfinite(p)]
    return float(max(finite)) if finite else 1.0


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _finite(metrics: dict[str, float]) -> dict[str, float | None]:
    return {k: (float(v) if np.isfinite(v) else None) for k, v in metrics.items()}


def _vector(body: dict[str, Any], key: str) -> Vector3:
    value = body.get(key)
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise BadRequest(f"{key} must be three numbers")
    try:
        x, y, z = (float(v) for v in value)
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be three numbers") from None
    return (x, y, z)


def _box(value: object) -> Box:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise BadRequest("a box is two corners")
    return (_vector({"c": value[0]}, "c"), _vector({"c": value[1]}, "c"))


def _region(item: dict[str, Any]) -> dict[str, Any]:
    """The placement keywords of a support, load or keep, as the page sends it."""
    if item.get("box") is not None:
        return {"box": _box(item["box"])}
    region: dict[str, Any] = {"at": _vector(item, "at")}
    if item.get("normal") is not None:
        region["normal"] = _vector(item, "normal")
    for key in ("radius", "depth"):
        if item.get(key) is not None:
            region[key] = _float(item, key, 1.0, 0.0, 1e9)
    return region


def _axis(value: object) -> str | None:
    if value in (None, "", "none"):
        return None
    if value not in ("x", "y", "z"):
        raise BadRequest(f"axis must be x, y or z, got {value!r}")
    return str(value)


def _int(body: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(body.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a whole number") from None
    return max(low, min(high, value))


def _float(body: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(body.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a number") from None
    if not np.isfinite(value):
        raise BadRequest(f"{key} must be finite")
    return max(low, min(high, value))


# -- HTTP --------------------------------------------------------------------------


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "whittle"

        def log_message(self, format: str, *args: Any) -> None:
            # The page polls every half second; logging every poll buries the
            # one line that matters. Studies that fail print their traceback.
            return

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                self._route(method, parts, parse_qs(url.query))
            except BadRequest as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                traceback.print_exc()
                message = {"error": f"{type(error).__name__}: {error}"}
                self._json(message, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _route(self, method: str, parts: list[str], query: dict[str, list[str]]) -> None:
            if method == "GET" and not parts:
                return self._static("index.html")
            if method == "GET" and parts[0] == "static" and len(parts) == 2:
                return self._static(parts[1])
            if parts[:1] != ["api"]:
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            route = parts[1:]
            if method == "GET" and route == ["info"]:
                return self._json(app.info())
            if method == "POST" and route == ["parts"]:
                name = query.get("name", ["part.stl"])[0]
                units = query.get("units", ["mm"])[0]
                return self._json(app.upload(self._body(MAX_UPLOAD), name, units))
            if method == "POST" and len(route) == 2 and route[0] == "examples":
                return self._json(app.example(route[1]))
            if method == "POST" and route == ["preview"]:
                return self._json(app.preview(self._json_body()))
            if method == "POST" and route == ["jobs"]:
                return self._json(app.start(self._json_body()))
            if len(route) >= 2 and route[0] == "jobs":
                job_id = route[1]
                if method == "GET" and len(route) == 2:
                    return self._json(app.status(job_id))
                if method == "POST" and route[2:] == ["cancel"]:
                    return self._json(app.cancel(job_id))
                if method == "GET" and len(route) >= 4 and route[2] == "stations":
                    index = _int({"i": route[3]}, "i", 0, -1, 10_000)
                    if route[4:] == ["stl"]:
                        data, filename = app.station_stl(job_id, index)
                        return self._send(data, "model/stl", filename)
                    if len(route) == 4:
                        return self._json(app.station_mesh(job_id, index))
                if method == "GET" and len(route) == 4 and route[2] == "shapes":
                    row = _int({"i": route[3]}, "i", 0, -1, 10_000)
                    return self._json(app.shape_mesh(job_id, row))
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _body(self, limit: int) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > limit:
                raise BadRequest(f"that file is over {limit // (1024 * 1024)} MB")
            return self.rfile.read(length)

        def _json_body(self) -> dict[str, Any]:
            try:
                body = json.loads(self._body(10 * 1024 * 1024) or b"{}")
            except json.JSONDecodeError as error:
                raise BadRequest(f"bad JSON: {error}") from error
            if not isinstance(body, dict):
                raise BadRequest("expected a JSON object")
            return body

        def _static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if path.parent != STATIC.resolve() or not path.is_file():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            kinds = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
            self._send(path.read_bytes(), kinds.get(path.suffix, "application/octet-stream"))

        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json", status=status)

        def _send(
            self,
            data: bytes,
            content_type: str,
            filename: str | None = None,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if filename is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """A server for a fresh :class:`App`, bound but not yet serving."""
    server = ThreadingHTTPServer((host, port), make_handler(App()))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="whittle-gui", description="Lighten or streamline a part in the browser."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)
    server = serve(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"whittle GUI at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
