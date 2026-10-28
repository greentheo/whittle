from __future__ import annotations

import numpy as np
import pytest

from whittle import DragModel, FlowStudy, Mesh, ParametricStudy, shapes
from whittle.surface import split_long_edges, weld

FLOW = (-1.0, 0.0, 0.0)


def _open_edges(mesh: Mesh) -> int:
    f = mesh.faces
    ends = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(ends, axis=0, return_counts=True)
    return int((counts != 2).sum())


def _cad_box(size: tuple[float, float, float], *, jitter: float = 1e-7) -> Mesh:
    """A box the way CAD tessellates one: two triangles a face, every face with its own corners."""
    rng = np.random.default_rng(0)
    half = np.asarray(size) / 2.0
    vertices, faces = [], []
    for axis in range(3):
        u, v = (a for a in range(3) if a != axis)
        for side in (-1.0, 1.0):
            base = len(vertices)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u], p[v] = side, su, sv
                vertices.append(p * half + rng.normal(0.0, jitter, 3))
            quad = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            a, b, c = (vertices[i] for i in quad[0])
            if np.cross(b - a, c - a)[axis] * side < 0:
                quad = [q[::-1] for q in quad]
            faces += quad
    return Mesh(np.array(vertices), np.array(faces, dtype=np.int64))


def _pod(p: dict[str, float]) -> Mesh:
    """A 2 m box whose tail, at -x, draws in by ``squeeze`` over its last ``tail`` mm."""
    box = shapes.box((2000.0, 900.0, 400.0), cell=100.0)
    v = box.vertices.copy()
    t = np.clip((-1000.0 + p["tail"] - v[:, 0]) / p["tail"], 0.0, 1.0)
    scale = 1.0 - p["squeeze"] * t
    v[:, 1] *= scale
    v[:, 2] *= scale
    return Mesh(v, box.faces)


START = {"tail": 400.0, "squeeze": 0.0}


def test_splitting_long_edges_keeps_the_shape_and_closes_up() -> None:
    box = shapes.box((991.0, 1880.0, 1041.0), cell=2000.0)
    assert len(box.faces) == 12
    split = split_long_edges(box, 60.0)
    lengths = np.linalg.norm(
        split.vertices[split.faces[:, [1, 2, 0]]] - split.vertices[split.faces], axis=2
    )
    assert lengths.max() <= 60.0
    assert _open_edges(split) == 0
    assert split.volume == pytest.approx(box.volume, rel=1e-9)
    assert split.area == pytest.approx(box.area, rel=1e-9)


def test_welding_finds_seam_copies_that_rounding_would_miss() -> None:
    cad = _cad_box((991.0, 1880.0, 1041.0))
    assert len(cad.vertices) == 24 and _open_edges(cad) > 0
    welded = weld(cad, 1e-3)
    assert len(welded.vertices) == 8
    assert _open_edges(welded) == 0


def test_a_cad_tessellation_scores_like_a_fine_mesh_once_prepared() -> None:
    wind = DragModel(FLOW)
    fine = wind(shapes.box((991.0, 1880.0, 1041.0), cell=40.0)).cd
    prepared = split_long_edges(weld(_cad_box((991.0, 1880.0, 1041.0)), 1e-3), 94.0)
    assert wind(prepared).cd == pytest.approx(fine, rel=0.01)
    # Unwelded, every face is a separate body, and the answer is wrong.
    assert wind(_cad_box((991.0, 1880.0, 1041.0))).cd != pytest.approx(fine, rel=0.03)


def test_the_builder_may_return_raw_arrays() -> None:
    def build(p: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        cad = _cad_box((p["length"], 900.0, 400.0))
        return cad.vertices, cad.faces

    study = ParametricStudy(
        build, {"length": (1000.0, 2000.0)}, start={"length": 2000.0}, flow=FLOW
    )
    assert _open_edges(study.base) == 0
    assert study.base.volume == pytest.approx(2000.0 * 900.0 * 400.0, rel=1e-6)


def test_obvious_mistakes_are_caught_up_front() -> None:
    with pytest.raises(ValueError, match="no value"):
        ParametricStudy(_pod, {"tail": (100, 800), "nose": (0, 1)}, start=START, flow=FLOW)
    with pytest.raises(ValueError, match="outside its bounds"):
        ParametricStudy(_pod, {"tail": (500, 800)}, start=START, flow=FLOW)
    with pytest.raises(ValueError, match="metric name"):
        ParametricStudy(_pod, {"drag": (0, 1)}, start={**START, "drag": 0.5}, flow=FLOW)

    def broken(p: dict[str, float]) -> Mesh:
        raise RuntimeError("fillet too big")

    with pytest.raises(ValueError, match="fillet too big"):
        ParametricStudy(broken, {"tail": (100, 800)}, start=START, flow=FLOW)

    def open_box(p: dict[str, float]) -> Mesh:
        box = shapes.box((1000.0, 500.0, 300.0))
        return Mesh(box.vertices, box.faces[1:])

    with pytest.raises(ValueError, match="open edges"):
        ParametricStudy(open_box, {"tail": (100, 800)}, start=START, flow=FLOW)


def test_a_sweep_tunes_the_parameters_and_hands_them_back() -> None:
    study = ParametricStudy(
        _pod, {"squeeze": (0.0, 0.6)}, start=START, flow=FLOW, speed_kmh=110.0
    )
    sweep = study.sweep(remove_up_to=0.1, stations=3, effort=0.3)
    assert len(sweep) == 3
    assert sweep.original.design == START
    assert sweep.original.metrics["drag"] == pytest.approx(1.0)
    drags = [s.metrics["drag"] for s in sweep]
    assert drags == sorted(drags, reverse=True)
    assert drags[-1] < 0.97
    last = sweep[-1]
    assert last.design["tail"] == START["tail"]
    assert 0.0 < last.design["squeeze"] <= 0.6
    assert last.metrics["squeeze"] == last.design["squeeze"]
    assert last.removed <= 0.1 + 0.01
    assert "squeeze" in sweep.table()
    # The station's mesh is the design, and its scoring is the same as asking for it.
    assert study.score(last.design).drag_area == pytest.approx(
        last.analysis.drag_area, rel=1e-9  # type: ignore[attr-defined]
    )


def test_designs_that_will_not_build_are_skipped() -> None:
    def fussy(p: dict[str, float]) -> Mesh:
        if p["squeeze"] > 0.3:
            raise ValueError("would crack the foam")
        return _pod(p)

    study = ParametricStudy(fussy, {"squeeze": (0.0, 0.6)}, start=START, flow=FLOW)
    sweep = study.sweep(remove_up_to=0.1, stations=2, effort=0.3)
    assert sweep[-1].design["squeeze"] <= 0.3
    assert "did not build" in sweep.note and "crack the foam" in sweep.note


def test_a_sweep_says_when_the_bounds_run_out() -> None:
    study = ParametricStudy(_pod, {"squeeze": (0.0, 0.05)}, start=START, flow=FLOW)
    sweep = study.sweep(remove_up_to=0.3, stations=3, effort=0.2)
    assert sweep[-1].removed < 0.05
    assert "stopped short" in sweep.note


def test_a_keep_box_holds() -> None:
    study = ParametricStudy(_pod, {"squeeze": (0.0, 0.6)}, start=START, flow=FLOW)
    # Cargo filling the tail end, nearly to its sides.
    study.keep(((-1000.0, -400.0, -180.0), (-700.0, 400.0, 180.0)))
    sweep = study.sweep(remove_up_to=0.1, stations=2, effort=0.3)
    assert sweep[-1].design["squeeze"] <= 0.15


# Obstacles: a box 457 mm behind a car-sized slab, with the car driving towards +x.
CAR = shapes.box((1500.0, 1770.0, 1400.0), center=(750.0, 0.0, 850.0), cell=2000.0)
HITCH_BOX = shapes.box((991.0, 1880.0, 1041.0), center=(-952.5, 0.0, 977.5), cell=40.0)


def test_an_obstacle_shelters_what_is_behind_it() -> None:
    alone = DragModel(FLOW)(HITCH_BOX)
    behind = DragModel(FLOW, obstacles=[CAR])(HITCH_BOX)
    assert behind.drag_area < 0.4 * alone.drag_area
    # The front is hidden, the wake behind the box is not.
    assert behind.forebody < 0.1 * alone.forebody
    assert behind.base == pytest.approx(alone.base)
    # Its own size does not change: Cd stays a coefficient on its own frontal area.
    assert behind.frontal_area == pytest.approx(alone.frontal_area)


def test_an_obstacle_shelters_only_downstream_and_only_its_own_silhouette() -> None:
    alone = DragModel(FLOW)(HITCH_BOX).drag_area
    downstream = shapes.box((1500.0, 1770.0, 1400.0), center=(-3000.0, 0.0, 850.0), cell=2000.0)
    assert DragModel(FLOW, obstacles=[downstream])(HITCH_BOX).drag_area == pytest.approx(alone)
    narrow = shapes.box((1500.0, 1200.0, 1400.0), center=(750.0, 0.0, 850.0), cell=2000.0)
    wide = DragModel(FLOW, obstacles=[CAR])(HITCH_BOX).drag_area
    assert wide < DragModel(FLOW, obstacles=[narrow])(HITCH_BOX).drag_area < alone


def test_an_obstacle_shelters_the_same_however_it_is_tessellated() -> None:
    coarse = DragModel(FLOW, obstacles=[CAR])(HITCH_BOX).drag_area
    fine_car = shapes.box((1500.0, 1770.0, 1400.0), center=(750.0, 0.0, 850.0), cell=50.0)
    fine = DragModel(FLOW, obstacles=[fine_car])(HITCH_BOX).drag_area
    assert coarse == pytest.approx(fine, rel=0.02)


def test_a_flow_study_takes_obstacles_too() -> None:
    free = FlowStudy(HITCH_BOX, flow=FLOW, resolution=20).analyze()
    behind = FlowStudy(HITCH_BOX, flow=FLOW, resolution=20, obstacles=[CAR]).analyze()
    assert behind.drag_area < 0.5 * free.drag_area
