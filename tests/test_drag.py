from __future__ import annotations

import numpy as np
import pytest

from whittle import DragModel, Grid, shapes
from whittle.types import Mesh

WIND = DragModel((1.0, 0.0, 0.0))


def _cd(mesh: Mesh) -> float:
    return WIND(mesh).cd


def _join(*meshes: Mesh) -> Mesh:
    vertices, faces, offset = [], [], 0
    for m in meshes:
        vertices.append(m.vertices)
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    return Mesh(np.concatenate(vertices), np.concatenate(faces))


def _taper_tail(mesh: Mesh, *, start: float, end: float, squeeze: float) -> Mesh:
    """Shrink a box's cross-section linearly towards its downstream end."""
    v = mesh.vertices.copy()
    t = np.clip((v[:, 0] - start) / (end - start), 0.0, 1.0)
    scale = 1.0 - squeeze * t
    v[:, 1] *= scale
    v[:, 2] *= scale
    return Mesh(v, mesh.faces)


# Reference values are high-Reynolds-number textbook figures (Hoerner): the
# regime of a roof box at highway speed, not of a model in a water tank.
def test_flat_plate_face_on() -> None:
    assert 1.1 < _cd(shapes.box((0.02, 1.0, 1.0), cell=0.05)) < 1.4


def test_cube() -> None:
    assert 0.95 < _cd(shapes.box((1.0, 1.0, 1.0), cell=0.05)) < 1.2


def test_sphere_above_its_drag_crisis() -> None:
    assert 0.12 < _cd(shapes.sphere(0.5, subdivisions=4)) < 0.3


def test_slender_streamlined_body() -> None:
    assert _cd(shapes.ellipsoid((2.0, 0.5, 0.5), subdivisions=4)) < 0.15


def test_ranking_of_classic_shapes() -> None:
    plate = _cd(shapes.box((0.02, 1.0, 1.0), cell=0.05))
    cube = _cd(shapes.box((1.0, 1.0, 1.0), cell=0.05))
    sphere = _cd(shapes.sphere(0.5, subdivisions=4))
    teardrop = _cd(shapes.ellipsoid((2.0, 0.5, 0.5), subdivisions=4))
    assert plate > cube > sphere > teardrop


def test_tessellation_does_not_change_the_answer() -> None:
    coarse = _cd(shapes.sphere(0.5, subdivisions=2))
    fine = _cd(shapes.sphere(0.5, subdivisions=5))
    assert coarse == pytest.approx(fine, rel=0.05)


def test_a_relaxed_voxel_remesh_scores_close_to_the_true_shape() -> None:
    grid = Grid((40, 40, 40), 1.0 / 36.0, (-0.54, -0.54, -0.54))
    solid = np.linalg.norm(grid.centers(), axis=-1) <= 0.5
    remeshed = _cd(grid.mesh(solid, relax=20))
    assert remeshed == pytest.approx(_cd(shapes.sphere(0.5, subdivisions=4)), abs=0.08)


def test_rounding_the_nose_of_a_box_helps() -> None:
    grid = Grid((90, 30, 30), 1.0 / 30.0, (-1.5, -0.5, -0.5))
    c = grid.centers()
    box = (np.abs(c[..., 0]) <= 1.5) & (np.abs(c[..., 1]) <= 0.5) & (np.abs(c[..., 2]) <= 0.5)
    r = 0.2
    core = np.stack(
        [
            np.clip(c[..., 0], -1.5 + r, None),
            np.clip(c[..., 1], -0.5 + r, 0.5 - r),
            np.clip(c[..., 2], -0.5 + r, 0.5 - r),
        ],
        axis=-1,
    )
    rounded = box & (np.linalg.norm(c - core, axis=-1) <= r)
    assert _cd(grid.mesh(rounded, relax=10)) < 0.85 * _cd(grid.mesh(box, relax=10))


def test_a_gentle_tail_taper_beats_a_square_back_and_a_steep_one() -> None:
    box = shapes.box((3.0, 1.0, 1.0), cell=0.05)
    square = WIND(box).drag_area
    gentle = WIND(_taper_tail(box, start=0.0, end=1.5, squeeze=0.4)).drag_area  # ~8 degrees
    steep = WIND(_taper_tail(box, start=1.2, end=1.5, squeeze=0.4)).drag_area  # ~34 degrees
    assert gentle < steep < square


def test_a_domed_front_beats_a_dished_one() -> None:
    box = shapes.box((3.0, 1.0, 1.0), cell=0.05)
    v = box.vertices
    radial = (v[:, 1] ** 2 + v[:, 2] ** 2) / 0.5
    nose = np.clip(1.0 - (v[:, 0] + 1.5) / 0.6, 0.0, 1.0)
    dome = v.copy()
    dome[:, 0] += 0.3 * radial * nose
    dish = v.copy()
    dish[:, 0] += 0.3 * (1.0 - radial) * nose
    assert _cd(Mesh(dome, box.faces)) < _cd(Mesh(dish, box.faces))


def test_a_body_in_another_ones_wake_is_sheltered() -> None:
    lead = shapes.box((1.0, 1.0, 1.0), cell=0.05)
    trail = shapes.box((1.0, 1.0, 1.0), center=(3.0, 0.0, 0.0), cell=0.05)
    alone = WIND(lead).drag_area
    together = WIND(_join(lead, trail)).drag_area
    # The trailing cube adds its wake and friction, but nothing like a second cube.
    assert together < 1.4 * alone
    side_by_side = WIND(_join(lead, shapes.box((1, 1, 1), center=(0, 3, 0), cell=0.05)))
    assert side_by_side.drag_area == pytest.approx(2.0 * alone, rel=0.02)


def test_turning_the_part_and_the_wind_together_changes_nothing() -> None:
    body = shapes.ellipsoid((1.5, 0.5, 0.3), subdivisions=3)
    angle = np.radians(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    turned = Mesh(body.vertices @ rotation.T, body.faces)
    wind = tuple(rotation @ np.array([1.0, 0.0, 0.0]))
    assert DragModel(wind)(turned).cd == pytest.approx(WIND(body).cd, rel=0.03)


def test_drag_follows_the_wind_direction() -> None:
    body = shapes.ellipsoid((1.5, 0.5, 0.5), subdivisions=3)
    end_on = DragModel((1.0, 0.0, 0.0))(body).drag_area
    broadside = DragModel((0.0, 1.0, 0.0))(body).drag_area
    assert broadside > 2.0 * end_on


def test_breakdown_adds_up() -> None:
    report = WIND(shapes.box((2.0, 1.0, 1.0), cell=0.1))
    assert report.per_face.sum() == pytest.approx(report.drag_area)
    assert report.forebody + report.base + report.friction == pytest.approx(report.drag_area)
    assert report.cd == pytest.approx(report.drag_area / report.frontal_area)


def test_force_in_newtons() -> None:
    # A 1 m^2 plate, in millimetres, at 100 km/h.
    report = WIND(shapes.box((20.0, 1000.0, 1000.0), cell=50.0))
    speed = 100.0 / 3.6
    assert report.force(speed) == pytest.approx(0.5 * 1.225 * speed**2 * report.cd, rel=1e-3)
    assert report.force(2 * speed) == pytest.approx(4 * report.force(speed))


def test_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match=r"non-zero"):
        DragModel((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"empty"):
        WIND(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))
    box = shapes.box((1.0, 1.0, 1.0), cell=0.5)
    inside_out = Mesh(box.vertices, np.ascontiguousarray(box.faces[:, ::-1]))
    # Inside out, the back face looks windward and the front face shelters it.
    with pytest.raises(ValueError, match=r"inside out"):
        WIND(inside_out)


def test_long_plates_follow_hoerner() -> None:
    cds = [_cd(shapes.box((0.02, float(ar), 1.0), cell=0.05)) for ar in (1, 10, 20)]
    # Hoerner: 1.18, 1.3, 1.5 -- the wake gets stronger as the plate gets longer.
    assert cds[0] < cds[1] < cds[2]
    assert 1.35 < cds[2] < 1.7


def test_an_endless_slice() -> None:
    endless = DragModel((1.0, 0.0, 0.0), aspect=float("inf"), ends=1)
    # Hoerner's infinitely long plate: 1.98.
    assert 1.8 < endless(shapes.box((0.02, 1.0, 1.0), cell=0.05)).cd < 2.2
    # And a slice's drag per unit length does not depend on how long a slice was cut.
    short = endless(shapes.box((1.0, 1.0, 0.6), cell=0.05)).drag_area
    long = endless(shapes.box((1.0, 2.0, 0.6), cell=0.05)).drag_area
    assert long == pytest.approx(2.0 * short, rel=0.02)
    with pytest.raises(ValueError, match=r"at least 1"):
        DragModel((1.0, 0.0, 0.0), aspect=0.5)
