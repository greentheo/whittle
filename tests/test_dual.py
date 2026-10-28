from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from whittle import DragModel, DualStudy, DualSweep, Grid, shapes
from whittle.dual import wind_forces

CARGO = ((-100.0, -40.0, 12.0), (100.0, 40.0, 45.0))


@pytest.fixture(scope="module")
def pod() -> tuple[DualStudy, DualSweep]:
    """A small roof pod: two bars underneath, a cargo space, wind from the front."""
    box = shapes.box((300.0, 120.0, 60.0), center=(0.0, 0.0, 30.0), cell=10.0)
    study = DualStudy(
        box,
        flow=(-1.0, 0.0, 0.0),
        material="xps",
        resolution=20,
        structure_resolution=30,
        mirror="y",
        flat=("-z",),
    )
    study.support(box=((50.0, -70.0, -5.0), (70.0, 70.0, 8.0)))
    study.support(box=((-70.0, -70.0, -5.0), (-50.0, 70.0, 8.0)))
    study.cargo(CARGO, kg=2.0)
    return study, study.sweep(aero_stations=3, weight_stations=3, effort=0.2)


def test_rows_run_from_the_original_to_the_most_carved(pod: tuple[DualStudy, DualSweep]) -> None:
    _, sweep = pod
    assert len(sweep.rows) == 3
    removed = [row.removed for row in sweep.rows]
    assert removed == sorted(removed) and removed[0] == 0.0
    drags = [row.drag for row in sweep.rows]
    assert drags[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-9 for a, b in zip(drags, drags[1:], strict=False))
    assert drags[-1] < drags[0]


def test_each_row_lightens_inside_its_own_shape(pod: tuple[DualStudy, DualSweep]) -> None:
    study, sweep = pod
    cargo = study.grid.box(*CARGO)
    for row in sweep.rows:
        masses = [s.metrics["mass_g"] for s in row.stations]
        assert masses == sorted(masses, reverse=True)
        assert len(row.stations) >= 2
        occupancy = study.grid.voxelize(row.shape)
        for station in row.stations:
            solid = np.asarray(station.design)
            # Never outside the carved shape, never in the cargo space.
            assert not (solid & ~occupancy).any()
            assert not (solid & cargo).any()


def test_the_skin_survives_lightening(pod: tuple[DualStudy, DualSweep]) -> None:
    study, sweep = pod
    for row in sweep.rows:
        occupancy = study.grid.voxelize(row.shape)
        outer = occupancy & ~ndimage.binary_erosion(occupancy, structure=np.ones((3, 3, 3)))
        first, last = (np.asarray(row.stations[i].design) for i in (0, -1))
        # Whatever lightening took, it took from inside: the outer layer is intact.
        assert last[outer & first].all()
        assert last.sum() < first.sum()


def test_wind_loads_shrink_as_the_shape_gets_slippery(pod: tuple[DualStudy, DualSweep]) -> None:
    _, sweep = pod
    forces = [row.stations[0].metrics["force_n"] for row in sweep.rows]
    assert forces[-1] < forces[0]
    assert all(row.stations[0].metrics["side_n"] > 0 for row in sweep.rows)


def test_the_front_holds_only_designs_nothing_beats(pod: tuple[DualStudy, DualSweep]) -> None:
    _, sweep = pod
    front = sweep.front()
    assert front
    designs = [(s.metrics["drag"], s.metrics["mass_g"]) for _, _, s in sweep.designs()]
    for i, j in front:
        metrics = sweep.rows[i].stations[j].metrics
        d, m = metrics["drag"], metrics["mass_g"]
        assert not any(d2 <= d and m2 <= m and (d2 < d or m2 < m) for d2, m2 in designs)
    strict = sweep.front(min_safety=1e9)
    assert strict == []
    station = sweep.at(aero=1.0, weight=1.0)
    assert station is sweep.rows[-1].stations[-1]
    assert "carved" in sweep.table()


def test_lightened_meshes_keep_the_part_outside() -> None:
    # 130 x 67 mm on 12.5 mm voxels: neither is a whole number of them.
    box = shapes.box((300.0, 130.0, 67.0), center=(0.0, 0.0, 33.5), cell=10.0)
    study = DualStudy(
        box, flow=(-1.0, 0.0, 0.0), material="xps", resolution=16, structure_resolution=24
    )
    study.support(box=((50.0, -70.0, -5.0), (70.0, 70.0, 8.0)))
    study.support(box=((-70.0, -70.0, -5.0), (-50.0, 70.0, 8.0)))
    sweep = study.sweep(aero_stations=2, weight_stations=2, effort=0.1)
    first = sweep.rows[0].stations[0]
    lo, hi = box.bounds
    tolerance = 0.2 * study.grid.spacing
    assert np.all(np.abs(first.mesh.bounds[0] - lo) < tolerance)
    assert np.all(np.abs(first.mesh.bounds[1] - hi) < tolerance)
    assert first.metrics["mass_g"] == pytest.approx(
        box.volume / 1000.0 * study.material.density, rel=0.03
    )


def test_wind_forces_keep_the_total() -> None:
    box = shapes.box((300.0, 120.0, 60.0), cell=10.0)
    grid = Grid.around(box, resolution=30)
    solid = grid.voxelize(box)
    report = DragModel((-1.0, 0.0, 0.0))(box)
    forces = wind_forces(box, report, grid, solid, speed=30.0)
    assert forces.reshape(-1, 3).sum(axis=0) == pytest.approx(report.force_vector(30.0))
    # All of it lands on the material, none in the air around it.
    assert not np.any(forces[~solid])


def test_dual_study_needs_supports_and_a_horizontal_wind() -> None:
    box = shapes.box((60.0, 30.0, 20.0), cell=5.0)
    study = DualStudy(box, flow=(-1.0, 0.0, 0.0), resolution=16, structure_resolution=16)
    with pytest.raises(ValueError, match=r"support"):
        study.sweep()
    with pytest.raises(ValueError, match=r"vertical"):
        DualStudy(box, flow=(0.0, 0.0, -1.0), resolution=16)
    with pytest.raises(ValueError, match=r"skin"):
        DualStudy(box, flow=(-1.0, 0.0, 0.0), skin=0, resolution=16)
