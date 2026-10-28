from __future__ import annotations

import numpy as np
import pytest

from whittle import FlowStudy, Grid, LoadStudy, Station, Sweep, shapes
from whittle.optimize.beso import beso
from whittle.sweep import Column
from whittle.types import Mesh


def _shelf_bracket() -> LoadStudy:
    """A solid triangular bracket, screwed to a wall along x=0, loaded at its tip."""
    grid = Grid((30, 6, 30), 2.0, (1.0, 1.0, 1.0))
    c = grid.centers()
    solid = c[..., 2] >= c[..., 0] - 6.0
    study = LoadStudy.from_voxels(grid, solid, material="pla")
    study.support(box=((0, 0, 0), (3, 12, 60)))
    study.load((0.0, 0.0, -50.0), box=((52, 0, 56), (60, 12, 60)))
    return study


@pytest.fixture(scope="module")
def bracket_sweep() -> tuple[LoadStudy, Sweep]:
    study = _shelf_bracket()
    return study, study.sweep(remove_up_to=0.6, stations=4)


def test_load_sweep_runs_from_the_original_to_the_target(
    bracket_sweep: tuple[LoadStudy, Sweep],
) -> None:
    _, sweep = bracket_sweep
    removed = [s.removed for s in sweep]
    assert removed[0] == 0.0
    assert removed == sorted(removed)
    assert removed[-1] == pytest.approx(0.6, abs=0.01)
    assert sweep.note == ""


def test_load_sweep_trades_stiffness_for_mass(bracket_sweep: tuple[LoadStudy, Sweep]) -> None:
    _, sweep = bracket_sweep
    mass = [s.metrics["mass_g"] for s in sweep]
    stiffness = [s.metrics["stiffness"] for s in sweep]
    assert mass == sorted(mass, reverse=True)
    assert stiffness[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-9 for a, b in zip(stiffness, stiffness[1:], strict=False))
    # The point of the method: the first chunk of material is nearly free.
    first = sweep.at(0.2)
    assert first.metrics["stiffness"] > 0.9
    assert all(s.metrics["safety_factor"] > 0 for s in sweep)


def test_load_sweep_keeps_the_attachments_and_stays_connected(
    bracket_sweep: tuple[LoadStudy, Sweep],
) -> None:
    study, sweep = bracket_sweep
    fea = study.solver()
    for station in sweep:
        solid = station.design
        assert isinstance(solid, np.ndarray)
        for load in study.loads:
            assert solid[load.region].all()
        for support in study.supports:
            assert solid[support.region].all()
        assert fea.solve(solid).supported


def test_station_meshes_are_closed_and_shrink(bracket_sweep: tuple[LoadStudy, Sweep]) -> None:
    _, sweep = bracket_sweep
    volumes = [sweep[i].mesh.volume for i in (0, -1)]
    assert volumes[0] > 0
    assert volumes[1] < 0.6 * volumes[0]


def test_load_study_from_a_mesh() -> None:
    beam = shapes.box((60.0, 10.0, 10.0), center=(30.0, 5.0, 5.0), cell=5.0)
    study = LoadStudy(beam, resolution=24, material="petg")
    study.support(box=((0, 0, 0), (3, 10, 10)))
    study.load((0, 0, -20.0), at=(60.0, 5.0, 10.0))
    result = study.analyze()
    assert result.supported
    assert result.max_displacement > 0
    sweep = study.sweep(remove_up_to=0.3, stations=2)
    assert len(sweep) == 2


def test_load_study_placement_rules() -> None:
    study = _shelf_bracket()
    with pytest.raises(ValueError, match=r"exactly one"):
        study.region()
    with pytest.raises(ValueError, match=r"exactly one"):
        study.region(at=(1, 1, 1), box=((0, 0, 0), (1, 1, 1)))
    with pytest.raises(ValueError, match=r"none of the part"):
        study.region(box=((500, 500, 500), (600, 600, 600)))
    empty = LoadStudy.from_voxels(study.grid, study.solid)
    with pytest.raises(ValueError, match=r"support and one load"):
        empty.analyze()


def test_beso_reports_when_it_cannot_go_further() -> None:
    study = _shelf_bracket()
    # Ask for more than a bracket can lose while still joining the tip to the wall.
    run = beso(study.solver(), study.solid, removals=[0.0, 0.99], evolution_rate=0.1)
    assert "disconnects" in run.note
    # The furthest design that still holds is kept as the end of the slider.
    assert len(run.stations) == 2
    assert 0.5 < run.stations[-1].removed < 0.99
    assert run.stations[-1].result.supported


def test_beso_rejects_bad_removals() -> None:
    study = _shelf_bracket()
    with pytest.raises(ValueError, match=r"fractions"):
        beso(study.solver(), study.solid, removals=[1.0])


def test_sweep_navigation() -> None:
    def station(removed: float) -> Station:
        empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        return Station(removed, {"x": removed}, lambda: empty)

    sweep = Sweep("load", [station(0.0), station(0.2), station(0.4)], [Column("x", "x", "{:.2f}")])
    assert sweep.at(0.25).removed == 0.2
    assert sweep.slide(0.0).removed == 0.0
    assert sweep.slide(1.0).removed == 0.4
    assert sweep.slide(7.0).removed == 0.4
    assert sweep.original is sweep[0]
    assert "removed" in sweep.table() and "0.40" in sweep.table()
    with pytest.raises(ValueError, match=r"at least one station"):
        Sweep("load", [])


def test_load_stations_keep_the_part_outside_and_weigh_what_it_weighs() -> None:
    # 13 x 7 mm is not a whole number of 4.2 mm voxels.
    beam = shapes.box((100.0, 13.0, 7.0), center=(50.0, 6.5, 3.5), cell=1.0)
    study = LoadStudy(beam, resolution=24, material="petg")
    study.support(box=((0.0, 0.0, 0.0), (3.0, 13.0, 7.0)))
    study.load((0.0, 0.0, -50.0), box=((96.0, 0.0, 0.0), (100.0, 13.0, 7.0)))
    sweep = study.sweep(remove_up_to=0.4, stations=3)
    lo, hi = beam.bounds
    tolerance = 0.2 * study.grid.spacing
    for station in sweep:
        slo, shi = station.mesh.bounds
        assert np.all(slo > lo - tolerance) and np.all(shi < hi + tolerance)
    original = sweep.original
    assert np.all(np.abs(original.mesh.bounds[0] - lo) < tolerance)
    assert np.all(np.abs(original.mesh.bounds[1] - hi) < tolerance)
    true_mass = beam.volume / 1000.0 * study.material.density
    assert original.metrics["mass_g"] == pytest.approx(true_mass, rel=0.05)
    # Without the part's coverage, a study goes by its voxels, as it always did.
    voxels = LoadStudy.from_voxels(study.grid, study.solid, material="petg")
    assert voxels.coverage is None


@pytest.fixture(scope="module")
def box_flow() -> tuple[FlowStudy, Sweep]:
    box = shapes.box((300.0, 120.0, 80.0), center=(0.0, 0.0, 40.0), cell=10.0)
    study = FlowStudy(box, flow=(-1.0, 0.0, 0.0), resolution=24, mirror="y", flat=("-z",))
    return study, study.sweep(remove_up_to=0.2, stations=3, effort=0.3)


def test_flow_sweep_lowers_drag_and_respects_the_budget(box_flow: tuple[FlowStudy, Sweep]) -> None:
    _, sweep = box_flow
    drag = [s.metrics["drag"] for s in sweep]
    assert drag[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-9 for a, b in zip(drag, drag[1:], strict=False))
    assert drag[-1] < 0.9
    budgets = [0.0, 0.1, 0.2]
    for station, budget in zip(sweep, budgets, strict=True):
        assert station.removed <= budget + 0.01


def test_flow_sweep_never_grows_the_part(box_flow: tuple[FlowStudy, Sweep]) -> None:
    study, sweep = box_flow
    lo, hi = study.base.bounds
    for station in sweep:
        slo, shi = station.mesh.bounds
        assert np.all(slo >= lo - study.grid.spacing)
        assert np.all(shi <= hi + study.grid.spacing)


def test_flow_sweep_keeps_symmetry_and_the_flat_bottom(box_flow: tuple[FlowStudy, Sweep]) -> None:
    study, sweep = box_flow
    base_bottom = study.base.vertices[:, 2].min()
    mesh = sweep[-1].mesh
    bottom = study.base.vertices[:, 2] < base_bottom + 1e-6
    assert np.allclose(mesh.vertices[bottom, 2], base_bottom)
    # The deformation is mirrored about the middle of y. (The remeshed base is
    # not vertex-for-vertex symmetric -- marching cubes is not -- so compare
    # the control lattice, not the vertices.)
    moves = study.lattice().displacements(np.asarray(sweep[-1].design)).reshape(3, 3, 3, 3)
    assert np.allclose(moves[:, 0, :, [0, 2]], moves[:, 2, :, [0, 2]])
    assert np.allclose(moves[:, 0, :, 1], -moves[:, 2, :, 1])
    assert np.allclose(moves[:, 1, :, 1], 0.0)
    lo, hi = mesh.bounds
    assert lo[1] == pytest.approx(-hi[1], abs=study.grid.spacing * 0.6)


def test_a_flow_study_carves_the_part_as_sent() -> None:
    # A box as CAD would send it, twelve triangles, its sides not a whole
    # number of voxels: station 0 is that box, not a voxel copy of it.
    box = shapes.box((990.6, 1879.6, 1041.4), center=(-952.5, 0.0, 977.9), cell=5000.0)
    study = FlowStudy(box, flow=(-1.0, 0.0, 0.0), resolution=16, mirror="y", flat=("-z",))
    assert not study.remeshed
    lo, hi = box.bounds
    assert np.allclose(study.base.bounds[0], lo) and np.allclose(study.base.bounds[1], hi)
    assert study.base.volume == pytest.approx(box.volume, rel=1e-9)
    sweep = study.sweep(remove_up_to=0.1, stations=2, effort=0.2)
    assert sweep.original.removed == pytest.approx(0.0, abs=1e-12)
    assert sweep.original.mesh.volume == pytest.approx(box.volume, rel=1e-9)
    # The flat bottom stays exactly where the part had it.
    assert sweep[-1].mesh.bounds[0][2] == pytest.approx(lo[2], abs=1e-9)


def test_a_part_that_is_not_one_clean_surface_is_remeshed_where_it_was() -> None:
    # Two boxes overlapping, as two shells: the signed volume counts the overlap twice.
    a = shapes.box((1000.0, 600.0, 400.0), center=(0.0, 0.0, 200.0), cell=50.0)
    b = shapes.box((600.0, 600.0, 400.0), center=(500.0, 0.0, 300.0), cell=50.0)
    both = Mesh(
        np.concatenate([a.vertices, b.vertices]),
        np.concatenate([a.faces, b.faces + len(a.vertices)]),
    )
    study = FlowStudy(both, flow=(-1.0, 0.0, 0.0), resolution=24)
    assert study.remeshed
    lo, hi = both.bounds
    blo, bhi = study.base.bounds
    # Within a quarter of a voxel, where taking which centres were inside left
    # it as much as half a voxel out.
    assert np.all(np.abs(blo - lo) < 0.25 * study.grid.spacing)
    assert np.all(np.abs(bhi - hi) < 0.25 * study.grid.spacing)


def test_flow_study_validates_sides() -> None:
    box = shapes.box((30.0, 12.0, 8.0), cell=2.0)
    with pytest.raises(ValueError, match=r"sides look like"):
        FlowStudy(box, flow=(1, 0, 0), flat=("bottom",))


def test_an_extruded_study_keeps_the_section_constant() -> None:
    bar = shapes.box((40.0, 60.0, 25.0), cell=2.0)
    study = FlowStudy(bar, flow=(-1.0, 0.0, 0.0), resolution=24, extruded="y")
    sweep = study.sweep(stations=[0.2], effort=0.2)
    assert sweep[-1].metrics["drag"] < 1.0
    moves = study.lattice().displacements(np.asarray(sweep[-1].design)).reshape(3, 2, 3, 3)
    # Both layers along the bar move together, and nothing moves along it.
    assert np.allclose(moves[:, 0, :, [0, 2]], moves[:, 1, :, [0, 2]])
    assert np.allclose(moves[..., 1], 0.0)
    with pytest.raises(ValueError, match=r"drop mirror"):
        FlowStudy(bar, flow=(-1.0, 0.0, 0.0), resolution=24, extruded="y", mirror="z")


def test_naming_a_surface_keeps_every_other_one_intact() -> None:
    study = _shelf_bracket()
    # Lighten only from the +y face; the -y face and the edges must stay as they are.
    study.whittle(at=(30.0, 12.0, 45.0), normal=(0.0, 1.0, 0.0))
    kept = study.kept()
    assert kept[:, 0, :][study.solid[:, 0, :]].all()  # the -y layer is all kept
    assert not kept[:, -1, :][study.solid[:, -1, :] & ~study.protected[:, -1, :]].all()
    sweep = study.sweep(remove_up_to=0.4, stations=2)
    last = np.asarray(sweep[-1].design)
    assert last[:, 0, :][study.solid[:, 0, :]].all()
    # The pockets open onto the named face, which really did lose material.
    assert last[:, -1, :].sum() < study.solid[:, -1, :].sum()


def test_without_named_surfaces_only_kept_regions_are_protected() -> None:
    study = _shelf_bracket()
    study.keep(at=(30.0, 6.0, 60.0), radius=4.0)
    assert np.array_equal(study.kept(), study.protected)


def test_carving_only_named_surfaces_leaves_the_rest_exactly_in_place() -> None:
    box = shapes.box((300.0, 120.0, 80.0), center=(0.0, 0.0, 40.0), cell=10.0)
    study = FlowStudy(box, flow=(-1.0, 0.0, 0.0), resolution=24, flat=("-z",))
    assert study.weights() is None
    study.whittle(at=(150.0, 0.0, 40.0), normal=(1.0, 0.0, 0.0))  # the nose only
    weights = study.weights()
    assert weights is not None
    assert weights.min() == 0.0 and weights.max() == 1.0
    noses = study.base.vertices[:, 0] > 140.0
    tails = study.base.vertices[:, 0] < -140.0
    assert weights[noses].mean() > 0.5 and weights[tails].max() == 0.0
    sweep = study.sweep(remove_up_to=0.1, stations=2, effort=0.3)
    moved = np.linalg.norm(sweep[-1].mesh.vertices - study.base.vertices, axis=1)
    assert moved[weights == 0].max() == 0.0
    assert moved[weights == 1].max() > 0.0
