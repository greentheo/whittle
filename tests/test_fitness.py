from __future__ import annotations

import numpy as np
import pytest

from whittle import (
    AnchorConnectivity,
    Candidate,
    Compliance,
    DensityField,
    DragArea,
    DragCoefficient,
    FFDLattice,
    Grid,
    KeepOut,
    Load,
    MaterialFraction,
    MeshVolume,
    NoFolding,
    StayInside,
    Support,
    VolumeBudget,
    WeightedSum,
    shapes,
)
from whittle.types import Mesh


def test_material_fraction_tracks_the_field() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    assert MaterialFraction()(Candidate(np.zeros(8), backend)) == pytest.approx(0.0)
    assert MaterialFraction()(Candidate(np.ones(8), backend)) == pytest.approx(1.0)


def test_material_fraction_is_monotone() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    objective = MaterialFraction()
    values = [objective(Candidate(np.full(8, x), backend)) for x in (0.2, 0.5, 0.8)]
    assert values == sorted(values)


def test_mesh_volume_works_without_a_field(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1))
    assert MeshVolume()(Candidate(np.zeros(lattice.n_params), lattice)) == pytest.approx(1.0)


def test_field_objective_rejects_a_fieldless_backend(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1))
    with pytest.raises(TypeError, match=r"exposes no density field"):
        MaterialFraction()(Candidate(np.zeros(lattice.n_params), lattice))


def _two_pads(shape: tuple[int, int, int]) -> list[np.ndarray]:
    left = np.zeros(shape, dtype=bool)
    left[0:2, :, 0:2] = True
    right = np.zeros(shape, dtype=bool)
    right[-2:, :, 0:2] = True
    return [left, right]


def test_connectivity_is_zero_when_everything_is_solid() -> None:
    shape = (12, 4, 6)
    anchors = _two_pads(shape)
    backend = DensityField(shape, (3, 2, 2))
    assert AnchorConnectivity(anchors)(Candidate(np.ones(12), backend)) == 0.0


def test_connectivity_penalizes_a_void_design() -> None:
    shape = (12, 4, 6)
    anchors = _two_pads(shape)
    keep = anchors[0] | anchors[1]
    backend = DensityField(shape, (3, 2, 2), keep=keep)
    # Pads are forced solid but nothing joins them, so the penalty is the gap.
    penalty = AnchorConnectivity(anchors)(Candidate(np.zeros(12), backend))
    assert penalty > 0.0


def test_connectivity_rewards_closing_the_gap() -> None:
    shape = (16, 4, 6)
    anchors = _two_pads(shape)
    keep = anchors[0] | anchors[1]
    backend = DensityField(shape, (16, 4, 6), keep=keep)
    objective = AnchorConnectivity(anchors)

    def bridge(reach: int) -> float:
        grid = np.zeros(shape)
        grid[0:reach, :, 0:2] = 1.0
        return objective(Candidate(grid.ravel(), backend))

    # A longer arm from the left pad leaves less distance to the right one.
    assert bridge(10) < bridge(5)


def test_connectivity_needs_two_anchors() -> None:
    with pytest.raises(ValueError, match=r"at least two anchor"):
        AnchorConnectivity([np.ones((4, 4, 4), dtype=bool)])


def test_weighted_sum_combines_and_reports_terms() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    objective = WeightedSum({MaterialFraction(): 2.0})
    candidate = Candidate(np.ones(8), backend)
    assert objective(candidate) == pytest.approx(2.0)
    assert objective.terms(candidate) == {"material": pytest.approx(1.0)}


def test_weighted_sum_needs_an_objective() -> None:
    with pytest.raises(ValueError, match=r"at least one objective"):
        WeightedSum({})


def _beam_problem() -> tuple[DensityField, list[Load], list[Support]]:
    shape = (16, 4, 4)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    backend = DensityField(shape, (16, 4, 4), keep=clamp | tip)
    return backend, [Load(tip, (0.0, 0.0, -5.0))], [Support(clamp)]


def test_compliance_is_one_for_the_full_box_and_rises_as_material_goes() -> None:
    backend, loads, supports = _beam_problem()
    objective = Compliance(loads=loads, supports=supports)
    full = np.ones(backend.shape)
    assert objective(Candidate(full.ravel(), backend)) == pytest.approx(1.0, rel=1e-4)
    hollow = full.copy()
    hollow[2:14, 1:3, 1:3] = 0.0
    assert objective(Candidate(hollow.ravel(), backend)) > 1.0


def test_compliance_penalizes_a_broken_load_path() -> None:
    backend, loads, supports = _beam_problem()
    objective = Compliance(loads=loads, supports=supports, unsupported=500.0)
    broken = np.ones(backend.shape)
    broken[8] = 0.0
    assert objective(Candidate(broken.ravel(), backend)) == 500.0


def test_compliance_needs_a_density_field(unit_cube: Mesh) -> None:
    _, loads, supports = _beam_problem()
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1))
    with pytest.raises(TypeError, match=r"DensityField"):
        Compliance(loads=loads, supports=supports)(Candidate(np.zeros(24), lattice))


def test_drag_objectives_score_the_mesh(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1))
    candidate = Candidate(np.zeros(24), lattice)
    area = DragArea((1.0, 0.0, 0.0))(candidate)
    assert DragArea((1.0, 0.0, 0.0), reference=area)(candidate) == pytest.approx(1.0)
    # The fixture is a unit cube made of twelve triangles: no edges to read
    # curvature from, but a flat front and a flat back are all a cube needs.
    assert 0.9 < DragCoefficient((1.0, 0.0, 0.0))(candidate) < 1.2


def test_shape_constraints_are_zero_when_satisfied() -> None:
    sphere = shapes.sphere(10.0, subdivisions=3)
    grid = Grid.around(sphere, spacing=1.0, pad=2)
    solid = grid.voxelize(sphere)
    lattice = FFDLattice(sphere, divisions=(1, 1, 1), max_displacement=5.0)
    still = Candidate(np.zeros(lattice.n_params), lattice)
    assert StayInside(grid, solid)(still) == 0.0
    assert VolumeBudget(sphere.volume, 0.1)(still) == 0.0
    assert NoFolding(sphere)(still) == 0.0
    assert KeepOut([((-3, -3, -3), (3, 3, 3))])(still) == 0.0


def test_shape_constraints_catch_violations() -> None:
    sphere = shapes.sphere(10.0, subdivisions=3)
    grid = Grid.around(sphere, spacing=1.0, pad=4)
    solid = grid.voxelize(sphere)

    def scaled(factor: float) -> Candidate:
        return Candidate(np.zeros(1), _Fixed(Mesh(sphere.vertices * factor, sphere.faces)))

    assert StayInside(grid, solid)(scaled(1.2)) > 1.0
    assert VolumeBudget(sphere.volume, 0.1)(scaled(0.8)) == pytest.approx(1 - 0.8**3 - 0.1)
    assert KeepOut([((-9, -9, -9), (9, 9, 9))])(scaled(0.5)) > 0.0
    inside_out = Mesh(sphere.vertices, np.ascontiguousarray(sphere.faces[:, ::-1]))
    inverted = Candidate(np.zeros(1), _Fixed(inside_out))
    assert NoFolding(sphere)(inverted) == pytest.approx(1.0)


class _Fixed:
    """A backend that always builds the same mesh, for scoring a shape directly."""

    changes_topology = False
    n_params = 1
    bounds = [(0.0, 1.0)]

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def build(self, genome: np.ndarray) -> Mesh:
        return self.mesh
