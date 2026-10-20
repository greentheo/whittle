from __future__ import annotations

import numpy as np
import pytest

from whittle import (
    AnchorConnectivity,
    Candidate,
    Compliance,
    DensityField,
    DragCoefficient,
    FFDLattice,
    MaterialFraction,
    MeshVolume,
    WeightedSum,
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


def test_unimplemented_physics_refuses_rather_than_guessing() -> None:
    backend = DensityField((4, 4, 4), (2, 2, 2))
    candidate = Candidate(np.ones(8), backend)
    with pytest.raises(NotImplementedError, match=r"FEA solver"):
        Compliance()(candidate)
    with pytest.raises(NotImplementedError, match=r"flow solver"):
        DragCoefficient()(candidate)
