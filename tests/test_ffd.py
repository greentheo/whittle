from __future__ import annotations

import numpy as np
import pytest

from whittle import FFDLattice
from whittle.types import Mesh


def test_param_count(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(2, 2, 2))
    assert lattice.n_control_points == 27
    assert lattice.n_params == 81
    assert lattice.changes_topology is False


def test_zero_displacement_is_the_identity(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(2, 2, 2))
    result = lattice.build(np.zeros(lattice.n_params))
    assert np.allclose(result.vertices, unit_cube.vertices)


def test_uniform_displacement_translates(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(2, 2, 2))
    genome = np.tile(np.array([0.3, 0.0, 0.0]), lattice.n_control_points)
    result = lattice.build(genome)
    # Bernstein weights are a partition of unity, so moving every control point
    # by the same amount is exactly a rigid translation.
    assert np.allclose(result.vertices - unit_cube.vertices, [0.3, 0.0, 0.0])


def test_topology_is_preserved(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(2, 2, 2))
    rng = np.random.default_rng(0)
    result = lattice.build(rng.uniform(-0.2, 0.2, lattice.n_params))
    assert np.array_equal(result.faces, unit_cube.faces)
    assert len(result.vertices) == len(unit_cube.vertices)


def test_frozen_axis_has_zero_width_bounds(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1), frozen_axes=(False, True, False))
    bounds = lattice.bounds
    assert len(bounds) == lattice.n_params
    assert all(bounds[i] == (0.0, 0.0) for i in range(1, len(bounds), 3))
    assert all(bounds[i] != (0.0, 0.0) for i in range(0, len(bounds), 3))


def test_wrong_genome_length_is_rejected(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1))
    with pytest.raises(ValueError, match=r"expected genome of 24 values"):
        lattice.build(np.zeros(5))


def test_rejects_empty_base_mesh() -> None:
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    with pytest.raises(ValueError, match=r"no vertices"):
        FFDLattice(empty)
