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


def test_per_axis_displacement_bounds(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1), max_displacement=(0.1, 0.2, 0.3))
    assert lattice.bounds[:3] == [(-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3)]


def test_frozen_points_do_not_move(unit_cube: Mesh) -> None:
    frozen = np.zeros((3, 3, 3, 3), dtype=bool)
    frozen[:, :, 0, 2] = True  # the bottom layer may not move in z
    lattice = FFDLattice(unit_cube, divisions=(2, 2, 2), frozen=frozen)
    rng = np.random.default_rng(0)
    lows = np.array([b[0] for b in lattice.bounds])
    highs = np.array([b[1] for b in lattice.bounds])
    result = lattice.build(rng.uniform(lows, highs))
    bottom = unit_cube.vertices[:, 2] == 0.0
    # A Bezier volume's bottom face depends only on its bottom layer.
    assert np.allclose(result.vertices[bottom, 2], 0.0)


def test_mirror_halves_the_genome_and_keeps_the_shape_symmetric() -> None:
    from whittle import shapes

    box = shapes.box((2.0, 1.0, 1.0), cell=0.25)
    full = FFDLattice(box, divisions=(2, 2, 2))
    mirrored = FFDLattice(box, divisions=(2, 2, 2), mirror=1)
    assert mirrored.n_params == 18 * 3
    assert full.n_params == 27 * 3
    rng = np.random.default_rng(3)
    lows = np.array([b[0] for b in mirrored.bounds])
    highs = np.array([b[1] for b in mirrored.bounds])
    moved = mirrored.build(rng.uniform(lows, highs)).vertices
    # The box is symmetric about y = 0, so every vertex has a mirror partner.
    reflected = moved * np.array([1.0, -1.0, 1.0])
    index = {tuple(np.round(v, 9)): i for i, v in enumerate(box.vertices * [1.0, -1.0, 1.0])}
    partner = np.array([index[tuple(np.round(v, 9))] for v in box.vertices])
    assert np.allclose(reflected, moved[partner])


def test_inward_lattice_only_pushes_its_outside_in(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(2, 2, 2), max_displacement=0.3, inward=True)
    bounds = np.array(lattice.bounds).reshape(3, 3, 3, 3, 2)
    assert np.all(bounds[0, :, :, 0] == (0.0, 0.3))  # low-x face moves only toward +x
    assert np.all(bounds[2, :, :, 0] == (-0.3, 0.0))  # high-x face only toward -x
    assert np.all(bounds[1, :, :, 0] == (-0.3, 0.3))  # interior is free


def test_rejects_bad_options(unit_cube: Mesh) -> None:
    with pytest.raises(ValueError, match=r"mirror must be"):
        FFDLattice(unit_cube, mirror=3)
    with pytest.raises(ValueError, match=r"frozen must have shape"):
        FFDLattice(unit_cube, divisions=(1, 1, 1), frozen=np.zeros((3, 3, 3, 3), dtype=bool))


def test_weights_scale_how_far_each_vertex_follows(unit_cube: Mesh) -> None:
    weights = np.linspace(0.0, 1.0, len(unit_cube.vertices))
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1), weights=weights)
    genome = np.tile([0.5, 0.0, 0.0], lattice.n_control_points)
    moved = lattice.build(genome).vertices - unit_cube.vertices
    # A uniform move of the lattice is a translation, scaled per vertex.
    assert np.allclose(moved[:, 0], 0.5 * weights)
    with pytest.raises(ValueError, match=r"one per vertex"):
        FFDLattice(unit_cube, divisions=(1, 1, 1), weights=np.ones(3))
