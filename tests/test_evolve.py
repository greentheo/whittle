from __future__ import annotations

import numpy as np
import pytest

from whittle import DensityField, FFDLattice, MaterialFraction, WeightedSum, evolve
from whittle.types import Mesh


def test_evolve_drives_the_objective_down() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    result = evolve(backend, WeightedSum({MaterialFraction(): 1.0}), seed=0, maxiter=8, popsize=4)
    # Minimizing material with nothing holding it up should empty the box.
    assert result.fitness < 0.2
    assert result.n_evaluations > 0
    assert result.terms["material"] == pytest.approx(result.fitness, abs=1e-6)


def test_result_exposes_a_mesh() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    keep = np.zeros((8, 8, 8), dtype=bool)
    keep[2:6, 2:6, 2:6] = True
    backend = DensityField((8, 8, 8), (2, 2, 2), keep=keep)
    result = evolve(backend, WeightedSum({MaterialFraction(): 1.0}), seed=1, maxiter=5, popsize=4)
    assert len(result.mesh.faces) > 0


def test_frozen_parameters_are_held_fixed(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1), frozen_axes=(False, True, False))
    result = evolve(
        lattice,
        WeightedSum({__import__("whittle").MeshVolume(): 1.0}),
        seed=0,
        maxiter=4,
        popsize=4,
    )
    # Every y displacement was frozen at zero and must have stayed there.
    assert np.allclose(result.genome.reshape(-1, 3)[:, 1], 0.0)


def test_all_frozen_is_an_error(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1), frozen_axes=(True, True, True))
    with pytest.raises(ValueError, match=r"every parameter is frozen"):
        evolve(lattice, WeightedSum({MaterialFraction(): 1.0}), maxiter=2)


def test_history_is_recorded() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    seen: list[tuple[int, float]] = []
    evolve(
        backend,
        WeightedSum({MaterialFraction(): 1.0}),
        seed=0,
        maxiter=6,
        popsize=4,
        callback=lambda gen, value: seen.append((gen, value)),
    )
    assert len(seen) > 0
    assert seen[0][0] == 1


def test_a_warm_start_is_never_lost(unit_cube: Mesh) -> None:
    lattice = FFDLattice(unit_cube, divisions=(1, 1, 1), max_displacement=0.5)
    objective = WeightedSum({__import__("whittle").MeshVolume(): 1.0})
    start = np.full(lattice.n_params, 0.0)
    # Pull the top face down: a good, but not optimal, place to begin.
    start.reshape(2, 2, 2, 3)[:, :, 1, 2] = -0.4
    start_value = objective(__import__("whittle").Candidate(start, lattice))
    result = evolve(lattice, objective, seed=0, maxiter=3, popsize=4, x0=start, spread=0.05)
    assert result.fitness <= start_value + 1e-12
