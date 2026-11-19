from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage, sparse

import whittle
from whittle import Load, Support, VoxelFEA
from whittle.physics import multigrid as mg


def _awkward() -> tuple[VoxelFEA, np.ndarray]:
    """Odd sizes, holes, a roller as well as a clamp, and a stiffness per voxel."""
    rng = np.random.default_rng(1)
    shape = (9, 6, 5)
    solid = rng.random(shape) > 0.25
    solid[0] = True
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    floor = np.zeros(shape, dtype=bool)
    floor[:, :, 0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    fea = VoxelFEA(
        shape,
        spacing=1.5,
        loads=[Load(tip, (0.0, 0.0, -1.0))],
        supports=[Support(clamp), Support(floor, (False, False, True))],
        stiffness=rng.uniform(0.3, 2.0, shape),
    )
    labels, _ = ndimage.label(solid)
    kept = np.isin(labels, np.unique(labels[clamp & solid]))
    return fea, kept


def test_coarse_grids_are_exactly_the_galerkin_product() -> None:
    # The element-by-element coarse stiffness is P^T K P with P the trilinear
    # interpolation onto the free fine DOFs -- held ones struck out -- two grids down.
    fea, kept = _awkward()
    elements = np.flatnonzero(kept.ravel())
    edof = fea._edof[elements]
    active = np.zeros(fea.n_dof, dtype=bool)
    active[edof.ravel()] = True
    free = np.flatnonzero(active & ~fea._fixed)
    assert fea.stiffness is not None
    scale = fea.stiffness.ravel()[elements]
    rows, cols = np.repeat(edof, 24, axis=1).ravel(), np.tile(edof, (1, 24)).ravel()
    values = (fea._ke[None] * scale[:, None, None]).ravel()
    k = sparse.csr_matrix((values, (rows, cols)), shape=(fea.n_dof, fea.n_dof))[free][:, free]

    coords = np.stack(np.unravel_index(elements, fea.shape), axis=1)
    fine = mg._nodes(coords, fea.shape)
    step = mg._coarsen(coords, fea.shape)
    nodes = mg._nodes(step.elements, step.shape)
    in_use = np.ravel_multi_index(fine.coords.T, tuple(s + 1 for s in fea.shape))
    slot = 3 * np.searchsorted(in_use, free // 3) + free % 3
    p = sparse.kron(mg._prolongation(fine.coords, nodes, step.factors, step.shape), np.eye(3))
    galerkin = (p.tocsr()[slot].T @ k @ p.tocsr()[slot]).toarray()
    first = mg.Multigrid._first(step, fea._ke, scale, fea._fixed[edof])
    coarse = mg._assemble(first, nodes.corners).toarray()
    live = np.diag(galerkin) > 1e-9 * np.abs(galerkin).max()
    assert (~live).sum() > 0  # coarse DOFs over the clamp interpolate only onto held ones
    assert np.allclose(coarse[np.ix_(live, live)], galerkin[np.ix_(live, live)], atol=1e-12)
    # ...and those sit out: nothing but a unit diagonal.
    assert np.allclose(coarse[~live], np.eye(len(live))[~live])

    step2 = mg._coarsen(step.elements, step.shape)
    nodes2 = mg._nodes(step2.elements, step2.shape)
    p2 = sparse.kron(mg._prolongation(nodes.coords, nodes2, step2.factors, step2.shape), np.eye(3))
    galerkin2 = (p2.T @ sparse.csr_matrix(galerkin) @ p2).toarray()
    coarse2 = mg._assemble(mg._galerkin(first, step2), nodes2.corners).toarray()
    live2 = np.diag(galerkin2) > 1e-9 * np.abs(galerkin2).max()
    assert np.allclose(coarse2[np.ix_(live2, live2)], galerkin2[np.ix_(live2, live2)], atol=1e-12)


def test_multigrid_gives_jacobis_answer_in_far_fewer_iterations() -> None:
    shape = (120, 6, 8)  # slender: Jacobi needs hundreds of iterations
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    solid = np.ones(shape, dtype=bool)
    solid[10:110:12, :, 2:6] = False  # lightening slots
    fea = VoxelFEA(
        shape,
        loads=[Load(tip, (0.0, 0.0, -10.0)), Load(tip, (0.0, 5.0, 0.0), case=1)],
        supports=[Support(clamp)],
    )
    jacobi = fea.solve(solid, preconditioner="jacobi", rtol=1e-9)
    multigrid = fea.solve(solid, rtol=1e-9)
    assert fea.preconditioner == "multigrid"
    assert multigrid.converged and multigrid.iterations * 10 < jacobi.iterations
    scale = np.abs(jacobi.displacement).max()
    assert np.allclose(multigrid.displacement, jacobi.displacement, atol=1e-7 * scale)
    assert multigrid.case_compliance == pytest.approx(jacobi.case_compliance, rel=1e-7)
    assert np.allclose(multigrid.von_mises, jacobi.von_mises, rtol=1e-5, atol=1e-8)


def _beam(shape: tuple[int, int, int]) -> tuple[VoxelFEA, np.ndarray]:
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    fea = VoxelFEA(shape, loads=[Load(tip, (0.0, 0.0, -10.0))], supports=[Support(clamp)])
    return fea, np.ones(shape, dtype=bool)


def _grids(fea: VoxelFEA, solid: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> mg.Multigrid:
    """The cycle a solve of ``solid`` builds."""
    built: list[mg.Multigrid] = []
    build = mg.Multigrid.__init__

    def keep(self: mg.Multigrid, *args: object) -> None:
        build(self, *args)  # type: ignore[arg-type]
        built.append(self)

    monkeypatch.setattr(mg.Multigrid, "__init__", keep)
    fea.solve(solid, maxiter=1)
    monkeypatch.setattr(mg.Multigrid, "__init__", build)
    return built[0]


def test_a_part_three_grids_deep_still_takes_few_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The smallest part that reaches the assembled grids below the first coarse one.
    # A transfer or smoother there that does nothing still gets the right answer --
    # CG sees to that -- in twenty times the iterations, so only the count can tell.
    fea, solid = _beam((300, 12, 16))
    assert len(_grids(fea, solid, monkeypatch).sizes) >= 3
    result = fea.solve(solid)
    assert result.converged and result.iterations <= 20


@pytest.mark.parametrize("part", ["beam", "awkward"])
def test_the_cycle_is_symmetric_and_positive(part: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # CG needs its preconditioner symmetric positive definite: on every grid the
    # restriction must be exactly the prolongation's transpose, and the smoothing
    # the same on the way down as on the way up.
    if part == "beam":
        fea, solid = _beam((300, 12, 16))
    else:
        fea, solid = _awkward()
        # Held DOFs, a roller and odd sizes, three grids down.
        monkeypatch.setattr(mg, "COARSEST", 30)
    grids = _grids(fea, solid, monkeypatch)
    assert len(grids.sizes) >= 3
    a, b = np.random.default_rng(3).standard_normal((2, grids.sizes[0]))
    # To round-off: the beam's coarsest grid is factored by sparse LU, whose solve is
    # symmetric only to its condition (3e7) times machine precision -- 4e-9 here. A
    # restriction that isn't the prolongation's transpose is off by a tenth or more.
    assert a @ grids.cycle(b) == pytest.approx(b @ grids.cycle(a), rel=1e-6)
    assert a @ grids.cycle(a) > 0 and b @ grids.cycle(b) > 0


@pytest.mark.parametrize("part", ["beam", "awkward"])
def test_each_restriction_is_its_prolongations_transpose(
    part: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Symmetry of the whole cycle cannot see a restriction off by a constant: R = P^T / 2
    # just halves the coarse correction, which is still symmetric, and costs only a few
    # iterations. So ask each grid directly: y . P x must be x . R y.
    if part == "beam":
        fea, solid = _beam((300, 12, 16))
    else:
        fea, solid = _awkward()
        monkeypatch.setattr(mg, "COARSEST", 30)
    grids = _grids(fea, solid, monkeypatch)
    rng = np.random.default_rng(4)
    for depth, level in enumerate(grids.levels):
        x, y = rng.standard_normal(grids.sizes[depth + 1]), rng.standard_normal(grids.sizes[depth])
        assert y @ level.prolong(x) == pytest.approx(x @ level.restrict(y), rel=1e-12)


def test_the_awkward_part_solves_the_same_either_way() -> None:
    fea, kept = _awkward()
    jacobi = fea.solve(kept, preconditioner="jacobi", rtol=1e-10)
    multigrid = fea.solve(kept, preconditioner="multigrid", rtol=1e-10)
    scale = np.abs(jacobi.displacement).max()
    assert np.allclose(multigrid.displacement, jacobi.displacement, atol=1e-8 * scale)


@pytest.mark.parametrize("shape", [(2, 1, 1), (3, 2, 1), (2, 2, 2)])
def test_tiny_parts_need_no_coarse_grid_to_speak_of(shape: tuple[int, int, int]) -> None:
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    fea = VoxelFEA(
        shape, loads=[Load(np.ones(shape, dtype=bool), (0.0, 0.0, -1.0))], supports=[Support(clamp)]
    )
    solid = np.ones(shape, dtype=bool)
    jacobi = fea.solve(solid, preconditioner="jacobi", rtol=1e-10)
    multigrid = fea.solve(solid, rtol=1e-10)
    assert multigrid.converged
    assert multigrid.compliance == pytest.approx(jacobi.compliance, rel=1e-8)


def test_preconditioner_names_are_checked() -> None:
    fea, kept = _awkward()
    with pytest.raises(ValueError, match="preconditioner must be one of"):
        fea.solve(kept, preconditioner="ilu")
    with pytest.raises(ValueError, match="preconditioner must be one of"):
        VoxelFEA(fea.shape, loads=fea.loads, supports=fea.supports, preconditioner="amg")


@pytest.mark.parametrize("dense", [True, False])
def test_a_singular_coarsest_grid_still_solves_what_it_can(
    dense: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Where a thin member meets a support, coarse DOFs can interpolate onto the same free
    # fine ones, and the coarsest matrix loses rank -- with round-off for its zero
    # eigenvalues, which neither factorization notices. Anything it is asked to solve
    # comes from the fine grid, so lies in its range; that much must come out as the
    # pseudo-inverse would have it, nothing blown up along the null space.
    monkeypatch.setattr(mg, "DENSE", 3000 if dense else 0)
    rng = np.random.default_rng(2)
    b = rng.standard_normal((40, 60))
    b[20:] = rng.standard_normal((20, 20)) @ b[:20]  # rank 20 of 60
    full = b.T @ b
    matrix = sparse.bsr_matrix(full, blocksize=(3, 3))
    solve = mg.Multigrid._factor(matrix)
    rhs = full @ rng.standard_normal(60)
    expected = np.linalg.pinv(full) @ rhs
    assert np.allclose(solve(rhs), expected, atol=1e-5 * np.abs(expected).max())


def test_a_cycle_that_points_the_wrong_way_hands_over_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fea, kept = _awkward()
    jacobi = fea.solve(kept, preconditioner="jacobi", rtol=1e-10)
    # Not positive: CG breaks down on its first step, and Jacobi starts from zero.
    monkeypatch.setattr(mg.Multigrid, "cycle", lambda self, b, depth=0: -b)
    rescued = fea.solve(kept, rtol=1e-10)
    assert rescued.converged and rescued.fell_back
    assert rescued.iterations == rescued.fallback_iterations == jacobi.iterations
    scale = np.abs(jacobi.displacement).max()
    assert np.allclose(rescued.displacement, jacobi.displacement, atol=1e-8 * scale)


def test_a_cycle_that_goes_wrong_mid_solve_hands_over_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fea, kept = _awkward()
    jacobi = fea.solve(kept, preconditioner="jacobi", rtol=1e-10)
    calls = 0
    cycle = mg.Multigrid.cycle

    def soured(self: mg.Multigrid, b: np.ndarray, depth: int = 0) -> np.ndarray:
        # Right for CG's first three calls -- the start and two iterations -- then
        # backwards. (The cycle calls itself once per grid; only the top call is CG's.)
        nonlocal calls
        result: np.ndarray = cycle(self, b, depth)
        if depth:
            return result
        calls += 1
        return result if calls <= 3 else -result

    monkeypatch.setattr(mg.Multigrid, "cycle", soured)
    rescued = fea.solve(kept, rtol=1e-10)
    assert rescued.converged and rescued.fell_back
    assert rescued.iterations == 3 + jacobi.iterations
    assert rescued.fallback_iterations == jacobi.iterations


def _slotted() -> tuple[VoxelFEA, np.ndarray]:
    shape = (120, 6, 8)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    solid = np.ones(shape, dtype=bool)
    solid[10:110:12, :, 2:6] = False
    return VoxelFEA(shape, loads=[Load(tip, (0.0, 0.0, -10.0))], supports=[Support(clamp)]), solid


def _slow(monkeypatch: pytest.MonkeyPatch) -> tuple[VoxelFEA, np.ndarray, float]:
    """A beam so slender, and a coarse correction so weakened, that the cycle takes ~280
    iterations -- slow, but its residual never stops falling for long. And the compliance
    the cycle as it should be gets."""
    shape = (400, 3, 4)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    fea = VoxelFEA(shape, loads=[Load(tip, (0.0, 0.0, -10.0))], supports=[Support(clamp)])
    solid = np.ones(shape, dtype=bool)
    reference = fea.solve(solid, rtol=1e-8).compliance
    factor = mg.Multigrid._factor

    def weakened(matrix: sparse.bsr_matrix) -> mg.Apply:
        solve = factor(matrix)
        return lambda b: 1e-3 * solve(b)

    monkeypatch.setattr(mg.Multigrid, "_factor", staticmethod(weakened))
    return fea, solid, reference


def test_a_slow_but_steady_solve_is_left_to_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    # Past the 250 iterations that used to hand any solve over; but its residual keeps
    # falling, so it is left to finish on multigrid rather than started again.
    fea, solid, reference = _slow(monkeypatch)
    slow = fea.solve(solid)
    assert slow.converged and not slow.fell_back and slow.fallback_iterations == 0
    assert slow.iterations > 250
    assert slow.compliance == pytest.approx(reference, rel=1e-5)


def test_patience_is_still_a_hard_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    fea, solid, reference = _slow(monkeypatch)
    jacobi = fea.solve(solid, preconditioner="jacobi")
    monkeypatch.setattr(mg, "PATIENCE", 100)
    capped = fea.solve(solid)
    assert capped.converged and capped.fell_back
    # Jacobi starts afresh, so it takes exactly what it takes alone.
    assert capped.fallback_iterations == jacobi.iterations
    assert capped.iterations == 100 + jacobi.iterations
    assert capped.compliance == pytest.approx(reference, rel=1e-5)


def test_a_solve_that_stalls_hands_over_when_its_window_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cycle that all but ignores half the unknowns: positive, so CG never breaks down,
    # but once it has fixed the other half its residual stops falling. It must hand over
    # as stalled -- well before the hard cap would have caught it.
    fea, solid = _slotted()
    jacobi = fea.solve(solid, preconditioner="jacobi")

    def half(self: mg.Multigrid, b: np.ndarray, depth: int = 0) -> np.ndarray:
        result: np.ndarray = self.levels[0].inverse_diagonal * b
        result[1::2] *= 1e-12
        return result

    monkeypatch.setattr(mg.Multigrid, "cycle", half)
    grids = _grids(fea, solid, monkeypatch)
    elements = np.flatnonzero(solid.ravel())
    active = np.zeros(fea.n_dof, dtype=bool)
    active[fea._edof[elements].ravel()] = True
    b = fea.forces[0][np.flatnonzero(active & ~fea._fixed)]
    _, ending, used = mg.pcg(grids.levels[0].apply, grids.cycle, b, rtol=1e-5, maxiter=mg.PATIENCE)
    assert ending == "stalled" and mg.STALL <= used < mg.PATIENCE // 2
    stalled = fea.solve(solid)
    assert stalled.converged and stalled.fell_back
    assert stalled.iterations == used + jacobi.iterations
    assert stalled.fallback_iterations == jacobi.iterations
    assert stalled.compliance == pytest.approx(jacobi.compliance, rel=1e-5)


def test_a_stall_is_judged_on_the_best_residual_not_the_latest() -> None:
    window = mg.STALL
    # Falling a tenth of a percent an iteration, with every other one spiking far above:
    # the best keeps falling, so it is not a stall, whatever the latest says.
    spiky = [0.999**k if k % 2 == 0 else 10.0 for k in range(3 * window)]
    assert not any(mg._stalled(spiky[: k + 1]) for k in range(len(spiky)))
    # Down by half, then flat: a stall, once a whole window has gone by on the flat.
    flat = [1.0] + [0.5] * (2 * window)
    assert not mg._stalled(flat[: window + 1])
    assert mg._stalled(flat[: window + 2])
    # Steady progress, a fifth of a percent an iteration: never.
    steady = [0.998**k for k in range(3 * window)]
    assert not any(mg._stalled(steady[: k + 1]) for k in range(len(steady)))


def test_a_result_says_when_jacobi_had_to_finish_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    # Caliper's timings read this to explain a slow case. One iteration is never enough
    # for the real cycle on this beam, so every load case falls back.
    shape = (60, 4, 6)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    fea = VoxelFEA(
        shape,
        loads=[Load(tip, (0.0, 0.0, -10.0)), Load(tip, (0.0, 5.0, 0.0), case=1)],
        supports=[Support(clamp)],
    )
    solid = np.ones(shape, dtype=bool)
    usual = fea.solve(solid)
    assert not usual.fell_back and usual.fallback_iterations == 0
    jacobi = fea.solve(solid, preconditioner="jacobi")
    assert not jacobi.fell_back and jacobi.fallback_iterations == 0
    monkeypatch.setattr(mg, "PATIENCE", 1)
    impatient = fea.solve(solid)
    assert impatient.converged and impatient.fell_back
    assert impatient.iterations == 2 + impatient.fallback_iterations  # one wasted per case
    assert impatient.fallback_iterations > 10 * usual.iterations
    assert impatient.case_compliance == pytest.approx(usual.case_compliance, rel=1e-5)


def _three_case_beam() -> tuple[VoxelFEA, np.ndarray]:
    shape = (120, 6, 8)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    near = np.zeros(shape, dtype=bool)
    near[8] = True
    solid = np.ones(shape, dtype=bool)
    solid[10:110:12, :, 2:6] = False
    loads = [
        Load(tip, (0.0, 0.0, -10.0)),
        Load(tip, (0.0, 5.0, 0.0), case=1),
        Load(near, (0.0, 0.0, -10.0), case=2),
    ]
    return VoxelFEA(shape, loads=loads, supports=[Support(clamp)]), solid


def test_threads_never_change_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Over repeated runs and any number of threads: the same bits. Nothing is summed
    # across cases, so there is no order for threads to change.
    fea, solid = _three_case_beam()
    results = []
    for threads in (1, 2, 4, 4, 2):
        monkeypatch.setattr("whittle.physics.fea.CASE_THREADS", threads)
        results.append(fea.solve(solid))
    first = results[0]
    for other in results[1:]:
        assert other.iterations == first.iterations
        assert other.case_compliance == first.case_compliance
        assert np.array_equal(other.displacement, first.displacement)
        assert np.array_equal(other.strain_energy, first.strain_energy)
        assert np.array_equal(other.von_mises, first.von_mises)


def test_a_fallback_or_a_failure_on_a_thread_is_never_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    fea, solid = _three_case_beam()
    monkeypatch.setattr(mg, "PATIENCE", 1)
    monkeypatch.setattr("whittle.physics.fea.CASE_THREADS", 1)
    serial = fea.solve(solid)
    monkeypatch.setattr("whittle.physics.fea.CASE_THREADS", 3)
    threaded = fea.solve(solid)
    assert threaded.fell_back and serial.fell_back
    assert threaded.fallback_iterations == serial.fallback_iterations > 0
    assert threaded.iterations == serial.iterations

    def boom(self: mg.Multigrid, b: np.ndarray, depth: int = 0) -> np.ndarray:
        raise RuntimeError("the cycle failed")

    monkeypatch.setattr(mg.Multigrid, "cycle", boom)
    with pytest.raises(RuntimeError, match="the cycle failed"):
        fea.solve(solid)


@pytest.mark.parametrize(("setting", "threads"), [("1", 1), ("2", 2), (None, 4)])
def test_case_threads_come_from_the_environment(setting: str | None, threads: int) -> None:
    # As caliper will set it: in the environment, before whittle is imported.
    env = {k: v for k, v in os.environ.items() if k != "WHITTLE_CASE_THREADS"}
    if setting is not None:
        env["WHITTLE_CASE_THREADS"] = setting
    env["PYTHONPATH"] = str(Path(whittle.__file__).parents[1])
    script = "from whittle.physics import fea; print(fea.CASE_THREADS)"
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout) == threads


@pytest.mark.parametrize(("setting", "cases", "most"), [(1, 3, 0), (2, 3, 2), (3, 3, 3), (4, 3, 3)])
def test_cases_run_on_as_many_threads_as_allowed(
    setting: int, cases: int, most: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fea, solid = _three_case_beam()
    seen: list[threading.Thread] = []
    pcg = mg.pcg

    def spy(*args: object, **kwargs: object) -> tuple[np.ndarray, str, int]:
        seen.append(threading.current_thread())
        return pcg(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mg, "pcg", spy)
    monkeypatch.setattr("whittle.physics.fea.CASE_THREADS", setting)
    # case threads run only on single-threaded BLAS (#9)
    monkeypatch.setattr("whittle.physics.fea._blas_single", lambda: True)
    fea.solve(solid)
    assert len(seen) == cases
    workers = {t for t in seen if t is not threading.main_thread()}
    if most == 0:
        assert not workers  # one at a time, on the caller's own thread
    else:
        assert len(workers) == most and threading.main_thread() not in seen


@pytest.mark.parametrize("setting", ["0", "-2", "four", ""])
def test_a_bad_thread_count_is_refused(setting: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from whittle.physics import fea as module

    monkeypatch.setenv("WHITTLE_CASE_THREADS", setting)
    with pytest.raises(ValueError, match="WHITTLE_CASE_THREADS"):
        module._case_threads()


def test_each_grids_largest_eigenvalue_is_estimated_closely_and_never_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The smoothing interval is set from it. A dozen power steps came out 15-25% low --
    # far enough to put the true value past the top of the interval -- so it is Lanczos.
    fea, kept = _awkward()
    monkeypatch.setattr(mg, "COARSEST", 30)
    grids = _grids(fea, kept, monkeypatch)
    for level in grids.levels:
        n = len(level.inverse_diagonal)
        scale = np.sqrt(level.inverse_diagonal)
        dense = np.column_stack([scale * level.apply(scale * e) for e in np.eye(n)])
        exact = np.linalg.eigvalsh((dense + dense.T) / 2)[-1]
        assert 0.99 * exact <= level.largest <= exact * (1 + 1e-9)
    # A space smaller than the steps: exact.
    small = np.diag([1.0, 2.0, 5.0])
    assert mg._largest(lambda x: small @ x, np.ones(3)) == pytest.approx(5.0, rel=1e-12)


def test_the_largest_eigenvalue_of_a_long_beam_is_not_underestimated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The smoothing interval runs to 1.1x this estimate. #5 took it from a dozen power
    # steps, 15-33% low on long beams and carved parts; Lanczos, as many steps, is within
    # a percent. Checked against an eigensolver run to convergence.
    from scipy.sparse.linalg import LinearOperator, eigsh

    fea, solid = _beam((300, 12, 16))
    fine = _grids(fea, solid, monkeypatch).levels[0]
    scale = np.sqrt(fine.inverse_diagonal)
    n = len(scale)
    op = LinearOperator((n, n), matvec=lambda x: scale * fine.apply(scale * x), dtype=float)
    exact = eigsh(op, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    assert 0.99 * exact <= fine.largest <= exact * (1 + 1e-9)


def test_the_smoothing_interval_is_safe_only_with_an_estimate_that_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Down to a twentieth of the estimate, a degree-2 smoother damps only up to 1.15x
    # the estimate. On this beam the old power-step estimate puts the true value past
    # that, and CG under the cycle breaks down within a few steps; with Lanczos it
    # converges in about ten.
    fea, solid = _beam((300, 12, 16))
    result = fea.solve(solid)
    assert result.converged and not result.fell_back and result.iterations <= 20

    def power_steps(apply: mg.Apply, inverse_diagonal: np.ndarray) -> float:
        x = np.random.default_rng(0).standard_normal(len(inverse_diagonal))
        estimate = 1.0
        for _ in range(12):
            kx = apply(x)
            estimate = float(x @ kx) / float(x @ (x / inverse_diagonal))
            x = inverse_diagonal * kx
            x /= np.linalg.norm(x)
        return estimate

    monkeypatch.setattr(mg, "_largest", power_steps)
    grids = _grids(fea, solid, monkeypatch)
    elements = np.flatnonzero(solid.ravel())
    active = np.zeros(fea.n_dof, dtype=bool)
    active[fea._edof[elements].ravel()] = True
    b = fea.forces[0][np.flatnonzero(active & ~fea._fixed)]
    _, ending, used = mg.pcg(grids.levels[0].apply, grids.cycle, b, rtol=1e-5, maxiter=100)
    assert ending == "broke down" and used < 20
