from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage, sparse

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


def test_a_cycle_that_does_not_help_falls_back_on_jacobi(monkeypatch: pytest.MonkeyPatch) -> None:
    fea, kept = _awkward()
    jacobi = fea.solve(kept, preconditioner="jacobi", rtol=1e-10)
    # A cycle that points the wrong way: CG under it gets nowhere.
    monkeypatch.setattr(mg.Multigrid, "cycle", lambda self, b, depth=0: -b)
    monkeypatch.setattr(mg, "PATIENCE", 20)
    monkeypatch.setattr("whittle.physics.fea.PATIENCE", 20)
    rescued = fea.solve(kept, rtol=1e-10)
    assert rescued.converged and rescued.iterations > jacobi.iterations
    scale = np.abs(jacobi.displacement).max()
    assert np.allclose(rescued.displacement, jacobi.displacement, atol=1e-8 * scale)
