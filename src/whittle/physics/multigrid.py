"""Geometric multigrid: the preconditioner that keeps the voxel solve's CG short.

Jacobi-preconditioned CG fixes the error one voxel's worth of distance per
iteration, so a part 200 voxels long needs several hundred iterations before a
load at one end is felt at the other. Multigrid fixes it at every wavelength
at once: a few smoothing steps take out what varies from voxel to voxel, and
what is left is smooth enough to solve on a grid twice as coarse -- and so on
down to a grid small enough to factor outright. Used as CG's preconditioner it
takes a slender part from hundreds of iterations to tens.

* **Coarse grids are voxel grids too.** A coarse element is a 2x2x2 block of
  the elements below it, and a fine node's displacement is trilinear between
  the coarse nodes around it. A coarse element's trilinear field is exactly a
  field its eight children can carry, so the Galerkin coarse stiffness
  ``P^T K P`` comes out element by element: a coarse element's 24x24 matrix is
  the sum of its children's, each seen through the fixed interpolation from
  where it sits in the block. Nothing is ever assembled on the fine grid, and
  empty voxels take no special handling: an empty child adds nothing.
* **Held degrees of freedom are the trap.** The fine solve is over the free
  ones only, so a child touching a support has to be seen with its held rows
  and columns struck out -- otherwise the coarse grid is stiffer than the part
  and the cycle stops being a good preconditioner near supports. Those few
  children are done one at a time; every other child is one of eight fixed
  matrices times its stiffness. A coarse DOF that interpolates only onto held
  fine ones gets a zero row, then a unit diagonal so it sits out. Coarse DOFs
  can also interpolate onto the same few free ones and leave the coarsest
  matrix singular, which is why it is factored with a small shift.
* **Chebyshev smoothing**, degree two, between a fifth of and just past the
  largest eigenvalue of ``D^-1 K``, which a few power steps estimate per level.
  Unlike damped Jacobi it needs no damping factor that is right for one part
  and unstable for the next.

The cycle is a fixed symmetric operator -- same smoothing before and after,
one solve at the bottom -- which is what CG needs of a preconditioner. Should
a part ever defeat it -- CG's residual stops falling -- :class:`~whittle.physics.fea.VoxelFEA`
gives up on it (:data:`STALL`) and solves that case with Jacobi.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from ..types import BoolArray, FloatArray, IntArray

Apply = Callable[[FloatArray], FloatArray]

_CORNERS = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=np.int64)

#: Below this many degrees of freedom a level is factored rather than coarsened.
COARSEST = 20_000
#: Up to this many, the coarsest grid is factored dense; above it, sparse.
DENSE = 3000
#: Added to the coarsest matrix's diagonal before factoring, relative to its largest entry.
SHIFT = 1e-9
#: Chebyshev degree of each smoothing pass, and the share of the spectrum it targets.
DEGREE = 2
SPECTRUM = (0.2, 1.1)
#: Power steps estimating each level's largest eigenvalue.
POWER_STEPS = 12
#: When :func:`pcg` gives up on multigrid, and the load case is done again with Jacobi
#: (which :attr:`~whittle.physics.fea.FEAResult.fell_back` reports): at once if the cycle
#: breaks down, which every broken cycle measured did within a few iterations; when its
#: best residual has fallen by less than a tenth over the last ``STALL`` iterations; or
#: after ``PATIENCE`` iterations, past which Jacobi would have been cheaper anyway. Jacobi
#: then starts from zero, not from multigrid's iterate: what multigrid leaves is error it
#: could not reduce, which was measured to cost Jacobi more than starting afresh.
#: Healthy solves of carved BESO designs fall by 2.3x or more in every 100 iterations;
#: a slow but steady one can sit flat for 100, never for 200.
STALL = 200
STALL_PROGRESS = 0.9
PATIENCE = 1000


@lru_cache(maxsize=64)
def _interpolation(factors: tuple[int, ...], position: tuple[int, ...]) -> FloatArray:
    """24x24: a child element's DOFs from its coarse element's, for the child at ``position``."""
    w = np.ones((8, 8), dtype=np.float64)
    for m, fine in enumerate(_CORNERS):
        for n, coarse in enumerate(_CORNERS):
            for d in range(3):
                t = (position[d] + fine[d]) / factors[d]
                w[m, n] *= t if coarse[d] else 1.0 - t
    p: FloatArray = np.asarray(np.kron(w, np.eye(3)), dtype=np.float64)
    p.setflags(write=False)
    return p


def _seen_from(k: FloatArray, p: FloatArray) -> FloatArray:
    """``p.T @ k[i] @ p`` for every symmetric ``k[i]``, as two large matrix products."""
    n = len(k)
    x = (k.reshape(n * 24, 24) @ p).reshape(n, 24, 24)
    # x[i] = k[i] p, so x[i].T = p.T k[i]: one more product with p finishes it.
    result: FloatArray = (x.transpose(0, 2, 1).reshape(n * 24, 24) @ p).reshape(n, 24, 24)
    return result


@dataclass
class _Coarsening:
    """One grid's elements gathered into the next one's."""

    factors: tuple[int, int, int]
    shape: tuple[int, int, int]
    #: Coarse element coordinates, one row per coarse element that has any child.
    elements: IntArray
    #: For each fine element: its coarse element's row, and where in the block it sits.
    parent: IntArray
    position: IntArray

    @property
    def positions(self) -> list[tuple[int, int, int]]:
        fx, fy, fz = self.factors
        return [(x, y, z) for z in range(fz) for y in range(fy) for x in range(fx)]


def _coarsen(elements: IntArray, shape: tuple[int, int, int]) -> _Coarsening:
    factors = tuple(2 if s > 1 else 1 for s in shape)
    f = np.array(factors)
    coarse_shape = tuple(int(s) for s in -(-np.array(shape) // f))
    parent_coords = elements // f
    pos = elements % f
    ids = np.ravel_multi_index(parent_coords.T, coarse_shape)
    unique, parent = np.unique(ids, return_inverse=True)
    position = pos[:, 0] + f[0] * (pos[:, 1] + f[1] * pos[:, 2])
    coords = np.stack(np.unravel_index(unique, coarse_shape), axis=1)
    return _Coarsening(factors, coarse_shape, coords, parent, position)  # type: ignore[arg-type]


@dataclass
class _Nodes:
    """A grid's nodes in use, numbered compactly, and each element's corners among them."""

    coords: IntArray
    corners: IntArray


def _nodes(elements: IntArray, shape: tuple[int, int, int]) -> _Nodes:
    grid = tuple(s + 1 for s in shape)
    corners = elements[:, None, :] + _CORNERS[None]
    ids = np.ravel_multi_index(corners.reshape(-1, 3).T, grid)
    unique, inverse = np.unique(ids, return_inverse=True)
    coords = np.stack(np.unravel_index(unique, grid), axis=1)
    return _Nodes(coords, inverse.reshape(-1, 8))


def _prolongation(
    fine: IntArray, coarse: _Nodes, factors: tuple[int, int, int], shape: tuple[int, int, int]
) -> sparse.csr_matrix:
    """Trilinear interpolation, node by node: rows are ``fine``'s nodes, columns ``coarse``'s."""
    f = np.array(factors)
    low = fine // f
    frac = (fine % f) / f
    grid = tuple(s + 1 for s in shape)
    lookup = np.ravel_multi_index(coarse.coords.T, grid)
    rows, cols, weights = [], [], []
    for corner in _CORNERS:
        w = np.prod(np.where(corner == 1, frac, 1.0 - frac), axis=1)
        used = w > 0
        ids = np.ravel_multi_index((low[used] + corner).T, grid)
        rows.append(np.flatnonzero(used))
        cols.append(np.searchsorted(lookup, ids))
        weights.append(w[used])
    return sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(fine), len(coarse.coords)),
    )


def _galerkin(k: FloatArray, step: _Coarsening) -> FloatArray:
    """The next grid's element matrices from this one's."""
    coarse = np.zeros((len(step.elements), 24, 24), dtype=np.float64)
    for index, position in enumerate(step.positions):
        children = np.flatnonzero(step.position == index)
        if len(children):
            # A coarse element has one child at each position: no two land on the same parent.
            p = _interpolation(step.factors, position)
            coarse[step.parent[children]] += _seen_from(k[children], p)
    return coarse


def _assemble(k: FloatArray, corners: IntArray) -> sparse.bsr_matrix:
    """The global matrix from element matrices, in 3x3 node blocks."""
    n, n_nodes = len(k), int(corners.max()) + 1
    rows = np.repeat(corners, 8, axis=1).ravel()
    cols = np.tile(corners, (1, 8)).ravel()
    keys, inverse = np.unique(rows * n_nodes + cols, return_inverse=True)
    blocks = k.reshape(n, 8, 3, 8, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9)
    data = np.stack(
        [np.bincount(inverse, weights=blocks[:, j], minlength=len(keys)) for j in range(9)], axis=1
    ).reshape(-1, 3, 3)
    block_rows, block_cols = keys // n_nodes, keys % n_nodes
    # A DOF that interpolates only onto held ones has nothing on its diagonal: give it a unit one.
    on_diagonal = np.flatnonzero(block_rows == block_cols)
    diagonal = data[on_diagonal][:, [0, 1, 2], [0, 1, 2]]
    empty = diagonal <= 1e-12 * diagonal.max()
    for d in range(3):
        data[on_diagonal[empty[:, d]], d, d] = 1.0
    indptr = np.concatenate([[0], np.cumsum(np.bincount(block_rows, minlength=n_nodes))])
    return sparse.bsr_matrix((data, block_cols, indptr), shape=(3 * n_nodes, 3 * n_nodes))


def _largest(apply: Apply, inverse_diagonal: FloatArray) -> float:
    """Estimate the largest eigenvalue of ``D^-1 K`` by power steps."""
    x = np.random.default_rng(0).standard_normal(len(inverse_diagonal))
    estimate = 1.0
    for _ in range(POWER_STEPS):
        kx = apply(x)
        # The Rayleigh quotient in D's inner product: never above the true value.
        estimate = float(x @ kx) / float(x @ (x / inverse_diagonal))
        x = inverse_diagonal * kx
        x /= np.linalg.norm(x)
    return estimate


def _chebyshev(
    apply: Apply, inverse_diagonal: FloatArray, largest: float, b: FloatArray, x: FloatArray | None
) -> FloatArray:
    """``DEGREE`` steps of Jacobi-preconditioned Chebyshev on ``K x = b``, from ``x`` or zero."""
    low, high = SPECTRUM[0] * largest, SPECTRUM[1] * largest
    theta, delta = (high + low) / 2, (high - low) / 2
    sigma = theta / delta
    rho = 1 / sigma
    r = b if x is None else b - apply(x)
    d = inverse_diagonal * r / theta
    x = d.copy() if x is None else x + d
    for _ in range(DEGREE - 1):
        r = r - apply(d)
        rho_next = 1 / (2 * sigma - rho)
        d = rho_next * rho * d + 2 * rho_next / delta * inverse_diagonal * r
        rho = rho_next
        x = x + d
    return x


def pcg(
    apply: Apply, cycle: Apply, b: FloatArray, *, rtol: float, maxiter: int
) -> tuple[FloatArray, str, int]:
    """CG on ``K x = b`` from zero, preconditioned by ``cycle``, watching for a stall.

    It stops where scipy's ``cg`` does, once ``|r| <= rtol |b|``. Returns the iterate, how
    the solve ended -- ``"converged"``, ``"stalled"`` (see :data:`STALL`), ``"broke down"``
    (the cycle or ``K`` was not positive along a direction, so the iterate is not to be
    trusted) or ``"out of iterations"`` -- and the iterations it took.
    """
    x = np.zeros_like(b)
    r = b.copy()
    target = rtol * float(np.linalg.norm(b))
    if float(np.linalg.norm(r)) <= target:
        return x, "converged", 0
    z = cycle(r)
    rz = float(r @ z)
    if not rz > 0:
        return x, "broke down", 0
    p = z
    norms = [float(np.linalg.norm(r))]
    for iteration in range(1, maxiter + 1):
        q = apply(p)
        pq = float(p @ q)
        if not pq > 0:
            return x, "broke down", iteration - 1
        step = rz / pq
        x += step * p
        r -= step * q
        norm = float(np.linalg.norm(r))
        if norm <= target:
            return x, "converged", iteration
        norms.append(norm)
        if _stalled(norms):
            return x, "stalled", iteration
        z = cycle(r)
        rz_next = float(r @ z)
        if not rz_next > 0:
            return x, "broke down", iteration
        p = z + (rz_next / rz) * p
        rz = rz_next
    return x, "out of iterations", maxiter


def _stalled(norms: list[float]) -> bool:
    """Whether a solve whose residual has gone ``norms`` (from the start) has stopped improving.

    Judged on the best residual so far, not the latest: CG's residual need not fall every
    step, and on a slender part can sit well above its best for a long while.
    """
    if len(norms) <= STALL:
        return False
    return min(norms) > STALL_PROGRESS * min(norms[:-STALL])


@dataclass
class _Level:
    apply: Apply
    inverse_diagonal: FloatArray
    largest: float
    #: To the next grid: residual down, correction up.
    restrict: Apply
    prolong: Apply


class Multigrid:
    """A V-cycle over the voxel grid, built for one set of solid voxels.

    ``elements`` are the solid voxels' grid coordinates, ``scale`` their stiffness relative
    to ``ke`` (None for all the same) and ``held`` which of their 24 DOFs are held. ``free``
    are the global DOFs CG solves for, in order; ``apply`` and ``inverse_diagonal`` are the
    fine grid's stiffness and its inverse diagonal over them.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        elements: IntArray,
        ke: FloatArray,
        scale: FloatArray | None,
        held: BoolArray,
        free: IntArray,
        apply: Apply,
        inverse_diagonal: FloatArray,
    ) -> None:
        step = _coarsen(elements, shape)
        nodes = _nodes(step.elements, step.shape)
        fine = _nodes(elements, shape).coords
        p = _prolongation(fine, nodes, step.factors, step.shape)
        in_use = np.ravel_multi_index(fine.T, tuple(s + 1 for s in shape))
        self.levels = [self._fine(apply, inverse_diagonal, p, free, in_use)]
        k = self._first(step, ke, scale, held)
        shape_now = step.shape
        while True:
            matrix = _assemble(k, nodes.corners)
            n_dof = matrix.shape[0]
            next_step = _coarsen(step.elements, shape_now)
            next_nodes = _nodes(next_step.elements, next_step.shape)
            if n_dof <= COARSEST or 3 * len(next_nodes.coords) > 0.8 * n_dof:
                break
            p = _prolongation(nodes.coords, next_nodes, next_step.factors, next_step.shape)
            self.levels.append(self._level(matrix, p))
            k = _galerkin(k, next_step)
            step, nodes, shape_now = next_step, next_nodes, next_step.shape
        self.coarsest = self._factor(matrix)
        #: Unknowns on each grid, finest first.
        self.sizes = [len(level.inverse_diagonal) for level in self.levels] + [matrix.shape[0]]

    @staticmethod
    def _fine(
        apply: Apply,
        inverse_diagonal: FloatArray,
        p: sparse.csr_matrix,
        free: IntArray,
        in_use: IntArray,
    ) -> _Level:
        """The fine grid: CG's free DOFs, over the nodes in use (``p``'s rows, by global id)."""
        slot = 3 * np.searchsorted(in_use, free // 3) + free % 3
        n_slots = 3 * p.shape[0]
        pt = p.T.tocsr()

        def restrict(r: FloatArray) -> FloatArray:
            spread = np.zeros(n_slots, dtype=np.float64)
            spread[slot] = r
            result: FloatArray = (pt @ spread.reshape(-1, 3)).ravel()
            return result

        def prolong(x: FloatArray) -> FloatArray:
            result: FloatArray = (p @ x.reshape(-1, 3)).ravel()[slot]
            return result

        return _Level(apply, inverse_diagonal, _largest(apply, inverse_diagonal), restrict, prolong)

    @staticmethod
    def _first(
        step: _Coarsening, ke: FloatArray, scale: FloatArray | None, held: BoolArray
    ) -> FloatArray:
        """The first coarse grid's element matrices, from the fine grid's one element matrix."""
        positions = step.positions
        interpolations = [_interpolation(step.factors, position) for position in positions]
        seen = np.stack([_seen_from(ke[None], p)[0] for p in interpolations])
        weight = np.zeros((len(step.elements), len(positions)), dtype=np.float64)
        stiffness = np.ones(len(step.parent)) if scale is None else scale
        weight[step.parent, step.position] = stiffness
        touching = np.flatnonzero(held.any(axis=1))
        weight[step.parent[touching], step.position[touching]] = 0.0
        k: FloatArray = (weight @ seen.reshape(len(positions), -1)).reshape(-1, 24, 24)
        if len(touching):
            free = ~held[touching]
            masked = ke[None] * (free[:, :, None] & free[:, None, :])
            masked *= stiffness[touching, None, None]
            for index, p in enumerate(interpolations):
                here = step.position[touching] == index
                if here.any():
                    k[step.parent[touching[here]]] += _seen_from(masked[here], p)
        return k

    @staticmethod
    def _level(matrix: sparse.bsr_matrix, p: sparse.csr_matrix) -> _Level:
        pt = p.T.tocsr()

        def apply(x: FloatArray) -> FloatArray:
            result: FloatArray = matrix @ x
            return result

        def restrict(r: FloatArray) -> FloatArray:
            result: FloatArray = (pt @ r.reshape(-1, 3)).ravel()
            return result

        def prolong(x: FloatArray) -> FloatArray:
            result: FloatArray = (p @ x.reshape(-1, 3)).ravel()
            return result

        inverse_diagonal = 1.0 / matrix.diagonal()
        return _Level(apply, inverse_diagonal, _largest(apply, inverse_diagonal), restrict, prolong)

    @staticmethod
    def _factor(matrix: sparse.bsr_matrix) -> Apply:
        # The coarsest matrix can be singular: where a thin part meets a support, coarse
        # DOFs can interpolate onto the same few free fine ones, and some combination of
        # them onto nothing at all. Neither factorization says so reliably -- SuperLU and
        # Cholesky both carry on through a pivot that is round-off, and hand back a solve
        # with huge, asymmetric errors that stall CG. A direction that interpolates to
        # nothing on the fine grid does not matter, so a small shift on the whole diagonal
        # makes the matrix definite without changing anything that does.
        shift = SHIFT * float(matrix.diagonal().max())
        if matrix.shape[0] <= DENSE:
            dense = matrix.toarray()
            dense[np.diag_indices_from(dense)] += shift
            factor = linalg.cho_factor(dense)
            return lambda b: linalg.cho_solve(factor, b)
        n = matrix.shape[0]
        lu = splu((matrix + shift * sparse.identity(n)).tocsc())
        return lu.solve  # type: ignore[no-any-return]

    def cycle(self, b: FloatArray, depth: int = 0) -> FloatArray:
        """One V-cycle on ``K x = b`` from zero: an approximation of ``K^-1 b``."""
        if depth == len(self.levels):
            return self.coarsest(b)
        level = self.levels[depth]
        x = _chebyshev(level.apply, level.inverse_diagonal, level.largest, b, None)
        r = b - level.apply(x)
        x = x + level.prolong(self.cycle(level.restrict(r), depth + 1))
        return _chebyshev(level.apply, level.inverse_diagonal, level.largest, b, x)
