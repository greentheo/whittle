"""Bi-directional evolutionary structural optimization: whittling, literally.

Start from the whole part. Solve for where it is working -- the strain energy in
every voxel. Remove the voxels doing the least work, let the voxels beside the
hardest-working ones grow back, and repeat, a couple of percent at a time
(Huang and Xie, *Evolutionary Topology Optimization of Continuum Structures*,
2010).

Why this rather than :func:`whittle.optimize.evolve` for structures: the FEA
solve tells every voxel how much it matters, which is far more information than
a fitness value, and using it directly takes about a hundred solves where a
population search over the same design would take tens of thousands. Its other
virtue is the one the slider needs. Its history *is* a sweep: every iteration
is a lighter version of the one before, so pausing at each removal level gives
a run of designs that morph smoothly from the original part to the most
aggressive one.

What it cannot do is score things the solver does not see -- printability,
overhangs, a minimum wall thickness beyond what the filter imposes. Those stay
the job of the population search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..physics.fea import FEAResult, VoxelFEA
from ..types import BoolArray, FloatArray


@dataclass
class BesoStation:
    removed: float
    solid: BoolArray
    result: FEAResult


@dataclass
class BesoRun:
    stations: list[BesoStation]
    #: Why the run stopped short of the requested removal, or empty if it did not.
    note: str
    n_solves: int


def beso(
    fea: VoxelFEA,
    domain: BoolArray,
    *,
    removals: Sequence[float],
    keep: BoolArray | None = None,
    evolution_rate: float = 0.02,
    filter_radius: float = 1.5,
    settle: int = 8,
    tol: float = 0.002,
    callback: Callable[[int, float, float], None] | None = None,
) -> BesoRun:
    """Remove material from ``domain`` in steps, pausing at each of ``removals``.

    ``removals`` are fractions of the domain's material, and each pause is a
    station: up to ``settle`` iterations at constant volume, so the design at
    that level has finished rearranging before it is recorded. ``keep`` is
    material that must stay; every load and support region is kept
    regardless.

    ``filter_radius`` (in voxels) is the length scale of the result. Below
    about 1.5 the design breaks into checkerboards and one-voxel threads;
    larger values give fewer, thicker members.

    ``callback(solve_count, removed, compliance)`` is called after every
    solve, for a progress bar.
    """
    domain = np.asarray(domain, dtype=bool)
    if domain.shape != fea.shape:
        raise ValueError(f"domain must have shape {fea.shape}, got {domain.shape}")
    if not 0 < evolution_rate < 1:
        raise ValueError(f"evolution_rate must be in (0, 1), got {evolution_rate}")
    targets = sorted({float(r) for r in removals})
    if not targets or targets[0] < 0 or targets[-1] >= 1:
        raise ValueError("removals must be fractions in [0, 1)")

    forced = np.zeros(fea.shape, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    for load in fea.loads:
        forced |= load.region
    for support in fea.supports:
        forced |= support.region
    forced &= domain
    removable = domain & ~forced
    n_domain = int(domain.sum())
    n_forced = int(forced.sum())

    kernel = _filter_kernel(filter_radius)
    coverage = ndimage.convolve(domain.astype(np.float64), kernel, mode="constant")
    coverage[coverage == 0] = 1.0

    solves = 0

    def solve(solid: BoolArray) -> FEAResult:
        nonlocal solves
        result = fea.solve(solid)
        solves += 1
        if callback is not None:
            callback(solves, 1.0 - int(solid.sum()) / n_domain, result.compliance)
        return result

    history: FloatArray | None = None

    def sensitivity(result: FEAResult) -> FloatArray:
        # Filtered strain energy. The filter is what lets an empty voxel grow
        # back: it has no energy of its own, but inherits some from working
        # neighbours. Averaging with the previous iteration damps the
        # oscillation a hard-kill method is otherwise prone to.
        nonlocal history
        spread: FloatArray = (
            ndimage.convolve(result.strain_energy, kernel, mode="constant") / coverage
        )
        if history is not None:
            spread = 0.5 * (spread + history)
        history = spread
        return spread

    def select(alpha: FloatArray, count: int) -> BoolArray:
        n_free = count - n_forced
        solid: BoolArray = forced.copy()
        if n_free <= 0:
            return solid
        index = np.flatnonzero(removable)
        scores = alpha.ravel()[index]
        chosen = index[np.argpartition(-scores, n_free - 1)[:n_free]]
        solid.ravel()[chosen] = True
        return solid

    solid = domain.copy()
    current = solve(solid)
    if not current.supported:
        raise ValueError("the original part does not connect every load to a support")

    stations: list[BesoStation] = []
    note = ""
    for target in targets:
        count = max(int(round((1.0 - target) * n_domain)), n_forced)
        # Descend to this station's volume, a step at a time.
        while int(solid.sum()) > count:
            alpha = sensitivity(current)
            n = int(solid.sum())
            step = max(count, min(n - 1, int(round(n * (1.0 - evolution_rate)))))
            candidate = select(alpha, step)
            result = solve(candidate)
            if not result.supported:
                # Too big a bite severed the load path. Try half the step once.
                step = max(count, n - max(1, (n - step) // 2))
                candidate = select(alpha, step)
                result = solve(candidate)
            if not result.supported:
                note = (
                    f"stopped at {1 - n / n_domain:.0%} removed: going further "
                    "disconnects a load from every support"
                )
                # The furthest design that still holds together is the end
                # of the slider: settle it and record it, then stop.
                count = n
                break
            solid, current = candidate, result
        # Settle at constant volume, keeping the stiffest design seen.
        best_solid, best = solid, current
        recent = [current.compliance]
        for _ in range(settle):
            candidate = select(sensitivity(current), count)
            if np.array_equal(candidate, solid):
                break
            result = solve(candidate)
            if not result.supported:
                break
            solid, current = candidate, result
            if current.compliance < best.compliance:
                best_solid, best = solid, current
            recent.append(current.compliance)
            if len(recent) >= 3 and np.ptp(recent[-3:]) <= tol * recent[-1]:
                break
        solid, current = best_solid, best
        stations.append(BesoStation(1.0 - int(solid.sum()) / n_domain, solid.copy(), current))
        if note:
            break
    return BesoRun(stations, note, solves)


def _filter_kernel(radius: float) -> FloatArray:
    """Linear-decay weights within ``radius`` voxels: the usual sensitivity filter."""
    reach = max(1, int(np.ceil(radius)) - 1) if radius > 1 else 0
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    weights: FloatArray = np.maximum(0.0, radius - np.sqrt(dx**2 + dy**2 + dz**2))
    return weights
