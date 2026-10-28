"""The search loop.

Differential evolution, because it is the method that actually suits this
problem: no gradients to derive through a meshing step, no requirement that the
objective be differentiable or even continuous, and an embarrassingly parallel
population. Those are the same properties that motivated EML, and swapping
scipy's DE for that one is the point of the ``optimizer`` seam below.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution

from ..fitness.base import Candidate, Objective
from ..geometry.base import GeometryBackend
from ..types import FloatArray, Genome, Mesh


@dataclass
class EvolutionResult:
    """What a run produced, and enough context to judge whether to believe it."""

    genome: Genome
    fitness: float
    backend: GeometryBackend
    n_evaluations: int
    history: list[float] = field(default_factory=list)
    terms: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    message: str = ""

    @property
    def mesh(self) -> Mesh:
        return self.backend.build(self.genome)

    def __str__(self) -> str:
        terms = ", ".join(f"{k}={v:.4f}" for k, v in self.terms.items())
        return (
            f"fitness={self.fitness:.6f} after {self.n_evaluations} evaluations"
            + (f" ({terms})" if terms else "")
            + (f" -- {self.message}" if self.message else "")
        )


def evolve(
    backend: GeometryBackend,
    objective: Objective,
    *,
    seed: int | None = None,
    maxiter: int = 60,
    popsize: int = 12,
    tol: float = 1e-4,
    mutation: tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    workers: int = 1,
    polish: bool = False,
    x0: Genome | None = None,
    spread: float | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> EvolutionResult:
    """Evolve a genome for ``backend`` that minimizes ``objective``.

    ``polish`` defaults to False, unlike scipy's own default. The polish step is
    a gradient-based local refinement, and our objective has no useful gradient
    -- it runs a meshing pass and a connectivity test. Letting it run burns
    evaluations to no purpose.

    Parameters whose bounds have zero width are held fixed and excluded from the
    search, which is what makes ``FFDLattice(frozen_axes=...)`` do anything.

    ``x0`` warm-starts the search from a known genome. With ``spread`` as
    well, the whole initial population is drawn around ``x0`` -- uniformly
    within ``spread`` of each parameter's range either side -- rather than
    across the full bounds. That is the right start for refining a design
    that is already good, and it is what keeps neighbouring stations of a
    sweep looking like each other.
    """
    bounds = list(backend.bounds)
    if len(bounds) != backend.n_params:
        raise ValueError(
            f"backend reports {backend.n_params} params but {len(bounds)} bounds"
        )

    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)
    free = highs > lows
    if not free.any():
        raise ValueError("every parameter is frozen; nothing to search")

    def expand(reduced: Sequence[float] | FloatArray) -> Genome:
        full: FloatArray = lows.copy()
        full[free] = np.asarray(reduced, dtype=np.float64)
        return full

    n_evaluations = 0

    def cost(reduced: FloatArray) -> float:
        nonlocal n_evaluations
        n_evaluations += 1
        return float(objective(Candidate(expand(reduced), backend)))

    history: list[float] = []

    def on_step(xk: FloatArray, convergence: float = 0.0) -> bool:
        value = cost(np.asarray(xk, dtype=np.float64))
        history.append(value)
        if callback is not None:
            callback(len(history), value)
        return False

    init: str | FloatArray = "latinhypercube"
    start: FloatArray | None = None
    if x0 is not None:
        start = np.clip(np.asarray(x0, dtype=np.float64).ravel(), lows, highs)[free]
        if spread is not None:
            rng = np.random.default_rng(seed)
            width = (highs - lows)[free]
            n = max(5, popsize * int(free.sum()))
            population = start + rng.uniform(-spread, spread, (n, int(free.sum()))) * width
            init = np.clip(population, lows[free], highs[free])
            init[0] = start
            start = None

    outcome = differential_evolution(
        cost,
        bounds=[(float(lo), float(hi)) for lo, hi in zip(lows[free], highs[free], strict=True)],
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        workers=workers,
        polish=polish,
        init=init,
        x0=start,
        callback=on_step,
    )

    genome = expand(outcome.x)
    candidate = Candidate(genome, backend)
    terms = objective.terms(candidate) if hasattr(objective, "terms") else {}
    return EvolutionResult(
        genome=genome,
        fitness=float(outcome.fun),
        backend=backend,
        n_evaluations=n_evaluations,
        history=history,
        terms=terms,
        converged=bool(outcome.success),
        message=str(outcome.message),
    )
