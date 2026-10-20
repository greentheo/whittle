"""The fitness seam.

Every objective returns a float where **lower is better**, so the optimizer
never has to know which direction any particular physics points in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..geometry.base import FieldBackend, GeometryBackend
from ..types import FloatArray, Genome, Mesh


class Candidate:
    """One genome, plus the geometry it decodes to.

    Decoding is lazy and cached. An objective that only needs the density field
    never pays to mesh, and two objectives in the same evaluation share one
    decode rather than repeating it -- which is most of the cost of a
    generation once the grid gets large.
    """

    def __init__(self, genome: Genome, backend: GeometryBackend) -> None:
        self.genome = np.asarray(genome, dtype=np.float64)
        self.backend = backend
        self._mesh: Mesh | None = None
        self._field: FloatArray | None = None

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = self.backend.build(self.genome)
        return self._mesh

    @property
    def field(self) -> FloatArray:
        if not isinstance(self.backend, FieldBackend):
            raise TypeError(
                f"{type(self.backend).__name__} exposes no density field; "
                "this objective needs a field backend such as DensityField"
            )
        if self._field is None:
            self._field = self.backend.field(self.genome)
        return self._field


@runtime_checkable
class Objective(Protocol):
    """A scalar to minimize."""

    #: Short label, used in reporting and history.
    name: str

    def __call__(self, candidate: Candidate) -> float:
        ...


class WeightedSum:
    """Scalarize several objectives into one.

    The simple thing, and enough to get a result. It is also the honest limit of
    this version: a weighted sum can only ever find one point on the trade-off
    between material and stiffness, and which point depends on weights you had
    to guess. Searching the whole front at once is a real Pareto method, and is
    the headline item in ROADMAP.md.
    """

    def __init__(self, objectives: dict[Objective, float]) -> None:
        if not objectives:
            raise ValueError("need at least one objective")
        self.objectives = objectives

    @property
    def name(self) -> str:
        return " + ".join(f"{w:g}*{o.name}" for o, w in self.objectives.items())

    def terms(self, candidate: Candidate) -> dict[str, float]:
        """Each objective's raw, unweighted value. For reporting, not for search."""
        return {o.name: o(candidate) for o in self.objectives}

    def __call__(self, candidate: Candidate) -> float:
        return float(sum(weight * obj(candidate) for obj, weight in self.objectives.items()))
