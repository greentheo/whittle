"""Fluid objectives: how hard is it to push this shape through air or water?

Both are scored by :class:`~whittle.physics.drag.DragModel`, a fast engineering
estimate rather than a flow solve. Read its docstring before trusting a number
from either: it ranks shapes well and gets absolute drag to perhaps 30%.

Use these with :class:`~whittle.geometry.ffd.FFDLattice`, which keeps the
surface closed and smooth. A density field's marching-cubes surface is
terraced, and the drag model reads its normals.
"""

from __future__ import annotations

from ..physics.drag import DragModel
from ..types import Vector3
from .base import Candidate


class DragArea:
    """Drag area ``Cd * A`` -- what the drag force is proportional to.

    Prefer this to :class:`DragCoefficient` as the thing to minimize. A shape
    can lower its coefficient by growing a bigger frontal area, which makes the
    number better and the fuel bill worse.

    ``reference`` divides the result, so it sits near 1 next to other
    objectives in a :class:`~whittle.fitness.base.WeightedSum`; pass the
    starting shape's drag area.
    """

    name = "drag"

    def __init__(self, flow: Vector3 | DragModel, *, reference: float = 1.0) -> None:
        self.model = flow if isinstance(flow, DragModel) else DragModel(flow)
        self.reference = float(reference)

    def __call__(self, candidate: Candidate) -> float:
        return self.model(candidate.mesh).drag_area / self.reference


class DragCoefficient:
    """Drag coefficient on the candidate's own frontal area."""

    name = "cd"

    def __init__(self, flow: Vector3 | DragModel) -> None:
        self.model = flow if isinstance(flow, DragModel) else DragModel(flow)

    def __call__(self, candidate: Candidate) -> float:
        return self.model(candidate.mesh).cd
