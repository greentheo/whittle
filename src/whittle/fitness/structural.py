"""Structural objectives: does the part actually carry its load?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from ..geometry.density import DensityField
from ..physics.fea import Load, Material, Support, VoxelFEA
from .base import Candidate


class AnchorConnectivity:
    """Penalize a candidate whose anchor regions are not joined by material.

    This is a **geometric proxy for structural validity, not a structural
    analysis**. It asks the much weaker question "is there a path of material
    from the load to the supports?" -- which is necessary for the part to carry
    anything, and nowhere near sufficient. A single-voxel thread satisfies it
    and would snap instantly.

    It still earns a place now that :class:`Compliance` exists: it costs a
    labelling pass instead of an FEA solve, so it can cheaply steer an early
    search towards connected designs before the expensive objective takes
    over, or stand in on a grid too large to solve thousands of times.

    When the anchors are disconnected the penalty is the distance still to be
    bridged, normalized by the grid diagonal, rather than a flat failure. A flat
    penalty is a cliff, and an evolutionary search gets no information from a
    cliff; a distance tells it which proposals were closer to joining up.
    """

    name = "connectivity"

    def __init__(self, anchors: list[np.ndarray], *, weight: float = 1.0) -> None:
        if len(anchors) < 2:
            raise ValueError("need at least two anchor regions to connect")
        self.anchors = [np.asarray(a, dtype=bool) for a in anchors]
        self.weight = float(weight)

    def __call__(self, candidate: Candidate) -> float:
        backend = candidate.backend
        if not isinstance(backend, DensityField):
            raise TypeError("AnchorConnectivity requires a DensityField backend")
        solid = candidate.field >= backend.threshold
        labels, n_components = ndimage.label(solid)
        if n_components == 0:
            return self.weight

        anchor_labels = [set(np.unique(labels[a & solid])) - {0} for a in self.anchors]
        if any(not labs for labs in anchor_labels):
            # An anchor has no material in it at all. Nothing to measure a
            # distance from, so this is the worst case.
            return self.weight

        root = anchor_labels[0]
        unreached = [i for i, labs in enumerate(anchor_labels[1:], start=1) if not (labs & root)]
        if not unreached:
            return 0.0

        reached = np.isin(labels, list(root))
        # Distance from every voxel to the nearest voxel of the root component.
        distance = ndimage.distance_transform_edt(~reached)
        diagonal = float(np.linalg.norm(np.array(backend.shape, dtype=np.float64)))
        gaps = [float(distance[self.anchors[i]].min()) / diagonal for i in unreached]
        return self.weight * float(np.mean(gaps))


class Compliance:
    """Work done by the loads -- the standard measure of how stiff a part is.

    Scored on the material a candidate would actually print: the field
    thresholded at the backend's threshold, solid or nothing, not the grey
    intermediate densities. What is meshed and exported is the thresholded
    geometry, so that is what the physics should judge.

    Reported relative to the design box filled solid wherever ``void`` allows
    -- the stiffest thing that box can be -- so it is 1.0 at best and climbs
    as material goes, and sits beside :class:`~whittle.fitness.mass.MaterialFraction`
    in a :class:`~whittle.fitness.base.WeightedSum` without a scale factor.

    A candidate whose loads are not joined to a support by material has no
    static answer; it scores ``unsupported``, high enough that any connected
    design beats it.

    This is the objective for a population search (:func:`whittle.evolve`)
    over a :class:`~whittle.geometry.density.DensityField`. It pays one FEA
    solve per candidate, so keep the grid small. To lighten a part, the
    sensitivity-driven :func:`whittle.optimize.beso.beso`, which
    :class:`whittle.LoadStudy` runs, needs a hundred solves where this would
    need tens of thousands.
    """

    name = "compliance"

    def __init__(
        self,
        *,
        loads: Sequence[Load],
        supports: Sequence[Support],
        material: str | Material = "pla",
        unsupported: float = 1e3,
    ) -> None:
        self.loads = list(loads)
        self.supports = list(supports)
        self.material = material
        self.unsupported = float(unsupported)
        self._backend: DensityField | None = None
        self._fea: VoxelFEA | None = None
        self._reference = 1.0

    def solver(self, backend: DensityField) -> VoxelFEA:
        """The solve for this backend's grid, built once and reused."""
        if self._fea is None or self._backend is not backend:
            fea = VoxelFEA(
                backend.shape,
                spacing=backend.spacing,
                material=self.material,
                loads=self.loads,
                supports=self.supports,
            )
            filled = np.ones(backend.shape, dtype=bool)
            if backend.void is not None:
                filled &= ~backend.void
            reference = fea.solve(filled)
            if not reference.supported:
                raise ValueError("even the full design box does not connect the loads to a support")
            self._fea, self._backend, self._reference = fea, backend, reference.compliance
        return self._fea

    def __call__(self, candidate: Candidate) -> float:
        backend = candidate.backend
        if not isinstance(backend, DensityField):
            raise TypeError("Compliance requires a DensityField backend")
        result = self.solver(backend).solve(candidate.field >= backend.threshold, rtol=1e-4)
        if not result.supported:
            return self.unsupported
        return result.compliance / self._reference
