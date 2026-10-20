"""Structural objectives: does the part actually carry its load?"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry.density import DensityField
from .base import Candidate


class AnchorConnectivity:
    """Penalize a candidate whose anchor regions are not joined by material.

    This is a **geometric proxy for structural validity, not a structural
    analysis**. It asks the much weaker question "is there a path of material
    from the load to the supports?" -- which is necessary for the part to carry
    anything, and nowhere near sufficient. A single-voxel thread satisfies it
    and would snap instantly.

    It earns its place for one reason: it makes the pipeline runnable today
    without an FEA solver, so the search loop, the backends and the export can
    be exercised and tested for real. Replacing it with :class:`Compliance` is
    the first item in ROADMAP.md, and the API is shaped so that it is a
    one-line substitution at the call site.

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
    """Strain energy under load: the real measure of stiffness per unit material.

    Not implemented. This is the centre of the project, and a placeholder is
    worse than an honest gap -- a stub that returned a plausible-looking number
    would let the whole pipeline produce confident, meaningless results.

    What it needs, in the order it should be built (see ROADMAP.md):

    1. A linear-elastic FEA solve on the voxel grid, trilinear hex elements,
       with density interpolated as ``E(rho) = E_min + rho**p * (E_0 - E_min)``
       -- the SIMP penalization, with ``p`` around 3, which is what pushes
       intermediate densities toward solid or void instead of grey mush.
    2. A conjugate-gradient solve with a matrix-free element-by-element
       operator. Assembling a global stiffness matrix for a 48x24x24 grid is
       possible; doing it once per candidate per generation is not.
    3. Reuse of the previous generation's solution as the starting guess, since
       candidates are mostly similar to their parents.

    Until this exists, :class:`AnchorConnectivity` stands in and the README says
    plainly that the structural claim is unproven.
    """

    name = "compliance"

    def __init__(self, *, load: object = None, supports: object = None) -> None:
        self.load = load
        self.supports = supports

    def __call__(self, candidate: Candidate) -> float:
        raise NotImplementedError(
            "Compliance needs an FEA solver, which is ROADMAP.md item 1. "
            "Use AnchorConnectivity for a geometric proxy in the meantime, and "
            "do not read a structural claim into results produced with it."
        )
