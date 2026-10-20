"""How much material a candidate spends. The term you are trying to drive down."""

from __future__ import annotations

import numpy as np

from .base import Candidate


class MaterialFraction:
    """Mean density over the design box, in ``[0, 1]``.

    Reads the continuous field rather than the meshed volume, deliberately. The
    field moves smoothly as the genome moves; meshed volume steps as triangles
    appear and disappear, and those steps read to an evolutionary search as
    noise rather than as progress.
    """

    name = "material"

    def __call__(self, candidate: Candidate) -> float:
        return float(np.mean(candidate.field))


class MeshVolume:
    """Enclosed volume of the meshed candidate, in model units cubed.

    Works with any backend, including deformation lattices that have no field.
    Prefer :class:`MaterialFraction` when the backend offers one.
    """

    name = "volume"

    def __call__(self, candidate: Candidate) -> float:
        return abs(candidate.mesh.volume)
