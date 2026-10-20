"""Fluid objectives: how hard is it to push this shape through air or water?"""

from __future__ import annotations

from .base import Candidate


class DragCoefficient:
    """Drag on the candidate under a prescribed flow.

    Not implemented, and the honest reason is that this one is not a weekend's
    work the way the structural solver is. A usable answer needs a real CFD
    solve per candidate, and an evolutionary search wants thousands of them.

    The three routes worth weighing before writing any code (ROADMAP.md item 4):

    * **Shell out to OpenFOAM per candidate.** Trustworthy, and far too slow to
      sit inside a generation loop unless the population is tiny and the budget
      is hours.
    * **A fast approximate solver -- panel method or lattice Boltzmann.** Good
      enough to rank candidates against each other, which is all a selection
      step actually needs. Probably where this should start.
    * **A surrogate trained on a few hundred real solves.** Fastest per
      evaluation by far, and the one that most easily produces confident
      nonsense once the search walks outside the training distribution.

    Note that for shape optimization the geometry side is already settled: use
    :class:`~whittle.geometry.ffd.FFDLattice`, which keeps the surface closed
    and the mesh valid for whatever solver you land on.
    """

    name = "drag"

    def __init__(self, *, velocity: float = 1.0, density: float = 1.225) -> None:
        self.velocity = float(velocity)
        self.density = float(density)

    def __call__(self, candidate: Candidate) -> float:
        raise NotImplementedError(
            "DragCoefficient needs a flow solver, which is ROADMAP.md item 4. "
            "See the class docstring for the three approaches and their trade-offs."
        )
