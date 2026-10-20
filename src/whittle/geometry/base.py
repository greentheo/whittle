"""The geometry seam.

A backend turns a float vector into a mesh. That is the entire contract, and it
is what lets one optimizer drive two very different kinds of search.

``changes_topology`` is the distinction that matters, and the reason this
library has two backends rather than one:

* A density field can open a hole or grow an internal lattice, so it can find a
  structure you did not think of. That is what minimizing material per unit of
  strength actually requires.
* A deformation lattice can only push an existing surface around. It cannot
  make a hole -- no arrangement of control points will do it -- but it keeps the
  input watertight and printable, which is what you want when you are tuning a
  shape against a flow.

Pick the backend that matches the question. Asking a lattice to find a truss is
asking it to do something it provably cannot.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import FloatArray, Genome, Mesh


@runtime_checkable
class GeometryBackend(Protocol):
    """Decodes a genome into geometry."""

    #: Whether this representation can change the topology of the result.
    changes_topology: bool

    @property
    def n_params(self) -> int:
        """Length of the genome this backend expects."""
        ...

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Per-parameter ``(low, high)``, as the optimizer wants them."""
        ...

    def build(self, genome: Genome) -> Mesh:
        """Decode a genome into a mesh."""
        ...


@runtime_checkable
class FieldBackend(GeometryBackend, Protocol):
    """A backend that also exposes a continuous scalar field.

    Objectives should prefer the field where one exists. It is far cheaper than
    meshing, and -- more importantly -- it is continuous, so a small change in
    the genome makes a small change in the objective. Meshed volume jumps as
    triangles appear and vanish, which is exactly the kind of cliff that makes
    an evolutionary search waste its budget.
    """

    def field(self, genome: Genome) -> FloatArray:
        """Decode a genome into a density grid in ``[0, 1]``."""
        ...
