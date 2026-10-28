"""The slider: a run of designs from conservative to extreme.

Neither physics model here is accurate enough to pick *the* answer -- a single
optimum computed against a load you estimated, on a material you idealized, is
false precision. What they are good for is the shape of the trade-off: how much
stiffness the first 20% of material removal costs, where the drag savings stop
being worth the volume. So a run produces a :class:`Sweep` -- a sequence of
designs at increasing amounts of material removed -- and choosing how far
along it to go is left to the person who will build the part.

Stations are ordered from least to most material removed, and each one evolves
from the one before it, so neighbouring stations look alike and sliding along
them reads as a part being whittled down rather than as unrelated designs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from .types import Mesh


@dataclass
class Station:
    """One stop on the slider: a design, and what it costs and buys."""

    #: Fraction of the original part's material removed, in ``[0, 1)`` --
    #: or negative, for a parametric design whose bounds let it grow.
    removed: float
    #: Everything measured about this design. Keys depend on the study.
    metrics: dict[str, float]
    #: Builds the mesh on demand -- most stations are looked at, not printed.
    build: Callable[[], Mesh] = field(repr=False)
    #: The raw design: a voxel occupancy for a load study, a genome for a flow
    #: study, a dict of parameter values for a parametric one.
    design: object = field(default=None, repr=False)
    #: What scored it: an :class:`~whittle.physics.fea.FEAResult` for a load
    #: study, a :class:`~whittle.physics.drag.DragReport` for a flow study.
    #: The stress and per-face drag in these are what a heatmap is drawn from.
    analysis: object = field(default=None, repr=False)
    _mesh: Mesh | None = field(default=None, init=False, repr=False)

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = self.build()
        return self._mesh


@dataclass(frozen=True)
class Column:
    """How to print one metric in a sweep's table, or with ``design`` one entry of its design."""

    key: str
    label: str
    format: str
    design: bool = False


@dataclass
class Sweep:
    """Designs at increasing amounts of material removed. Index 0 is the original part."""

    kind: str
    stations: list[Station]
    columns: list[Column] = field(default_factory=list)
    #: Why the sweep stopped short of the removal that was asked for, if it did.
    note: str = ""

    def __post_init__(self) -> None:
        if not self.stations:
            raise ValueError("a sweep needs at least one station")

    def __len__(self) -> int:
        return len(self.stations)

    def __iter__(self) -> Iterator[Station]:
        return iter(self.stations)

    def __getitem__(self, index: int) -> Station:
        return self.stations[index]

    @property
    def original(self) -> Station:
        return self.stations[0]

    def at(self, removed: float) -> Station:
        """The station closest to removing ``removed`` of the original material."""
        gaps = [abs(s.removed - removed) for s in self.stations]
        return self.stations[int(np.argmin(gaps))]

    def slide(self, position: float) -> Station:
        """The station at ``position`` along the slider: 0 is conservative, 1 is extreme."""
        position = float(np.clip(position, 0.0, 1.0))
        lo, hi = self.stations[0].removed, self.stations[-1].removed
        return self.at(lo + position * (hi - lo))

    def table(self) -> str:
        """The sweep as a fixed-width table, one row per station."""
        columns = [Column("removed", "removed", "{:.0%}"), *self.columns]
        rows = [[c.label for c in columns]]
        for station in self.stations:
            values: dict[str, object] = {"removed": station.removed, **station.metrics}
            design = station.design if isinstance(station.design, dict) else {}
            rows.append(
                [_cell(c, design.get(c.key) if c.design else values.get(c.key)) for c in columns]
            )
        widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
        lines = [
            "  ".join(cell.rjust(w) for cell, w in zip(row, widths, strict=True)) for row in rows
        ]
        lines.insert(1, "  ".join("-" * w for w in widths))
        if self.note:
            lines.append(f"\n{self.note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def _cell(column: Column, value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float | int) and not np.isfinite(value):
        return "inf"
    return column.format.format(value)
