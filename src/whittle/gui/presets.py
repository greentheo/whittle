"""Example parts, each with a set-up ready to run: something to try before your own STL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .. import shapes
from ..types import Mesh
from ..voxel import Grid


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    description: str
    build: Callable[[], Mesh]
    #: The settings the page fills in: study kind, placed items, sliders.
    setup: dict[str, object] = field(default_factory=dict)


def _bracket() -> Mesh:
    grid = Grid((80, 8, 80), 1.0, (0.5, 0.5, 0.5))
    c = grid.centers()
    return grid.mesh(c[..., 2] >= c[..., 0] - 8.0, smooth=0.0)


def _hook() -> Mesh:
    """A solid L-shaped wall hook: a plate on the wall and an arm reaching out."""
    grid = Grid((60, 16, 70), 1.0, (0.5, 0.5, 0.5))
    c = grid.centers()
    x, z = c[..., 0], c[..., 2]
    plate = x <= 10.0
    arm = (z <= 18.0) & (x <= 60.0)
    lip = (x >= 50.0) & (z <= 30.0)
    return grid.mesh(plate | arm | lip, smooth=0.0)


PRESETS: list[Preset] = [
    Preset(
        "shelf-bracket",
        "Shelf bracket",
        "A solid 80 mm triangular bracket, 8 mm thick, screwed to a wall with 25 kg on its tip.",
        _bracket,
        {
            "kind": "load",
            "material": "pla",
            "resolution": 48,
            "removeUpTo": 0.8,
            "stations": 9,
            "supports": [{"at": [0.0, 4.0, 40.0], "normal": [-1.0, 0.0, 0.0]}],
            "loads": [
                {
                    "at": [76.0, 4.0, 80.0],
                    "normal": [0.0, 0.0, 1.0],
                    "radius": 6.0,
                    "force": [0.0, 0.0, -245.0],
                }
            ],
        },
    ),
    Preset(
        "wall-hook",
        "Wall hook",
        "A blocky L-shaped hook, screwed on through its back plate, holding 8 kg at the lip.",
        _hook,
        {
            "kind": "load",
            "material": "petg",
            "resolution": 48,
            "removeUpTo": 0.7,
            "stations": 8,
            "supports": [{"at": [0.0, 8.0, 40.0], "normal": [-1.0, 0.0, 0.0]}],
            "loads": [
                {
                    "at": [45.0, 8.0, 18.0],
                    "normal": [0.0, 0.0, 1.0],
                    "radius": 5.0,
                    "force": [0.0, 0.0, -78.5],
                }
            ],
        },
    ),
    Preset(
        "roof-box",
        "Roof box",
        "A square-edged 2 m roof box on a car driving towards +x, cargo space protected.",
        lambda: shapes.box((2000.0, 900.0, 400.0), center=(0.0, 0.0, 200.0), cell=50.0),
        {
            "kind": "flow",
            "flow": [-1.0, 0.0, 0.0],
            "resolution": 40,
            "detail": 2,
            "mirror": "y",
            "flat": ["-z"],
            "extruded": None,
            "speedKmh": 110.0,
            "removeUpTo": 0.25,
            "stations": 6,
            "keepBoxes": [[[-750.0, -350.0, 0.0], [750.0, 350.0, 280.0]]],
        },
    ),
    Preset(
        "roof-bar",
        "Roof-rack bar",
        "An 80 mm slice of a plain 40 x 25 mm bar, scored as a section of an endless one.",
        lambda: shapes.box((40.0, 80.0, 25.0), cell=2.0),
        {
            "kind": "flow",
            "flow": [-1.0, 0.0, 0.0],
            "resolution": 40,
            "detail": 2,
            "mirror": None,
            "flat": [],
            "extruded": "y",
            "speedKmh": 110.0,
            "removeUpTo": 0.55,
            "stations": 5,
            "keepBoxes": [],
        },
    ),
]


def preset(preset_id: str) -> Preset:
    for p in PRESETS:
        if p.id == preset_id:
            return p
    raise KeyError(preset_id)
