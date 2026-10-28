"""Tune a parametric hitch box's own dimensions, in free air and behind the car that tows it.

Run:  python examples/06_parametric_hitch_box.py

A cargo box on a hitch mount: 39" deep, 74" wide, 41" tall (991 x 1880 x
1041 mm), its front 18" behind the bumper and its floor 18" off the road. The
car drives towards +x, so the air comes from the front: ``flow=(-1, 0, 0)``.

Instead of carving a mesh, this searches five numbers of a model built from
rounded cross-sections lofted front to back -- the kind of thing a CadQuery
script describes:

* ``front``   radius rounding the top and side edges of the front face;
* ``corner``  radius of the two long top edges;
* ``tail``    how far forward of the back the tail starts to taper;
* ``drop``    how far the top falls by the very back;
* ``pinch``   how far each side draws in by the very back.

The floor stays flat and the front stays where it is, because no parameter
moves them. Every station is a real design from the model, as smooth as the
model makes it, and what it hands back is five numbers to build to.

The same search runs twice. In free air the square front takes most of the
drag, so rounding it pays. Behind the car -- a plain slab standing in for a
Nissan Leaf's back end -- the front sits in the car's wake, and what is left
to win is in the tail.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from whittle import Mesh, ParametricStudy, io, shapes

OUT = Path(__file__).resolve().parent.parent / "out"

DEPTH, WIDTH, HEIGHT = 991.0, 1880.0, 1041.0
FRONT_X = -457.0
FLOOR_Z = 457.0


def hitch_box(p: dict[str, float]) -> Mesh:
    """The box, lofted through rounded-rectangle sections from front to back."""
    r, tail = p["front"], p["tail"]
    # Sections closely spaced round the front arc, then every 40 mm or so.
    theta = np.linspace(0.0, np.pi / 2.0, 9)
    rest = np.linspace(r, DEPTH, max(2, int(np.ceil((DEPTH - r) / 40.0))) + 1)[1:]
    depth = np.concatenate([r * (1.0 - np.cos(theta)), rest])
    inset = np.concatenate([r * (1.0 - np.sin(theta)), np.zeros(len(rest))])
    taper = np.clip((depth - (DEPTH - tail)) / tail, 0.0, 1.0) ** 2
    top = HEIGHT - inset - p["drop"] * taper
    half = WIDTH / 2.0 - inset - p["pinch"] * taper

    rings = [_section(h, t, p["corner"]) for h, t in zip(half, top, strict=True)]
    n = len(rings[0])
    points = np.concatenate(
        [
            np.column_stack([np.full(n, FRONT_X - d), ring[:, 0], FLOOR_Z + ring[:, 1]])
            for d, ring in zip(depth, rings, strict=True)
        ]
    )
    faces = []
    i = np.arange(n)
    j = (i + 1) % n
    for k in range(len(rings) - 1):
        a, b = k * n + i, (k + 1) * n + i
        c, d = (k + 1) * n + j, k * n + j
        faces += [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
    # Flat caps, fanned from a centre point at each end.
    centres = np.array(
        [
            [FRONT_X, 0.0, FLOOR_Z + top[0] / 2.0],
            [FRONT_X - DEPTH, 0.0, FLOOR_Z + top[-1] / 2.0],
        ]
    )
    front, back = len(points), len(points) + 1
    last = (len(rings) - 1) * n
    faces += [
        np.column_stack([np.full(n, front), i, j]),
        np.column_stack([np.full(n, back), last + j, last + i]),
    ]
    mesh = Mesh(np.concatenate([points, centres]), np.concatenate(faces).astype(np.int64))
    if mesh.volume < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    return mesh


def _section(half: float, top: float, corner: float) -> np.ndarray:
    """A section's outline, (y, z) from the floor: square at the bottom, rounded on top."""
    rc = min(corner, 0.95 * half, 0.95 * top)

    def line(a: tuple[float, float], b: tuple[float, float], count: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
        return np.asarray(a) + t * (np.asarray(b) - np.asarray(a))

    def arc(cy: float, cz: float, start: float, count: int) -> np.ndarray:
        angle = start + np.linspace(0.0, np.pi / 2.0, count, endpoint=False)
        return np.column_stack([cy + rc * np.cos(angle), cz + rc * np.sin(angle)])

    return np.concatenate(
        [
            line((-half, 0.0), (half, 0.0), 16),
            line((half, 0.0), (half, top - rc), 12),
            arc(half - rc, top - rc, 0.0, 8),
            line((half - rc, top), (-(half - rc), top), 16),
            arc(-(half - rc), top - rc, np.pi / 2.0, 8),
            line((-half, top - rc), (-half, 0.0), 12),
        ]
    )


BOUNDS = {
    "front": (20.0, 300.0),
    "corner": (20.0, 300.0),
    "tail": (100.0, 600.0),
    "drop": (0.0, 400.0),
    "pinch": (0.0, 300.0),
}
#: The box as first drawn: square but for a 20 mm round-over on every edge.
START = {"front": 20.0, "corner": 20.0, "tail": 300.0, "drop": 0.0, "pinch": 0.0}

#: The back of the car, as a slab: 1770 mm wide, from 150 mm to 1550 mm up.
CAR = shapes.box((1500.0, 1770.0, 1400.0), center=(750.0, 0.0, 850.0), cell=2000.0)


def run(label: str, obstacles: list[Mesh]) -> None:
    study = ParametricStudy(
        hitch_box, BOUNDS, start=START, flow=(-1.0, 0.0, 0.0), speed_kmh=110.0, obstacles=obstacles
    )
    # Cargo space that must stay whole: 70 cm deep, 1.4 m wide, 75 cm tall.
    study.keep(((FRONT_X - 850.0, -700.0, FLOOR_Z), (FRONT_X - 150.0, 700.0, FLOOR_Z + 750.0)))
    original = study.score()
    print(f"{label}: Cd {original.cd:.2f}, {original.force(110 / 3.6):.0f} N at 110 km/h, "
          f"{original.forebody / original.drag_area:.0%} of it from the front")
    started = time.perf_counter()
    sweep = study.sweep(
        remove_up_to=0.2,
        stations=5,
        callback=lambda n, removed, drag: print(
            f"  station {n}: {removed:5.1%} removed, drag {drag:.0%} of original", flush=True
        ),
    )
    print(f"  sweep: {time.perf_counter() - started:.0f}s\n")
    print(sweep, "\n")
    for station in sweep:
        io.save(station.mesh, OUT / f"hitch_box_{label}_{round(station.removed * 100):02d}.stl")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    run("free-air", [])
    run("behind-car", [CAR])
    print(f"written: {OUT / 'hitch_box_*.stl'}")


if __name__ == "__main__":
    main()
