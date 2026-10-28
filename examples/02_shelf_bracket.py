"""Lighten a shelf bracket: screw it to a wall, put a load on it, slide.

Run:  python examples/02_shelf_bracket.py

Starts from a solid triangular bracket -- the kind anyone would model first --
saved as an STL and loaded back the way your own part would be. It is 8 mm
thick, printed in PLA, screwed flat to the wall, and carries 25 kg of shelf
at its tip. The sweep whittles it down in steps and scores every step, so the
question "how light can I go?" becomes a table to read rather than a number to
trust.

Watch for the classic answer appearing on its own: a diagonal strut under
compression and a tie along the top in tension. Nothing in the code knows what
a bracket should look like; the FEA solve finds where the material works.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from whittle import Grid, LoadStudy, io

OUT = Path(__file__).resolve().parent.parent / "out"


def solid_bracket(path: Path) -> None:
    """Write an 80 x 80 mm triangular gusset bracket, 8 mm thick, as an STL."""
    grid = Grid((80, 8, 80), 1.0, (0.5, 0.5, 0.5))
    c = grid.centers()
    # Wall along x = 0, shelf along z = 80, hypotenuse between them, with an
    # 8 mm band left along the wall and the top so the corners are not points.
    solid = c[..., 2] >= c[..., 0] - 8.0
    io.save(grid.mesh(solid, smooth=0.0), path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    source = OUT / "bracket_solid.stl"
    solid_bracket(source)

    part = io.load(source)
    study = LoadStudy(part, material="pla", resolution=48)
    print(f"part:     {source.name}, {study.solid.sum()} voxels of {study.grid.spacing:.1f} mm")

    # Screwed flat to the wall: hold the 4 mm of material against it.
    study.support(box=((0, 0, 0), (4, 8, 80)))
    # 25 kg of shelf, resting on the last 10 mm of the top.
    study.load((0.0, 0.0, -245.0), box=((70, 0, 76), (80, 8, 80)))

    started = time.perf_counter()
    sweep = study.sweep(remove_up_to=0.8, stations=9)
    print(f"sweep:    {time.perf_counter() - started:.1f}s\n")
    print(sweep)

    for target in (0.0, 0.4, 0.6, 0.8):
        station = sweep.at(target)
        path = OUT / f"bracket_{round(station.removed * 100):02d}.stl"
        io.save(station.mesh, path)
        grams, safety = station.metrics["mass_g"], station.metrics["safety_factor"]
        print(f"\nwritten:  {path}  ({grams:.0f} g, {safety:.1f}x)", end="")
    print()

    safe = [s for s in sweep if s.metrics["safety_factor"] >= 3.0][-1]
    grams, original = safe.metrics["mass_g"], sweep.original.metrics["mass_g"]
    print(
        f"\nlightest design with a 3x safety factor: {safe.removed:.0%} removed, "
        f"{grams:.0f} g instead of {original:.0f} g"
    )
    stiffness = np.array([s.metrics["stiffness"] for s in sweep])
    softer = sweep[int(np.argmax(stiffness < 0.9))]
    print(f"stiffness first drops below 90% at {softer.removed:.0%} removed")


if __name__ == "__main__":
    main()
