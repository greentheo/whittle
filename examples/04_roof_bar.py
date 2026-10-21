"""Reshape a roof-rack bar's cross-section, and see what each step is worth.

Run:  python examples/04_roof_bar.py

A plain 40 x 25 mm rectangular bar, mounted across the car, with the air
coming from the front: ``flow=(-1, 0, 0)``. Only a short slice of it is
modelled -- 80 mm -- with ``extruded="y"``, which scores the slice as a piece
of an endless bar: its cut ends do not count, the wake behind it is the
strong two-dimensional kind a long bar really sheds, and the carving keeps
the cross-section the same all along.

Read the sequence, not just the end. The first thing the search does with a
small budget is round the leading edge, because that buys the most drag per
gram removed; only once that is spent does it start tapering the tail.

One thing to know before printing the extreme end: this model scores a sharp
nose and a rounded one about the same, because both keep the flow attached
head-on. In a crosswind a sharp nose stalls and a rounded one does not -- which
is why real aero bars have round leading edges. Round it off by hand.
"""

from __future__ import annotations

import time
from pathlib import Path

from whittle import FlowStudy, io, shapes

OUT = Path(__file__).resolve().parent.parent / "out"
SLICE_MM = 80.0


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bar = shapes.box((40.0, SLICE_MM, 25.0), cell=2.0)
    study = FlowStudy(bar, flow=(-1.0, 0.0, 0.0), extruded="y", speed_kmh=110.0)

    started = time.perf_counter()
    sweep = study.sweep(stations=[0.15, 0.3, 0.45, 0.55])
    print(f"sweep: {time.perf_counter() - started:.0f}s\n")
    print(sweep)

    per_metre = 1000.0 / SLICE_MM
    print("\ndrag per metre of bar at 110 km/h:")
    for station in sweep:
        newtons = station.metrics["force_n"] * per_metre
        print(f"  {station.removed:4.0%} removed: {newtons:5.1f} N")
        io.save(station.mesh, OUT / f"roof_bar_{round(station.removed * 100):02d}.stl")
    print(f"\nwritten: {OUT / 'roof_bar_*.stl'} (80 mm slices -- extrude the section in CAD)")


if __name__ == "__main__":
    main()
