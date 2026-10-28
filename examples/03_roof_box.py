"""A first-pass digital wind tunnel: carve a roof box until it slips through the air.

Run:  python examples/03_roof_box.py

Starts from the box anyone would build first -- 2 m long, 90 cm wide, 40 cm
tall, all square edges -- on a car driving towards +x, so the air comes at it
from the front: ``flow=(-1, 0, 0)``. Three constraints make it a roof box
rather than a sculpture:

* ``flat=("-z",)``  the bottom stays flat and where it is, on the bars;
* ``mirror="y"``    left and right stay the same;
* ``keep(...)``     a 1.5 m x 70 cm x 28 cm block of cargo space stays whole.

Each station is the lowest-drag shape found that removes at most that much of
the box. Read the drag column as a ranking and a trend, not a promise: the
model scores a box in free air, and a real one sits in the disturbed flow over
a car roof. What it gets right is *which* shapes are better and roughly how
quickly the savings run out -- usually long before the volume does.
"""

from __future__ import annotations

import time
from pathlib import Path

from whittle import FlowStudy, io, shapes

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    source = OUT / "roof_box_square.stl"
    io.save(shapes.box((2000.0, 900.0, 400.0), center=(0.0, 0.0, 200.0), cell=50.0), source)

    study = FlowStudy(
        io.load(source),
        flow=(-1.0, 0.0, 0.0),
        mirror="y",
        flat=("-z",),
        speed_kmh=110.0,
    )
    study.keep(((-750.0, -350.0, 0.0), (750.0, 350.0, 280.0)))
    original = study.analyze()
    how = "re-meshed from voxels" if study.remeshed else "as sent, evened out"
    print(f"part:     {source.name}, {len(study.base.faces)} faces, {how}")
    print(
        f"original: Cd {original.cd:.2f}, frontal {original.frontal_area * 1e-6:.3f} m2, "
        f"{original.force(110 / 3.6):.0f} N of drag at 110 km/h"
    )
    print(
        f"          {original.forebody / original.drag_area:.0%} of it from the front, "
        f"{original.base / original.drag_area:.0%} from the wake, "
        f"{original.friction / original.drag_area:.0%} friction\n"
    )

    started = time.perf_counter()
    sweep = study.sweep(
        remove_up_to=0.25,
        stations=6,
        callback=lambda n, removed, drag: print(
            f"  station {n}: {removed:5.1%} removed, drag {drag:.0%} of original", flush=True
        ),
    )
    print(f"sweep:    {time.perf_counter() - started:.0f}s\n")
    print(sweep)

    for station in sweep:
        path = OUT / f"roof_box_{round(station.removed * 100):02d}.stl"
        io.save(station.mesh, path)
    print(f"\nwritten:  {OUT / 'roof_box_*.stl'}")

    # Where is the 80/20 point? The first station that gets 80% of the way
    # to the best drag reduction on offer.
    best = sweep[-1].metrics["drag"]
    enough = next(s for s in sweep if 1.0 - s.metrics["drag"] >= 0.8 * (1.0 - best))
    print(
        f"80% of the available drag saving by {enough.removed:.0%} removed "
        f"(drag {enough.metrics['drag']:.0%} of the square box's)"
    )


if __name__ == "__main__":
    main()
