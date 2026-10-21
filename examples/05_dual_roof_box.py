"""Shape a foam roof pod for the wind and lighten it for its load, together.

Run:  python examples/05_dual_roof_box.py

A DIY roof pod: a 2 m block of XPS foam board on two roof bars, carrying
50 kg of cargo in a space that has to stay empty. Driving at 110 km/h it
feels its own drag; in a 40 km/h crosswind it feels that and a push from the
side too, and both reach the bars through the foam.

The study carves the outside for drag -- a few shapes, coarse first, then
filling in where neighbouring shapes differ most -- and inside each shape
lightens the foam for those loads, keeping the outer skin so the shape and its
drag survive. The result is a map: down it, slipperier; along it, lighter.

Why foam: a voxel here is about 40 mm, so walls come out 40 mm or thicker.
That is a real foam wall -- and nonsense for a 4 mm moulded plastic shell,
whose weight this could not tell you. The foam is bare, without the glass or
carbon skin usually laid over it, so a design that is safe here has strength
to spare once skinned.
"""

from __future__ import annotations

import time
from pathlib import Path

from whittle import DualStudy, io, shapes

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pod = shapes.box((2000.0, 900.0, 400.0), center=(0.0, 0.0, 200.0), cell=50.0)
    study = DualStudy(
        pod,
        flow=(-1.0, 0.0, 0.0),
        material="xps",
        speed_kmh=110.0,
        crosswind_kmh=40.0,
        mirror="y",
        flat=("-z",),
    )
    # Two roof bars under the floor, 800 mm apart.
    study.support(box=((350.0, -460.0, -10.0), (450.0, 460.0, 40.0)))
    study.support(box=((-450.0, -460.0, -10.0), (-350.0, 460.0, 40.0)))
    # Cargo space, with 50 kg resting on the floor beneath it.
    study.cargo(((-750.0, -350.0, 60.0), (750.0, 350.0, 320.0)), kg=50.0)

    started = time.perf_counter()
    sweep = study.sweep(remove_up_to=0.25, aero_stations=5, lighten_up_to=0.6, weight_stations=5)
    print(f"map: {len(sweep.designs())} designs in {time.perf_counter() - started:.0f}s\n")
    print(sweep)

    # Carving the nose removes foam too, so here slipperier and lighter pull
    # the same way: the trade-off is against strength. How conservative to be
    # is the real choice, so show the best designs at a few safety margins.
    saved = set()
    for margin in (5.0, 20.0, 40.0):
        print(f"\nat least {margin:g}x safe -- nothing else is both slipperier and lighter:")
        for i, j in sweep.front(min_safety=margin):
            row, station = sweep.rows[i], sweep.rows[i].stations[j]
            m = station.metrics
            print(
                f"  carved {row.removed:4.0%}, lightened {station.removed:4.0%}: "
                f"{m['mass_g'] / 1000:5.1f} kg, drag {m['drag']:4.0%} ({m['force_n']:5.1f} N), "
                f"safety {m['safety_factor']:5.1f}x, lid sags {m['deflection_mm']:.1f} mm"
            )
            if (i, j) not in saved:
                saved.add((i, j))
                carved, light = round(row.removed * 100), round(station.removed * 100)
                io.save(station.mesh, OUT / f"roof_pod_carved{carved:02d}_light{light:02d}.stl")
    print(f"\nwritten: {OUT / 'roof_pod_*.stl'}")

if __name__ == "__main__":
    main()
