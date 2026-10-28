"""Gusset plates for the two corners of a hitch platform's riser, ready for the plasma table.

Run:  python examples/08_hitch_gussets.py

The frame is example 07's, in the tubes its sweep ends on. Where the shank
jogs up into the riser and the riser turns back into the upper shank are the
two corners that carry the most bending, and the welds there are where a
frame shaken on the road cracks. A pair of cheek plates -- one each side of
the tubes -- at each corner takes most of that bending off the weld.

For each corner: what the plates take off the weld, and a run of plates from
whole to lightened, each cleaned up for a plasma table and checked again as
it will be cut. The lightest one that keeps 3.5x goes to out/ as DXF and SVG.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from whittle import TUBES

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
sys.path.insert(0, str(HERE))
hitch = importlib.import_module("07_hitch_frame")

#: Where example 07's sweep ends.
BUILT = {"shank": "2x2xsolid", "riser": "2x2xsolid", "upper_shank": "3x3x1/8",
         "rails": "1x1x1/16", "cross": "2x2x0.083"}  # fmt: skip
PLATE = 4.7625  # 3/16"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame = hitch.frame()
    frame.sections = [
        TUBES[BUILT[g]] if g in BUILT else s
        for g, s in zip(frame.groups, frame.sections, strict=True)
    ]
    started = time.perf_counter()
    # Member 2 is the riser; 1 the shank below it, 3 the upper shank above.
    for corner, other in (("jog_low", 1), ("jog_high", 3)):
        plate = frame.gusset(corner, 2, other, thickness=PLATE)
        sweep = plate.sweep(remove_up_to=0.5, stations=6)
        print(f"{corner}: 2 plates, 3/16 steel\n")
        print(sweep)
        keep = [s for s in sweep if s.metrics["safety_factor"] >= 3.5]
        chosen = keep[-1] if keep else sweep[0]
        print(f"\n{chosen.note}")
        for suffix in (".dxf", ".svg"):
            path = chosen.design.save(OUT / f"gusset_{corner}{suffix}")  # type: ignore[attr-defined]
        print(f"cut: {path.with_suffix('.dxf').name} and .svg, {chosen.metrics['mass_kg']:.2f} kg "
              f"each, {chosen.metrics['safety_factor']:.1f}x\n")  # fmt: skip
    print(f"{time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
