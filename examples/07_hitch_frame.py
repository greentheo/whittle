"""Size the steel frame of a hitch-mounted cargo platform, driving over bumps.

Run:  python examples/07_hitch_frame.py

A shank in a 2" receiver, a riser where it jogs up 6", an upper shank running
back 50", and a ladder 60" wide on top of that: two rails and a cross member
at each end. Everything is square steel tube. The car drives towards +x, the
receiver's mouth is 3" behind the bumper line and 9" up, and the shank runs
6" inside it.

The receiver is a socket with a pin: the shank bears on its walls at the
mouth and at the inner end, the pin stops it sliding out, and the square
socket stops it turning. 100 lb of box and gear sits on the rails, and the
whole lot -- frame included -- is taken at 2.5x for hitting bumps.

What comes out: how close every member is to yielding, how much the far end
of the ladder dips, what the receiver has to hold -- the number to compare
with the hitch's rating -- and the lightest tube for each part of the frame
that still keeps 3.5x, the margin below which a welded frame shaken on the
road is in fatigue territory.
"""

from __future__ import annotations

import time

import numpy as np

from whittle import FrameStudy

INCH = 25.4
LB = 4.448  # newtons


def frame() -> FrameStudy:
    f = FrameStudy("steel")

    def node(x: float, y: float, z: float, name: str) -> int:
        return f.node((x * INCH, y * INCH, z * INCH), name=name)

    node(9, 0, 9, "inner")
    node(3, 0, 9, "mouth")
    node(-7, 0, 9, "jog_low")
    node(-7, 0, 15, "jog_high")
    for x, where in ((-18, "front"), (-57, "back")):
        node(x, 0, 15, f"{where}_shank")
        node(x, 0, 17, f"{where}_middle")
        node(x, -30, 17, f"{where}_left")
        node(x, 30, 17, f"{where}_right")

    tube = "2x2x1/8"
    f.member("inner", "mouth", tube, group="shank")
    f.member("mouth", "jog_low", tube, group="shank")
    f.member("jog_low", "jog_high", tube, group="riser")
    f.member("jog_high", "front_shank", tube, group="upper_shank")
    f.member("front_shank", "back_shank", tube, group="upper_shank")
    for where in ("front", "back"):
        # The cross member sits on the upper shank, welded where they cross.
        f.member(f"{where}_shank", f"{where}_middle", tube, group="upper_shank")
        f.member(f"{where}_left", f"{where}_middle", tube, group="cross")
        f.member(f"{where}_middle", f"{where}_right", tube, group="cross")
    left = f.member("front_left", "back_left", tube, group="rails")
    right = f.member("front_right", "back_right", tube, group="rails")

    f.support("mouth", fix=("ux", "uy", "uz", "rx"))
    f.support("inner", fix=("uy", "uz"))
    f.case("driving", factor=2.5)
    for rail in (left, right):
        f.spread(rail, total=(0.0, 0.0, -50.0 * LB), case="driving")
    return f


def main() -> None:
    study = frame()
    result = study.analyze()
    case = result.cases["driving"]
    groups = study.groups
    print(f"as drawn, every member 2x2x1/8: {result.mass:.1f} kg of steel\n")
    print("member                       group         stress MPa  safety")
    for i, (a, b) in enumerate(study.members):
        names = {v: k for k, v in study.names.items()}
        print(
            f"  {names[a]:>11s} - {names[b]:<11s}  {groups[i] or '':12s}  "
            f"{case.stress[i]:9.0f}  {case.safety[i]:6.1f}x"
        )
    force, moment = case.resultant(about=(3 * INCH, 0.0, 9 * INCH))
    down, twist = force[2] / LB, moment[1] / INCH / LB
    print(
        f"\nthe receiver holds {down:.0f} lb up and {twist:.0f} lb-in against tipping over bumps"
        f" -- {down / 2.5:.0f} lb and {twist / 2.5:.0f} lb-in standing still"
    )
    tip = int(np.argmax(case.deflection))
    print(f"the far end of the ladder dips {case.deflection[tip]:.1f} mm over bumps")
    for note in result.notes:
        print(f"note: {note}")

    started = time.perf_counter()
    walls = ["2x2x0.083", "2x2x1/8", "2x2x3/16", "2x2x1/4", "2x2x5/16", "2x2xsolid"]
    light = ["1x1x1/16", "1x1x0.083", "1x1x1/8", "1.25x1.25x0.083", "1.5x1.5x0.083",
             "1.5x1.5x1/8", "2x2x0.083", "2x2x1/8"]  # fmt: skip
    sweep = study.sweep(
        groups={
            "shank": walls,
            "riser": walls,
            "upper_shank": walls + ["2.5x2.5x1/8", "2.5x2.5x3/16", "3x3x1/8", "3x3x3/16"],
            "rails": light,
            "cross": light,
        },
        min_safety=3.5,
    )
    print(f"\nsweep: {len(sweep)} stations in {time.perf_counter() - started:.1f}s\n")
    print(sweep)
    print(f"\nwhere it ends: {sweep[-1].design}")


if __name__ == "__main__":
    main()
