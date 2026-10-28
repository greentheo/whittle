from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from whittle import Cut, FrameStudy, PlateStudy

INCH, LB = 25.4, 4.448
GUSSET = [(0, 0), (250, 0), (250, 60), (60, 250), (0, 250)]
RISER = [(0, 0), (60, 0), (60, 250), (0, 250)]
SHANK = [(60, 0), (250, 0), (250, 60), (60, 60)]


def test_a_load_is_spread_to_give_exactly_its_force_and_moment() -> None:
    plate = PlateStudy(GUSSET, thickness=5.0, resolution=50)
    plate.fix(RISER)
    plate.load(SHANK, force=(300.0, -2000.0), moment=-4e5, about=(30.0, 30.0))
    forces = plate._forces()[:, :, 0, :2] * plate.thickness / plate.spacing
    cells = np.argwhere(np.any(forces != 0, axis=-1))
    r = plate.corner + plate.spacing * cells
    f = forces[cells[:, 0], cells[:, 1]]
    assert np.allclose(f.sum(axis=0), [300.0, -2000.0])
    arm = r - np.array([30.0, 30.0])
    assert (arm[:, 0] * f[:, 1] - arm[:, 1] * f[:, 0]).sum() == pytest.approx(-4e5)


def test_a_plate_cantilever_is_stressed_as_beam_theory_says() -> None:
    # 300 x 60 x 5 mm, held over its first 30 mm, pulled down at its tip:
    # bending stress at the root is 6 P L / (t h^2).
    plate = PlateStudy([(0, 0), (300, 0), (300, 60), (0, 60)], thickness=5.0, resolution=100)
    plate.fix([(0, 0), (30, 0), (30, 60), (0, 60)])
    plate.load([(285, 0), (300, 0), (300, 60), (285, 60)], force=(0.0, -500.0))
    result = plate.analyze()
    beam = 6 * 500.0 * (292.5 - 30.0) / (5.0 * 60.0**2)
    assert result.peak_stress == pytest.approx(beam, rel=0.25)
    # Thicker plate, same loads: stress goes down as the thickness goes up.
    thicker = PlateStudy([(0, 0), (300, 0), (300, 60), (0, 60)], thickness=10.0, resolution=100)
    thicker.fix([(0, 0), (30, 0), (30, 60), (0, 60)])
    thicker.load([(285, 0), (300, 0), (300, 60), (285, 60)], force=(0.0, -500.0))
    assert thicker.analyze().peak_stress == pytest.approx(result.peak_stress / 2, rel=1e-6)


def test_a_sweep_makes_plates_a_plasma_table_can_cut(tmp_path: Path) -> None:
    plate = PlateStudy(GUSSET, thickness=4.7625, resolution=60)
    plate.fix(RISER)
    plate.load(SHANK, force=(0.0, -2000.0), moment=-4e5, about=(30.0, 30.0))
    sweep = plate.sweep(remove_up_to=0.4, stations=4)
    masses = [s.metrics["mass_kg"] for s in sweep]
    assert masses == sorted(masses, reverse=True)
    assert sweep[0].design.holes == []  # type: ignore[attr-defined]
    cut = sweep[-1].design
    assert isinstance(cut, Cut) and cut.holes
    for hole in cut.holes:
        assert np.ptp(hole, axis=0).min() >= 12.7 * 0.8  # no hole narrower than the plasma can cut
    assert cut.area * 4.7625 * 7.85e-6 == pytest.approx(sweep[-1].metrics["mass_kg"])
    dxf = cut.save(tmp_path / "gusset.dxf").read_text()
    assert "$INSUNITS\n70\n4" in dxf and dxf.count("POLYLINE") == 1 + len(cut.holes)
    svg = cut.save(tmp_path / "gusset.svg").read_text()
    assert 'fill-rule="evenodd"' in svg and 'width="250.000mm"' in svg
    with pytest.raises(ValueError, match=".dxf or .svg"):
        cut.save(tmp_path / "gusset.stl")


def _riser() -> FrameStudy:
    f = FrameStudy("steel")
    f.node((9 * INCH, 0, 9 * INCH), name="inner")
    f.node((3 * INCH, 0, 9 * INCH), name="mouth")
    f.node((-14 * INCH, 0, 9 * INCH), name="corner")
    f.node((-14 * INCH, 0, 25 * INCH), name="top")
    f.node((-52 * INCH, 0, 25 * INCH), name="tail")
    f.member("inner", "mouth", "2x2x5/16", group="shank")
    f.member("mouth", "corner", "2x2x5/16", group="shank")
    f.member("corner", "top", "2.5x2.5x1/8", group="riser")
    f.member("top", "tail", "2x2x1/8", group="spine")
    f.support("mouth", fix=("ux", "uy", "uz", "rx"))
    f.support("inner", fix=("uy", "uz"))
    f.case("driving", factor=2.5)
    f.spread(3, total=(0, 0, -100 * LB), case="driving")
    return f


def test_a_gusset_carries_what_the_member_pushes_into_the_joint() -> None:
    frame = _riser()
    plate = frame.gusset("corner", 2, 1, thickness=4.7625, reach=150, resolution=50)
    place = plate.placement
    assert place is not None
    u, v, n = (np.asarray(a) for a in (place.u, place.v, place.normal))
    assert np.allclose([u @ v, u @ n, v @ n], 0) and np.allclose(u, [0, 0, 1])
    # Everything the riser pushes into the corner, in the plates' plane, half on each plate.
    case = frame.analyze().cases["driving"]
    turn = frame._model().rotation[2]
    force, moment = -(turn.T @ case.end_forces[2, :3]), -(turn.T @ case.end_forces[2, 3:6])
    forces = plate._forces()[:, :, 0, :2] * plate.thickness / plate.spacing
    cells = np.argwhere(np.any(forces != 0, axis=-1))
    f = forces[cells[:, 0], cells[:, 1]]
    r = plate.corner + plate.spacing * cells
    assert np.allclose(f.sum(axis=0), [force @ u / 2, force @ v / 2])
    assert (r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]).sum() == pytest.approx(moment @ n / 2)
    # The tubes are 2.5" and 2" wide: flat cheek plates need a spacer.
    assert any("spacer" in note for note in plate.notes)
    assert plate.offset == pytest.approx((2.5 * INCH + 4.7625) / 2)


def test_a_gusset_says_how_much_it_takes_off_the_corner_weld() -> None:
    sweep = _riser().gusset("corner", 2, 1, thickness=4.7625, reach=150, resolution=40).sweep(
        remove_up_to=0.3, stations=3
    )
    whole, lightest = sweep[0], sweep[-1]
    assert 0.5 < lightest.metrics["plate_share"] <= whole.metrics["plate_share"] < 1.0
    assert lightest.metrics["weld_moment_nmm"] >= whole.metrics["weld_moment_nmm"]
    assert "riser-to-shank weld at corner" in whole.note and "lb in" in whole.note
    assert "welded over the tubes" in sweep.note  # why it stopped short of 30%
    assert lightest.mesh.volume > 0


def test_gusset_mistakes_are_caught() -> None:
    frame = _riser()
    with pytest.raises(ValueError, match="does not meet"):
        frame.gusset("corner", 3, 1, thickness=5.0)
    with pytest.raises(ValueError, match="in line"):
        frame.gusset("mouth", 0, 1, thickness=5.0)
    with pytest.raises(ValueError, match="no load case"):
        frame.gusset("corner", 2, 1, thickness=5.0, case="parked")
