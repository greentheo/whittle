from __future__ import annotations

import numpy as np
import pytest

from whittle import TUBES, FrameStudy, tube_size
from whittle.physics.frame import section, tube_dimensions

INCH = 25.4
E = 200000.0  # mild steel, MPa
G = E / (2.0 * 1.29)


def _cantilever(length: float = 1000.0, key: str = "2x2x1/8") -> FrameStudy:
    frame = FrameStudy("steel")
    root, tip = frame.node((0.0, 0.0, 0.0), name="root"), frame.node((length, 0.0, 0.0), name="tip")
    frame.member(root, tip, key, group="arm")
    frame.support(root)
    return frame


def test_tube_properties_match_the_published_tables() -> None:
    # AISC's HSS tables, which use the same 0.93 design wall and 2t corner radius.
    for key, area, inertia, torsion in (
        ("2x2x1/8", 0.840, 0.486, 0.796),
        ("2x2x1/4", 1.51, 0.747, 1.31),
        ("3x3x3/16", 1.89, 2.46, 4.03),
    ):
        s = section(key)
        assert s.area / INCH**2 == pytest.approx(area, rel=0.01)
        assert s.iy / INCH**4 == pytest.approx(inertia, rel=0.01)
        assert s.iz == pytest.approx(s.iy)
        assert s.j / INCH**4 == pytest.approx(torsion, rel=0.01)
    solid = section("2x2xsolid")
    assert solid.solid and solid.iy == pytest.approx((2 * INCH) ** 4 / 12.0)


def test_tube_weighs_what_the_catalog_says() -> None:
    # The same tables' nominal weights, which are at the nominal wall.
    for key, pounds_per_foot in (("2x2x1/8", 3.05), ("2x2x1/4", 5.41), ("3x3x3/16", 6.87)):
        weight = section(key).mass_per_mm(7.85) * 12.0 * INCH / 0.45359
        assert weight == pytest.approx(pounds_per_foot, rel=0.01)
    solid = section("2x2xsolid")
    assert solid.mass_area == solid.area


def test_tube_keys_read_as_inches() -> None:
    assert tube_size("2x2x1/8") == pytest.approx((50.8, 3.175))
    assert tube_size("1.5x1.5x0.083") == pytest.approx((38.1, 2.1082))
    assert tube_size("2x2xsolid") == pytest.approx((50.8, 25.4))
    assert tube_dimensions("2x3x3/16") == pytest.approx((50.8, 76.2, 4.7625))
    assert "2x2x1/4" in TUBES
    with pytest.raises(ValueError, match="not square"):
        tube_size("2x3x1/8")
    with pytest.raises(ValueError, match="look like"):
        tube_size("2 inch tube")


def test_a_cantilever_matches_beam_theory() -> None:
    frame = _cantilever()
    frame.case("tip", self_weight=False)
    frame.load("tip", force=(0.0, 0.0, -500.0), case="tip")
    result = frame.analyze()
    case = result.cases["tip"]
    s = section("2x2x1/8")
    assert case.displacements[1, 2] == pytest.approx(-500.0 * 1000.0**3 / (3 * E * s.iy), rel=1e-6)
    assert case.stress[0] == pytest.approx(500.0 * 1000.0 / s.sy, rel=0.01)
    assert result.safety == pytest.approx(250.0 / case.stress[0])
    force, moment = case.resultant(about=(0.0, 0.0, 0.0))
    assert np.allclose(force, [0.0, 0.0, 500.0])
    assert np.allclose(moment, [0.0, -500.0 * 1000.0, 0.0])


def test_a_load_spread_along_a_member_matches_beam_theory() -> None:
    frame = _cantilever()
    frame.case("spread", self_weight=False)
    frame.spread(0, total=(0.0, 0.0, -1000.0), case="spread")
    case = frame.analyze().cases["spread"]
    s, w, length = section("2x2x1/8"), 1.0, 1000.0
    assert case.displacements[1, 2] == pytest.approx(-w * length**4 / (8 * E * s.iy), rel=1e-6)
    assert case.stress[0] == pytest.approx(w * length**2 / 2.0 / s.sy, rel=0.02)


def test_twisting_one_member_by_bending_another() -> None:
    # An L: a column along x fixed at its root, an arm along y off its end,
    # a load at the arm's tip -- which twists the column.
    frame = FrameStudy("steel")
    root, corner, tip = (frame.node(p) for p in ((0, 0, 0), (800, 0, 0), (800, 500, 0)))
    frame.member(root, corner, "2x2x1/8")
    frame.member(corner, tip, "2x2x1/8")
    frame.support(root)
    frame.case("tip", self_weight=False)
    frame.load(tip, force=(0, 0, -300), case="tip")
    case = frame.analyze().cases["tip"]
    s, p = section("2x2x1/8"), 300.0
    bending = p * 500**3 / (3 * E * s.iy) + p * 800**3 / (3 * E * s.iy)
    expected = bending + p * 500**2 * 800 / (G * s.j)
    assert -case.displacements[tip, 2] == pytest.approx(expected, rel=1e-6)


def test_the_frame_carries_its_own_weight_times_the_case_factor() -> None:
    frame = _cantilever()
    frame.case("bumps", factor=2.5)
    case = frame.analyze().cases["bumps"]
    weight = section("2x2x1/8").mass_per_mm(7.85) * 1000.0 * 9.81
    force, _ = case.resultant(about=(0.0, 0.0, 0.0))
    assert force[2] == pytest.approx(2.5 * weight)


def test_a_column_buckles_before_it_yields() -> None:
    frame = FrameStudy("steel")
    base, top = frame.node((0, 0, 0)), frame.node((0, 0, 3000))
    frame.member(base, top, "1x1x1/16")
    frame.support(base, fix="pin")
    frame.support(top, fix=("ux", "uy", "rz"))
    frame.case("press", self_weight=False)
    frame.load(top, force=(0, 0, -1000), case="press")
    case = frame.analyze().cases["press"]
    s = section("1x1x1/16")
    euler = np.pi**2 * E * s.iy / 3000.0**2
    assert case.compression[0] == pytest.approx(1000.0)
    assert case.safety[0] == pytest.approx(euler / 1000.0)
    assert case.safety[0] < 250.0 / case.stress[0]


def test_a_receiver_socket_carries_what_a_fixed_end_would() -> None:
    def frame(socket: bool) -> FrameStudy:
        f = FrameStudy("steel")
        inner, mouth = f.node((228.6, 0, 228.6)), f.node((76.2, 0, 228.6))
        tip = f.node((-1000.0, 0, 228.6))
        f.member(inner, mouth, "2x2x1/4")
        f.member(mouth, tip, "2x2x1/4")
        if socket:
            f.support(mouth, fix=("ux", "uy", "uz", "rx"))
            f.support(inner, fix=("uy", "uz"))
        else:
            f.support(mouth)
        f.case("driving", factor=2.5)
        f.load(tip, force=(0, 0, -445), case="driving")
        return f

    held, fixed = (frame(s).analyze().cases["driving"] for s in (True, False))
    mouth = (76.2, 0.0, 228.6)
    for a, b in zip(held.resultant(mouth), fixed.resultant(mouth), strict=True):
        assert np.allclose(a, b, atol=1e-6)
    # Same moment at the mouth, so the same stress in the shank there.
    assert held.stress[1] == pytest.approx(fixed.stress[1], rel=1e-6)


def test_a_sweep_lightens_what_is_overbuilt() -> None:
    frame = _cantilever(key="2x2x1/4")
    frame.case("tip")
    frame.load("tip", force=(0.0, 0.0, -100.0), case="tip")
    sweep = frame.sweep(groups={"arm": ["2x2x1/4", "2x2x3/16", "2x2x1/8", "2x2x0.083"]})
    masses = [s.metrics["mass_kg"] for s in sweep]
    assert masses == sorted(masses, reverse=True)
    assert sweep.original.design == {"arm": "2x2x1/4"}
    assert sweep[-1].design == {"arm": "2x2x0.083"}
    assert all(s.metrics["safety_factor"] >= 3.0 for s in sweep)
    assert "2x2x0.083" in sweep.table()


def test_a_sweep_strengthens_what_falls_short_and_says_so() -> None:
    frame = _cantilever(key="2x2x1/8")
    frame.case("tip", factor=2.5)
    frame.load("tip", force=(0.0, 0.0, -500.0), case="tip")
    sweep = frame.sweep(groups={"arm": ["2x2x1/8", "2x2x3/16", "2x2x1/4", "2x2xsolid"]})
    assert sweep.original.metrics["safety_factor"] < 3.0
    assert sweep[-1].metrics["safety_factor"] >= 3.0
    assert sweep[-1].metrics["mass_kg"] > sweep.original.metrics["mass_kg"]
    assert sweep[-1].removed < 0
    assert "on the road" in sweep.note


def test_a_sweep_that_cannot_get_there_says_so() -> None:
    frame = _cantilever(key="1x1x1/16")
    frame.case("tip")
    frame.load("tip", force=(0.0, 0.0, -5000.0), case="tip")
    sweep = frame.sweep(groups={"arm": ["1x1x1/16", "1x1x1/8"]})
    assert sweep[-1].design == {"arm": "1x1x1/8"}
    assert "nothing in the lists reaches" in sweep.note


def test_a_frame_from_plain_data() -> None:
    spec = {
        "material": "steel",
        "nodes": {"root": [0, 0, 0], "tip": [1000, 0, 0]},
        "members": [{"a": "root", "b": "tip", "section": "2x2x1/8", "group": "arm"}],
        "supports": [{"node": "root", "fix": "all"}],
        "cases": {"driving": {"factor": 2.5, "self_weight": False}},
        "loads": [
            {"case": "driving", "node": "tip", "force": [0, 0, -100]},
            {"case": "driving", "member": 0, "total": [0, 0, -200]},
        ],
    }
    result = FrameStudy.from_dict(spec).analyze()
    force, _ = result.cases["driving"].resultant(about=(0, 0, 0))
    assert force[2] == pytest.approx(2.5 * 300.0)
    assert result.notes == [] or "welds" in result.notes[0]


def test_mistakes_are_caught() -> None:
    frame = FrameStudy("steel")
    a, b = frame.node((0, 0, 0)), frame.node((1000, 0, 0))
    frame.member(a, b, "2x2x1/8")
    with pytest.raises(ValueError, match="support"):
        frame.analyze()
    with pytest.raises(ValueError, match="fix names"):
        frame.support(a, fix=("uz", "spin"))
    frame.support(a, fix="pin")  # free to spin about the member's axis
    with pytest.raises(ValueError, match="move without bending"):
        frame.analyze()
    with pytest.raises(ValueError, match="no node"):
        frame.member("nowhere", b, "2x2x1/8")


def test_the_frame_meshes_as_bars() -> None:
    frame = _cantilever()
    mesh = frame.mesh()
    s = section("2x2x1/8")
    assert mesh.volume == pytest.approx(s.width * s.height * 1000.0, rel=1e-9)
