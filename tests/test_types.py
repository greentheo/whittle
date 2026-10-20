from __future__ import annotations

import numpy as np
import pytest

from whittle.types import Mesh


def test_unit_cube_volume_and_area(unit_cube: Mesh) -> None:
    assert unit_cube.volume == pytest.approx(1.0)
    assert unit_cube.area == pytest.approx(6.0)


def test_volume_scales_cubically(unit_cube: Mesh) -> None:
    doubled = Mesh(unit_cube.vertices * 2.0, unit_cube.faces)
    assert doubled.volume == pytest.approx(8.0)


def test_bounds(unit_cube: Mesh) -> None:
    lo, hi = unit_cube.bounds
    assert np.allclose(lo, 0.0)
    assert np.allclose(hi, 1.0)


def test_empty_mesh_is_allowed() -> None:
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert empty.volume == 0.0
    assert empty.area == 0.0


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"vertices must be"):
        Mesh(np.zeros((4, 2)), np.zeros((0, 3), dtype=np.int64))
    with pytest.raises(ValueError, match=r"faces must be"):
        Mesh(np.zeros((4, 3)), np.zeros((2, 4), dtype=np.int64))


def test_rejects_out_of_range_face_index() -> None:
    with pytest.raises(ValueError, match=r"out of range"):
        Mesh(np.zeros((3, 3)), np.array([[0, 1, 9]], dtype=np.int64))
