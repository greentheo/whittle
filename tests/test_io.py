from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from whittle import io
from whittle.types import Mesh


def test_obj_round_trip(unit_cube: Mesh, tmp_path: Path) -> None:
    path = tmp_path / "cube.obj"
    io.save(unit_cube, path)
    loaded = io.load(path)
    assert np.allclose(loaded.vertices, unit_cube.vertices)
    assert np.array_equal(loaded.faces, unit_cube.faces)


def test_stl_round_trip_preserves_geometry(unit_cube: Mesh, tmp_path: Path) -> None:
    path = tmp_path / "cube.stl"
    io.save(unit_cube, path)
    loaded = io.load(path)
    # STL welding renumbers vertices, so compare geometry rather than indices.
    assert len(loaded.vertices) == 8
    assert len(loaded.faces) == 12
    assert abs(loaded.volume) == pytest.approx(1.0, abs=1e-5)
    assert loaded.area == pytest.approx(6.0, abs=1e-5)


def test_stl_is_written_binary(unit_cube: Mesh, tmp_path: Path) -> None:
    path = tmp_path / "cube.stl"
    io.save(unit_cube, path)
    raw = path.read_bytes()
    assert len(raw) == 84 + 12 * 50


def test_ascii_stl_is_read(tmp_path: Path) -> None:
    path = tmp_path / "tri.stl"
    path.write_text(
        "solid t\n"
        "facet normal 0 0 1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
        " endloop\nendfacet\nendsolid t\n"
    )
    mesh = io.load(path)
    assert len(mesh.faces) == 1
    assert mesh.area == pytest.approx(0.5)


def test_obj_fan_triangulates_quads(tmp_path: Path) -> None:
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    mesh = io.load(path)
    assert len(mesh.faces) == 2
    assert mesh.area == pytest.approx(1.0)


def test_obj_ignores_texture_and_normal_indices(tmp_path: Path) -> None:
    path = tmp_path / "t.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n")
    assert len(io.load(path).faces) == 1


def test_unknown_extension_is_rejected(unit_cube: Mesh, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unsupported mesh format"):
        io.save(unit_cube, tmp_path / "cube.ply")
    with pytest.raises(ValueError, match=r"unsupported mesh format"):
        io.load(tmp_path / "cube.ply")
