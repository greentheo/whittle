from __future__ import annotations

import numpy as np
import pytest

from whittle import Grid, shapes
from whittle.types import Mesh
from whittle.voxel import taubin


def test_voxelizing_an_aligned_box_is_exact() -> None:
    box = shapes.box((10.0, 6.0, 4.0), center=(5.0, 3.0, 2.0), cell=1.0)
    grid = Grid.around(box, spacing=1.0)
    assert grid.shape == (10, 6, 4)
    solid = grid.voxelize(box)
    assert solid.all()


def test_voxelized_sphere_has_the_right_volume() -> None:
    sphere = shapes.sphere(10.0, subdivisions=4)
    grid = Grid.around(sphere, spacing=0.5, pad=1)
    solid = grid.voxelize(sphere)
    volume = solid.sum() * grid.voxel_volume
    # The icosphere is inscribed, so a shade under the true sphere.
    assert volume == pytest.approx(4.0 / 3.0 * np.pi * 10.0**3, rel=0.03)
    # And nothing leaked outside it, which a parity error would do a column at a time.
    assert not solid[0].any() and not solid[-1].any()


def test_unit_cube_fixture_voxelizes_despite_its_large_triangles(unit_cube: Mesh) -> None:
    grid = Grid.around(unit_cube, spacing=0.1)
    assert grid.voxelize(unit_cube).sum() == 1000


def test_grid_is_centred_on_the_part() -> None:
    sphere = shapes.sphere(5.0, center=(100.0, -20.0, 3.0))
    grid = Grid.around(sphere, resolution=10)
    middle = np.asarray(grid.origin) + grid.spacing * (np.asarray(grid.shape) - 1) / 2.0
    assert np.allclose(middle, (100.0, -20.0, 3.0), atol=1e-9)


def test_box_and_ball_select_by_voxel_centre() -> None:
    grid = Grid((10, 10, 10), 1.0)
    assert grid.box((0, 0, 0), (2, 2, 2)).sum() == 27
    # Corners in either order.
    assert grid.box((2, 2, 2), (0, 0, 0)).sum() == 27
    assert grid.ball((5, 5, 5), 1.0).sum() == 7


def test_near_never_comes_back_empty() -> None:
    grid = Grid((10, 10, 10), 1.0)
    within = np.zeros(grid.shape, dtype=bool)
    within[0, 0, 0] = True
    # A click far from the only material still selects it.
    region = grid.near((6.0, 6.0, 6.0), within)
    assert region[0, 0, 0] and region.sum() == 1
    with pytest.raises(ValueError, match=r"no material"):
        grid.near((0, 0, 0), np.zeros(grid.shape, dtype=bool))


def test_meshing_a_voxelization_round_trips_the_volume() -> None:
    sphere = shapes.sphere(10.0, subdivisions=4)
    grid = Grid.around(sphere, spacing=0.5, pad=2)
    mesh = grid.mesh(grid.voxelize(sphere))
    assert mesh.volume == pytest.approx(sphere.volume, rel=0.03)
    lo, hi = mesh.bounds
    assert np.allclose((lo + hi) / 2.0, 0.0, atol=0.3)


def test_taubin_smoothing_keeps_the_volume_and_flattens_the_terraces() -> None:
    sphere = shapes.sphere(10.0, subdivisions=4)
    grid = Grid.around(sphere, spacing=1.0, pad=2)
    raw = grid.mesh(grid.voxelize(sphere))
    relaxed = taubin(raw, 15)
    assert relaxed.volume == pytest.approx(raw.volume, rel=0.02)
    # On a sphere every vertex should sit at the same radius; terracing scatters them.
    spread_raw = np.std(np.linalg.norm(raw.vertices, axis=1))
    spread_relaxed = np.std(np.linalg.norm(relaxed.vertices, axis=1))
    assert spread_relaxed < 0.6 * spread_raw


def test_rejects_bad_grids() -> None:
    with pytest.raises(ValueError, match=r"positive"):
        Grid((0, 4, 4), 1.0)
    with pytest.raises(ValueError, match=r"spacing"):
        Grid((4, 4, 4), 0.0)
    with pytest.raises(ValueError, match=r"empty mesh"):
        Grid.around(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))


def test_a_click_on_a_face_selects_that_face_and_only_it() -> None:
    grid = Grid((20, 10, 10), 1.0)
    solid = np.zeros(grid.shape, dtype=bool)
    solid[:8] = True  # one block
    solid[12:] = True  # another, its +x face coplanar with nothing but itself
    face = grid.patch((-0.5, 5.0, 5.0), (-1.0, 0.0, 0.0), solid)
    # Two voxel layers of the clicked block's -x face, all of it.
    assert face[:2].all() and not face[2:].any()
    # A coplanar face on a different block is not part of the click.
    top = grid.patch((2.0, 5.0, 9.5), (0.0, 0.0, 1.0), solid)
    assert top[:8, :, 8:].all() and not top[12:].any()
    with pytest.raises(ValueError, match=r"non-zero"):
        grid.patch((0, 0, 0), (0, 0, 0), solid)
