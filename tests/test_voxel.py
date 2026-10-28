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


def test_coverage_puts_an_unaligned_surface_back_where_it_was() -> None:
    # 41" is not a whole number of 47 mm voxels: meshed from which centres are
    # inside, the box comes back about half a voxel proud on that axis.
    box = shapes.box((990.6, 1879.6, 1041.4), center=(0.0, 0.0, 977.9), cell=2000.0)
    grid = Grid.around(box, resolution=40, pad=2)
    filled = grid.coverage(box)
    assert filled.min() >= 0.0 and filled.max() <= 1.0
    assert filled.sum() * grid.spacing**3 == pytest.approx(box.volume, rel=0.02)
    lo, hi = box.bounds
    from_centres = grid.mesh(grid.voxelize(box))
    from_coverage = grid.mesh(filled, smooth=0.0)
    assert from_centres.bounds[0][2] < lo[2] - 0.4 * grid.spacing
    assert from_coverage.bounds[0][2] == pytest.approx(lo[2], abs=0.15 * grid.spacing)
    assert from_coverage.bounds[1][2] == pytest.approx(hi[2], abs=0.15 * grid.spacing)


def test_coverage_is_exact_along_the_rays() -> None:
    # A plate a third of a voxel thick: voxel centres miss it entirely or take
    # a whole layer, but where the rays cross it is known exactly.
    plate = shapes.box((100.0, 100.0, 3.0), center=(0.0, 0.0, 1.5), cell=50.0)
    grid = Grid((12, 12, 3), 10.0, (-55.0, -55.0, -8.0))
    filled = grid.coverage(plate)
    assert filled.sum() * grid.voxel_volume == pytest.approx(plate.volume, rel=1e-6)
    assert grid.voxelize(plate).sum() * grid.voxel_volume != pytest.approx(plate.volume, rel=0.5)


def test_a_design_meshed_by_its_share_keeps_the_part_outside() -> None:
    box = shapes.box((100.0, 13.0, 7.0), center=(50.0, 6.5, 3.5), cell=1.0)
    grid = Grid.around(box, resolution=24)
    solid, filled = grid.voxelize(box), grid.coverage(box)
    assert grid.share(solid, solid, filled).sum() * grid.voxel_volume == pytest.approx(
        box.volume, rel=0.05
    )
    lo, hi = box.bounds
    mesh = grid.mesh(grid.fitted(solid, solid, filled), smooth=0.0)
    # Two voxels thick, which is as thin as this gets right: within a fifth.
    assert np.all(np.abs(mesh.bounds[0] - lo) < 0.2 * grid.spacing)
    assert np.all(np.abs(mesh.bounds[1] - hi) < 0.2 * grid.spacing)
    # Taking the middle away opens a hole through it, and the share drops to match.
    design = solid & ~grid.box((40.0, -10.0, -10.0), (60.0, 30.0, 30.0))
    assert grid.share(design, solid, filled).sum() < grid.share(solid, solid, filled).sum()
    halves = grid.mesh(grid.fitted(design, solid, filled), smooth=0.0)
    assert halves.volume < 0.9 * mesh.volume


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
