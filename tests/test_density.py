from __future__ import annotations

import numpy as np
import pytest

from whittle import DensityField


def test_param_count_and_bounds() -> None:
    backend = DensityField((8, 8, 8), (4, 4, 4))
    assert backend.n_params == 64
    assert backend.bounds == [(0.0, 1.0)] * 64
    assert backend.changes_topology is True


def test_field_has_evaluation_shape() -> None:
    backend = DensityField((9, 7, 5), (3, 3, 3))
    assert backend.field(np.full(27, 0.5)).shape == (9, 7, 5)


def test_field_is_clipped_to_unit_range() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    # Cubic upsampling overshoots on a sharp step; clipping is what keeps the
    # field a density rather than an arbitrary scalar.
    field = backend.field(np.array([0.0, 1.0] * 4))
    assert field.min() >= 0.0
    assert field.max() <= 1.0


def test_keep_and_void_masks_are_enforced() -> None:
    shape = (8, 8, 8)
    keep = np.zeros(shape, dtype=bool)
    keep[0:2, 0:2, 0:2] = True
    void = np.zeros(shape, dtype=bool)
    void[6:8, 6:8, 6:8] = True
    backend = DensityField(shape, (4, 4, 4), keep=keep, void=void)
    # Even the genome that would erode everything cannot touch the keep region.
    field = backend.field(np.zeros(64))
    assert np.all(field[keep] == 1.0)
    field = backend.field(np.ones(64))
    assert np.all(field[void] == 0.0)


def test_overlapping_keep_and_void_is_rejected() -> None:
    mask = np.ones((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match=r"keep and void overlap"):
        DensityField((4, 4, 4), (2, 2, 2), keep=mask, void=mask)


def test_rejects_control_grid_larger_than_evaluation_grid() -> None:
    with pytest.raises(ValueError, match=r"must not exceed"):
        DensityField((4, 4, 4), (8, 8, 8))


def test_rejects_mask_of_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"must have shape"):
        DensityField((4, 4, 4), (2, 2, 2), keep=np.ones((3, 3, 3), dtype=bool))


def test_full_field_meshes_to_the_design_box() -> None:
    backend = DensityField((10, 10, 10), (3, 3, 3), spacing=0.5)
    mesh = backend.build(np.ones(27))
    # Padded and meshed at the threshold, so the hull sits half a voxel outside
    # the solid; what matters is that it is closed and of the right order.
    assert len(mesh.faces) > 0
    assert mesh.volume > 0
    lo, hi = mesh.bounds
    assert np.all(hi - lo > 4.0)


def test_empty_field_meshes_to_nothing() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    mesh = backend.build(np.zeros(8))
    assert len(mesh.faces) == 0
    assert mesh.volume == 0.0


def test_solid_matches_threshold() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2), threshold=0.5)
    genome = np.full(8, 0.9)
    assert np.array_equal(backend.solid(genome), backend.field(genome) >= 0.5)


def test_wrong_genome_length_is_rejected() -> None:
    backend = DensityField((8, 8, 8), (2, 2, 2))
    with pytest.raises(ValueError, match=r"expected genome of 8 values"):
        backend.field(np.zeros(7))
