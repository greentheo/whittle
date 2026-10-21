from __future__ import annotations

import numpy as np
import pytest

from whittle import MATERIALS, Load, Support, VoxelFEA
from whittle.physics.fea import _unit_stiffness, material

E_PLA = MATERIALS["pla"].youngs_modulus
NU_PLA = MATERIALS["pla"].poisson_ratio


def _cantilever(
    length: int, width: int, height: int, *, spacing: float = 1.0, force: float = -10.0
) -> tuple[VoxelFEA, np.ndarray]:
    """A beam clamped over its first voxel layer and pulled down over its last."""
    shape = (length, width, height)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    fea = VoxelFEA(
        shape,
        spacing=spacing,
        material="pla",
        loads=[Load(tip, (0.0, 0.0, force))],
        supports=[Support(clamp)],
    )
    return fea, np.ones(shape, dtype=bool)


def _beam_theory(length: int, width: int, height: int, spacing: float, force: float) -> float:
    """Tip deflection of a Timoshenko cantilever, with the arm the voxel model actually has.

    The clamp holds both node planes of the first voxel and the load acts at
    the middle of the last, so the lever arm is a voxel and a half short.
    """
    arm = (length - 1.5) * spacing
    b, h = width * spacing, height * spacing
    inertia = b * h**3 / 12.0
    shear_modulus = E_PLA / (2.0 * (1.0 + NU_PLA))
    bending = abs(force) * arm**3 / (3.0 * E_PLA * inertia)
    shear = abs(force) * arm / (5.0 / 6.0 * shear_modulus * b * h)
    return bending + shear


def test_unit_stiffness_is_symmetric_with_six_rigid_body_modes() -> None:
    ke = _unit_stiffness(0.3)
    assert np.allclose(ke, ke.T)
    eigenvalues = np.linalg.eigvalsh(ke)
    # Three translations and three rotations cost no energy; nothing else is free.
    assert np.sum(np.abs(eigenvalues) < 1e-10) == 6
    assert eigenvalues.min() > -1e-10


@pytest.mark.parametrize(
    ("length", "width", "height", "spacing"), [(40, 4, 8, 1.0), (40, 4, 4, 1.0)]
)
def test_cantilever_deflection_matches_beam_theory(
    length: int, width: int, height: int, spacing: float
) -> None:
    fea, solid = _cantilever(length, width, height, spacing=spacing)
    result = fea.solve(solid)
    tip = -result.displacement[-1, :, :, 2].mean()
    expected = _beam_theory(length, width, height, spacing, -10.0)
    # Fully integrated hexes shear-lock a little in bending, so the model reads
    # slightly stiff; with four or more voxels through the depth, within 10%.
    assert 0.9 * expected < tip <= 1.02 * expected
    assert result.converged


def test_peak_stress_is_near_the_root_bending_stress() -> None:
    length, width, height = 40, 4, 6
    fea, solid = _cantilever(length, width, height)
    result = fea.solve(solid)
    inertia = width * height**3 / 12.0
    root = 10.0 * (length - 1.5) * (height / 2.0) / inertia
    # Corner sampling reaches the surface fibre; the margin kept clear of the
    # clamp's singularity costs a little. Reads 85-95% of beam theory.
    assert 0.8 * root < result.peak_stress < 1.05 * root


def test_deflection_scales_inversely_with_stiffness() -> None:
    shape = (20, 4, 4)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1] = True
    deflections = []
    for name in ("pla", "aluminum"):
        fea = VoxelFEA(
            shape, material=name, loads=[Load(tip, (0, 0, -5.0))], supports=[Support(clamp)]
        )
        deflections.append(fea.solve(np.ones(shape, dtype=bool)).max_displacement)
    ratio = deflections[0] / deflections[1]
    stiffness_ratio = MATERIALS["aluminum"].youngs_modulus / MATERIALS["pla"].youngs_modulus
    # Poisson's ratio differs a little between the two, so not exactly.
    assert ratio == pytest.approx(stiffness_ratio, rel=0.05)


def test_floating_material_is_dropped_and_changes_nothing() -> None:
    shape = (20, 4, 6)
    clamp = np.zeros(shape, dtype=bool)
    clamp[0, :, :4] = True
    tip = np.zeros(shape, dtype=bool)
    tip[-1, :, :4] = True
    fea = VoxelFEA(shape, loads=[Load(tip, (0, 0, -5.0))], supports=[Support(clamp)])
    beam = np.zeros(shape, dtype=bool)
    beam[:, :, :4] = True
    with_island = beam.copy()
    with_island[8:12, :, 5] = True  # a slab above the beam, touching nothing
    plain, island = fea.solve(beam), fea.solve(with_island)
    assert not island.solid[8:12, :, 5].any()
    assert island.compliance == pytest.approx(plain.compliance, rel=1e-6)


def test_a_load_with_no_path_to_a_support_is_unsupported() -> None:
    fea, solid = _cantilever(20, 4, 4)
    cut = solid.copy()
    cut[10] = False
    result = fea.solve(cut)
    assert not result.supported
    assert result.compliance == float("inf")
    assert result.max_displacement == float("inf")


def test_removing_material_never_makes_a_part_stiffer() -> None:
    fea, solid = _cantilever(30, 4, 6)
    trimmed = solid.copy()
    trimmed[5:25, 1:3, 2:4] = False
    assert fea.solve(trimmed).compliance > fea.solve(solid).compliance


def test_needs_loads_and_supports() -> None:
    region = np.ones((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match=r"at least one load"):
        VoxelFEA((4, 4, 4), loads=[], supports=[Support(region)])
    with pytest.raises(ValueError, match=r"at least one support"):
        VoxelFEA((4, 4, 4), loads=[Load(region, (0, 0, 1))], supports=[])


def test_rejects_empty_or_misshapen_regions() -> None:
    region = np.ones((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match=r"must have shape"):
        VoxelFEA((4, 4, 4), loads=[Load(np.ones((3, 3, 3), dtype=bool), (0, 0, 1))],
                 supports=[Support(region)])
    with pytest.raises(ValueError, match=r"empty"):
        VoxelFEA((4, 4, 4), loads=[Load(np.zeros((4, 4, 4), dtype=bool), (0, 0, 1))],
                 supports=[Support(region)])


def test_material_lookup() -> None:
    assert material("PLA") is MATERIALS["pla"]
    assert material(MATERIALS["steel"]) is MATERIALS["steel"]
    with pytest.raises(ValueError, match=r"unknown material"):
        material("unobtainium")
