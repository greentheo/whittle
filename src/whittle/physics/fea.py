"""Linear-elastic finite elements on the voxel grid.

The simplest solver that answers the question this library actually asks:
*roughly how much stiffness does this design keep, and is it anywhere near
breaking?* It is not the most accurate solver one could write, and is not
trying to be.

* **One element per voxel**, an 8-node trilinear hexahedron, so the geometry
  backend and the solver share a grid and a problem is posed in exactly one
  coordinate system. Fully integrated trilinear hexes are a little too stiff in
  bending -- they shear-lock -- so deflections come out low by roughly 5-20%,
  depending on how many voxels span the thickness. Ranking designs against each
  other is unaffected, which is what the search needs.
* **Hard-kill material.** A voxel is either the part's material or nothing.
  Material that is not connected to a support carries no load and is dropped
  before the solve; it would only make the system singular.
* **Matrix-free conjugate gradients**, element by element, Jacobi-preconditioned.
  No global stiffness matrix is ever assembled. A direct sparse factorization
  of the same system is exact, and on a 60x30x20 grid it takes minutes where
  this takes seconds -- the fill-in of a 3D factorization is the whole cost.
  Jacobi is the weak point: a slender part needs several hundred iterations.
  Two cheaper fixes were measured and do not work -- starting from the
  previous design's displacement saves nothing, because removing even 2% of
  the voxels leaves a residual as large as starting from zero, and an
  incomplete-LU preconditioner makes CG diverge. The fix that would work is
  geometric multigrid (ROADMAP.md).

Units are whatever the inputs are, consistently. With millimetres, newtons
and moduli in MPa, displacements come out in millimetres and stresses in MPa,
which is what :data:`MATERIALS` assumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg

from ..types import BoolArray, FloatArray, IntArray, Vector3


@dataclass(frozen=True)
class Material:
    """Isotropic and linear-elastic, plus one strength to compare stress against.

    ``strength`` is the stress at which to expect the part to fail -- yield for
    metals, ultimate tensile strength for printed plastics, which tend to
    break before they visibly yield.
    """

    name: str
    #: Young's modulus, MPa (N/mm^2).
    youngs_modulus: float
    poisson_ratio: float
    #: Stress at failure, MPa.
    strength: float
    #: g/cm^3, for reporting mass.
    density: float

    def __post_init__(self) -> None:
        if self.youngs_modulus <= 0:
            raise ValueError("youngs_modulus must be positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")


#: Nominal properties for the materials a hobby part is likely to be made of.
#:
#: Printed plastics are taken at 100% infill, loaded in the plane of the
#: layers. Across the layers a print is often half as strong, and at partial
#: infill both stiffness and strength drop roughly in proportion to the infill
#: -- so read a safety factor below about 3 on a printed part as "test it",
#: not as "fine". Plywood is not isotropic and is modelled as if it were.
#: XPS is extruded-polystyrene insulation board, the core of a carved-foam
#: build: bare foam, without the glass or carbon skin that usually goes over
#: it, so a design that survives in it has strength to spare once skinned.
MATERIALS: dict[str, Material] = {
    "pla": Material("PLA (printed)", 3500.0, 0.36, 40.0, 1.24),
    "petg": Material("PETG (printed)", 2100.0, 0.38, 35.0, 1.27),
    "abs": Material("ABS (printed)", 2000.0, 0.35, 28.0, 1.04),
    "nylon": Material("Nylon (printed)", 1500.0, 0.39, 40.0, 1.14),
    "aluminum": Material("Aluminium 6061-T6", 69000.0, 0.33, 275.0, 2.70),
    "steel": Material("Mild steel", 200000.0, 0.29, 250.0, 7.85),
    "plywood": Material("Birch plywood", 8000.0, 0.30, 30.0, 0.68),
    "xps": Material("XPS foam board", 15.0, 0.20, 0.25, 0.035),
}


def material(spec: str | Material) -> Material:
    """Look a material up by name, or pass one straight through."""
    if isinstance(spec, Material):
        return spec
    try:
        return MATERIALS[spec.lower()]
    except KeyError:
        known = ", ".join(sorted(MATERIALS))
        raise ValueError(f"unknown material {spec!r}; known: {known}") from None


@dataclass(frozen=True)
class Load:
    """A force applied to a region of the part.

    ``force`` is the *total*, in newtons, spread evenly over the nodes of the
    region's voxels. Ten kilograms hanging from a hook is ``(0, 0, -98.1)``.

    ``case`` groups loads into load cases. Loads in the same case act at the
    same time; different cases are separate situations the part must survive
    one at a time -- a roof box driving straight into the wind, and the same
    box in a crosswind.
    """

    region: BoolArray
    force: Vector3
    case: int = 0


@dataclass(frozen=True)
class FieldLoad:
    """A force on every voxel that carries one: a pressure spread over a surface.

    ``forces`` has the grid's shape plus a trailing 3, in newtons per voxel.
    This is how wind gets onto a part: the drag model's pressure on each face
    of the surface, handed to the voxel under that face
    (:func:`whittle.dual.wind_forces`). A distributed load does not
    concentrate stress where it is applied, so unlike a :class:`Load` its
    voxels are not kept out of :meth:`VoxelFEA.peak_stress`.
    """

    forces: FloatArray
    case: int = 0

    @property
    def region(self) -> BoolArray:
        mask: BoolArray = np.any(self.forces != 0.0, axis=-1)
        return mask


@dataclass(frozen=True)
class Support:
    """A region of the part held in place -- bolted, clamped, or sitting on something.

    ``fixed`` picks which axes are held. All three is a bolted or glued face;
    only the axis normal to a surface is a part resting on it.
    """

    region: BoolArray
    fixed: tuple[bool, bool, bool] = (True, True, True)


@dataclass(frozen=True)
class FEAResult:
    """One static solve. Everything a design decision needs, and nothing more."""

    #: Whether every load is connected by material to a support. When it is
    #: not, there is no static answer and everything below is a placeholder.
    supported: bool
    #: Work done by the loads, ``F . u`` -- N mm, which is mJ. Lower is stiffer.
    compliance: float
    #: Nodal displacement, shape ``(nx + 1, ny + 1, nz + 1, 3)``. Of the first
    #: load case, when there are several.
    displacement: FloatArray
    #: Strain energy per voxel, N mm, summed over load cases. Where the part is working.
    strain_energy: FloatArray
    #: von Mises stress per voxel, the worst of its eight corners and of every
    #: load case, MPa.
    von_mises: FloatArray
    #: The material actually analyzed: the input minus anything floating free.
    solid: BoolArray
    #: A robust maximum stress -- see :meth:`VoxelFEA.peak_stress`.
    peak_stress: float
    converged: bool
    iterations: int
    #: The largest displacement of any node in any load case, mm.
    max_displacement: float = float("inf")
    #: Compliance of each load case on its own; ``compliance`` is their sum.
    case_compliance: tuple[float, ...] = ()


# Corner order of a hex element: x varies fastest, then y, then z.
_CORNERS = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=np.int64)


def _elasticity(nu: float) -> FloatArray:
    """Isotropic 6x6 constitutive matrix at unit modulus. Voigt order xx yy zz xy yz zx."""
    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = 1.0 / (2.0 * (1.0 + nu))
    d = np.zeros((6, 6), dtype=np.float64)
    d[:3, :3] = lam
    d[[0, 1, 2], [0, 1, 2]] += 2.0 * mu
    d[[3, 4, 5], [3, 4, 5]] = mu
    return d


def _strain_displacement(xi: FloatArray) -> FloatArray:
    """The 6x24 B matrix of the unit hex at local coordinates ``xi`` in ``[0, 1]^3``.

    Shear rows are engineering strains, which is what pairs with the shear
    modulus on the diagonal of :func:`_elasticity`.
    """
    b = np.zeros((6, 24), dtype=np.float64)
    for a, corner in enumerate(_CORNERS):
        # Trilinear shape function of this corner, and its derivative per axis.
        f = np.where(corner == 1, xi, 1.0 - xi)
        sign = np.where(corner == 1, 1.0, -1.0)
        dx = sign[0] * f[1] * f[2]
        dy = f[0] * sign[1] * f[2]
        dz = f[0] * f[1] * sign[2]
        col = 3 * a
        b[0, col] = dx
        b[1, col + 1] = dy
        b[2, col + 2] = dz
        b[3, col], b[3, col + 1] = dy, dx
        b[4, col + 1], b[4, col + 2] = dz, dy
        b[5, col], b[5, col + 2] = dz, dx
    return b


@lru_cache(maxsize=8)
def _unit_stiffness(nu: float) -> FloatArray:
    """Stiffness of the unit hex at unit modulus, by 2x2x2 Gauss quadrature."""
    d = _elasticity(nu)
    gauss = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))
    ke = np.zeros((24, 24), dtype=np.float64)
    for x in gauss:
        for y in gauss:
            for z in gauss:
                b = _strain_displacement(np.array([x, y, z]))
                ke += b.T @ d @ b / 8.0
    ke.setflags(write=False)
    return ke


class VoxelFEA:
    """A static linear-elastic solve on a voxel grid, set up once and run many times.

    Everything that does not depend on which voxels are solid -- element
    connectivity, the load vector, which degrees of freedom are held -- is
    built here, so :meth:`solve` pays only for the solve.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        *,
        spacing: float = 1.0,
        material: str | Material = "pla",
        loads: Sequence[Load | FieldLoad],
        supports: Sequence[Support],
        stiffness: FloatArray | None = None,
    ) -> None:
        if not loads:
            raise ValueError("need at least one load")
        if not supports:
            raise ValueError("need at least one support -- an unsupported part just flies away")
        self.shape = shape
        self.spacing = float(spacing)
        self.material = _resolve(material)
        #: A part of two materials: each voxel's stiffness as a multiple of ``material``'s (a plywood top on
        #: printed sides). ``None`` is one material throughout. Stress follows each voxel's own stiffness.
        self.stiffness = self._check_stiffness(stiffness)
        self.loads = list(loads)
        self.supports = list(supports)
        for load in self.loads:
            if isinstance(load, FieldLoad) and load.forces.shape != (*shape, 3):
                raise ValueError(f"forces must have shape {(*shape, 3)}, got {load.forces.shape}")
        regions: list[BoolArray] = [item.region for item in self.loads]
        regions += [item.region for item in self.supports]
        for region in regions:
            if region.shape != shape:
                raise ValueError(f"region must have shape {shape}, got {region.shape}")
            if not region.any():
                raise ValueError("a load or support region is empty")

        nx, ny, nz = shape
        node_id = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
        ex, ey, ez = (axis.ravel() for axis in np.indices(shape))
        nodes = node_id[
            ex[:, None] + _CORNERS[:, 0], ey[:, None] + _CORNERS[:, 1], ez[:, None] + _CORNERS[:, 2]
        ]
        self._nodes: IntArray = nodes
        self._edof: IntArray = (3 * nodes[:, :, None] + np.arange(3)).reshape(-1, 24)
        self.n_dof = 3 * int(node_id.size)

        e0 = self.material.youngs_modulus
        nu = self.material.poisson_ratio
        # A hex of side h is h times stiffer than the unit hex: B scales as 1/h
        # and the volume as h^3.
        self._ke: FloatArray = e0 * self.spacing * np.asarray(_unit_stiffness(nu))
        self._ke_diag: FloatArray = np.diag(self._ke).copy()
        # Stress is sampled at each element's eight corners and the worst kept.
        # Bending stress peaks at the surface, and a voxel's centre sits half a
        # voxel in from it: centre sampling reads a beam three voxels thick at
        # 75% of its true root stress, corner sampling at about 90%, nearly
        # independent of thickness (tests/test_fea.py checks both).
        d = e0 * _elasticity(nu)
        self._stress: FloatArray = np.stack(
            [d @ _strain_displacement(c.astype(np.float64)) / self.spacing for c in _CORNERS]
        )

        # One force vector per load case, in case order.
        self.cases = sorted({load.case for load in self.loads})
        self.forces = np.zeros((len(self.cases), self.n_dof), dtype=np.float64)
        n_nodes = int(node_id.size)
        for load in self.loads:
            force = self.forces[self.cases.index(load.case)]
            if isinstance(load, FieldLoad):
                carrying = load.region.ravel()
                corners = nodes[carrying].ravel()
                per_voxel = load.forces.reshape(-1, 3)[carrying]
                for axis in range(3):
                    # Each voxel's force is shared equally by its eight nodes.
                    share = np.repeat(per_voxel[:, axis] / 8.0, 8)
                    force[axis::3] += np.bincount(corners, weights=share, minlength=n_nodes)
            else:
                load_nodes = np.unique(nodes[load.region.ravel()])
                for axis, f in enumerate(load.force):
                    force[3 * load_nodes + axis] += float(f) / len(load_nodes)
        self._fixed = np.zeros(self.n_dof, dtype=bool)
        support_mask = np.zeros(shape, dtype=bool)
        for support in self.supports:
            support_mask |= support.region
            held = np.unique(nodes[support.region.ravel()])
            for axis, is_fixed in enumerate(support.fixed):
                if is_fixed:
                    self._fixed[3 * held + axis] = True
        self._support_mask = support_mask
        attached = support_mask.copy()
        for load in self.loads:
            if isinstance(load, Load):
                attached |= load.region
        # Stress right where a load or support attaches is dominated by how the
        # attachment was modelled -- a force smeared over a few nodes, a face
        # held perfectly rigid -- rather than by the part. One voxel of margin.
        self._attachment = ndimage.binary_dilation(attached)

    def _check_stiffness(self, stiffness: FloatArray | None) -> FloatArray | None:
        if stiffness is None:
            return None
        stiffness = np.asarray(stiffness, dtype=np.float64)
        if stiffness.shape != self.shape:
            raise ValueError(f"stiffness must have shape {self.shape}, got {stiffness.shape}")
        if not np.all(stiffness > 0):
            raise ValueError("stiffness must be positive everywhere")
        return stiffness

    @property
    def attachment(self) -> BoolArray:
        """Voxels at or next to a load or support, excluded from :meth:`peak_stress`."""
        mask: BoolArray = self._attachment
        return mask

    def solve(
        self,
        solid: BoolArray,
        *,
        stiffness: FloatArray | None = None,
        rtol: float = 1e-5,
        maxiter: int = 20_000,
    ) -> FEAResult:
        """Solve for the displacement of the part made of ``solid`` voxels.

        ``stiffness`` (each voxel's, as a multiple of the material's) overrides the one this solver was made
        with, for this solve.
        """
        solid = np.asarray(solid, dtype=bool)
        scaled = self.stiffness if stiffness is None else self._check_stiffness(stiffness)
        if solid.shape != self.shape:
            raise ValueError(f"solid must have shape {self.shape}, got {solid.shape}")

        labels, _ = ndimage.label(solid)
        anchored = np.unique(labels[self._support_mask & solid])
        kept = np.isin(labels, anchored[anchored > 0])
        if not all(self._reaches(load, kept) for load in self.loads):
            return self._unsupported(kept)

        elements = np.flatnonzero(kept.ravel())
        edof = self._edof[elements]
        flat = edof.ravel()
        active = np.zeros(self.n_dof, dtype=bool)
        active[flat] = True
        free = np.flatnonzero(active & ~self._fixed)
        if not len(free):
            return self._unsupported(kept)
        n_dof = self.n_dof
        ke = self._ke
        scale = None if scaled is None else scaled.ravel()[elements]

        def matvec(x: FloatArray) -> FloatArray:
            full = np.zeros(n_dof, dtype=np.float64)
            full[free] = x
            product = full[edof] @ ke
            if scale is not None:
                product *= scale[:, None]
            y = np.bincount(flat, weights=product.ravel(), minlength=n_dof)
            result: FloatArray = y[free].astype(np.float64, copy=False)
            return result

        weights = np.tile(self._ke_diag, len(elements))
        if scale is not None:
            weights = weights * np.repeat(scale, 24)
        diagonal = np.bincount(flat, weights=weights, minlength=n_dof)
        diagonal = diagonal.astype(np.float64, copy=False)
        inverse_diagonal = 1.0 / diagonal[free]

        def precondition(r: FloatArray) -> FloatArray:
            result: FloatArray = r * inverse_diagonal
            return result

        size = (len(free), len(free))
        operator = LinearOperator(size, matvec=matvec, dtype=np.float64)
        jacobi = LinearOperator(size, matvec=precondition, dtype=np.float64)
        iterations = 0

        def count(_: FloatArray) -> None:
            nonlocal iterations
            iterations += 1

        energy = np.zeros(int(np.prod(self.shape)), dtype=np.float64)
        vm = np.zeros_like(energy)
        compliances: list[float] = []
        largest = 0.0
        converged = True
        first: FloatArray | None = None
        # Each load case is its own solve. The design is judged on all of them:
        # strain energy adds up, so material that works in either case is kept,
        # and stress is the worst any case puts on a voxel.
        for force in self.forces:
            u_free, info = cg(
                operator, force[free], rtol=rtol, maxiter=maxiter, M=jacobi, callback=count
            )
            converged &= info == 0
            u = np.zeros(n_dof, dtype=np.float64)
            u[free] = u_free
            ue = u[edof]
            element_energy = 0.5 * np.einsum("ij,ij->i", ue @ ke, ue)
            stress = np.max([_von_mises(ue @ corner.T) for corner in self._stress], axis=0)
            if scale is not None:
                element_energy = element_energy * scale
                stress = stress * scale
            energy[elements] += element_energy
            vm[elements] = np.maximum(vm[elements], stress)
            compliances.append(float(force[free] @ u_free))
            largest = max(largest, float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()))
            if first is None:
                first = u
        assert first is not None
        von_mises = vm.reshape(self.shape)
        nx, ny, nz = self.shape
        return FEAResult(
            supported=True,
            compliance=float(sum(compliances)),
            displacement=first.reshape(nx + 1, ny + 1, nz + 1, 3),
            strain_energy=energy.reshape(self.shape),
            von_mises=von_mises,
            solid=kept,
            peak_stress=self.peak_stress(von_mises, kept),
            converged=converged,
            iterations=iterations,
            max_displacement=largest,
            case_compliance=tuple(compliances),
        )

    @staticmethod
    def _reaches(load: Load | FieldLoad, kept: BoolArray) -> bool:
        """Whether a load is carried by the material connected to a support.

        A point load must touch that material. A distributed one may lose a
        sliver -- a voxel of surface the meshing left floating -- but not more
        than 1% of its force, or the design is ignoring a real part of its load.
        """
        if isinstance(load, Load):
            return bool((load.region & kept).any())
        magnitude = np.linalg.norm(load.forces, axis=-1)
        total = float(magnitude.sum())
        return total == 0.0 or float(magnitude[~kept].sum()) <= 0.01 * total

    def peak_stress(
        self, von_mises: FloatArray, solid: BoolArray, *, percentile: float = 99.0
    ) -> float:
        """The stress to judge the part by: a high percentile, away from the attachments.

        Not the maximum. A voxel model has stress singularities wherever a load
        is applied over a few nodes, a support is held perfectly rigid, or a
        staircase surface makes a re-entrant corner, and the single worst voxel
        reports the mesh rather than the part. The 99th percentile of what is
        left is stable under refinement and still finds a member that is
        genuinely overloaded.
        """
        judged = solid & ~self._attachment
        if judged.sum() < 20:
            judged = solid
        if not judged.any():
            return float("inf")
        return float(np.percentile(von_mises[judged], percentile))

    def _unsupported(self, kept: BoolArray) -> FEAResult:
        nx, ny, nz = self.shape
        zeros = np.zeros(self.shape, dtype=np.float64)
        return FEAResult(
            supported=False,
            compliance=float("inf"),
            displacement=np.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=np.float64),
            strain_energy=zeros,
            von_mises=zeros.copy(),
            solid=kept,
            peak_stress=float("inf"),
            converged=False,
            iterations=0,
            max_displacement=float("inf"),
            case_compliance=tuple(float("inf") for _ in self.cases),
        )


def _resolve(spec: str | Material) -> Material:
    # VoxelFEA's ``material`` argument shadows the module function inside it.
    return material(spec)


def _von_mises(stress: FloatArray) -> FloatArray:
    sx, sy, sz, txy, tyz, tzx = stress.T
    result: FloatArray = np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )
    return result
