# whittle

Evolutionary optimization of 3D parts. Take material away until only what
carries the load is left, or push a surface around until the flow over it costs
less. OBJ and STL at both ends.

## The bet

Strength-per-material and aerodynamic shaping sound like one problem and are
not. They want different geometry representations, and most tools pick one and
quietly inherit its limits.

| | can open a hole | keeps the mesh watertight | the question it answers |
|---|---|---|---|
| `DensityField` | yes | after meshing | where does material need to be at all? |
| `FFDLattice` | **no** | always | what shape should this surface be? |

A deformation lattice cannot discover that a bracket wanted a hole through the
middle — no arrangement of control points will produce one. A density field can,
but it has to be meshed before anyone can print it, and it has no notion of the
smooth surface a duct needs. So this library has both, behind one interface, and
the optimizer never learns which it is driving.

That seam is the whole design. Everything else is replaceable.

## Why evolutionary, when gradient methods are faster

For textbook compliance minimization they *are* faster, and considerably. If
your problem is a linear-elastic part under a fixed load with a volume
constraint, adjoint-based SIMP will beat this and you should use it.

Evolutionary search earns its place where the gradient isn't available or isn't
meaningful:

- **Objectives you cannot differentiate through.** A CFD solve, a meshing step,
  a slicer's own verdict on printability.
- **Manufacturing constraints that are discrete.** Minimum tool reach, a
  draft-angle rule, "this face must stay flat", a fixed catalogue of lattice
  cells. These are conditionals, not smooth penalties.
- **Several objectives at once,** where you want the trade-off curve rather
  than one point you got by guessing weights.

That is the same reasoning behind [EML](https://github.com/greentheo/EML), and
the `optimizer` seam in `whittle.optimize.evolve` exists so that engine can
replace scipy's differential evolution without touching anything else.

## Status

Honest version, because a topology optimizer that reports confident numbers
from a solver it does not have is worse than no optimizer.

**Works, tested, runs today**

- OBJ and STL read and write, including welding STL's per-triangle corners back
  into a shared-vertex mesh
- `DensityField` — coarse control grid, upsampled, thresholded, marching cubes;
  `keep`/`void` masks the optimizer cannot erode
- `FFDLattice` — trivariate Bezier deformation, basis precomputed per vertex
- `MaterialFraction`, `MeshVolume`, `AnchorConnectivity`
- The evolutionary loop, frozen-parameter handling, end-to-end to an STL

**Deliberately not implemented**

- `Compliance` — needs a real FEA solve. Raises `NotImplementedError` rather
  than returning a plausible number. **This is why no structural claim in this
  repo is proven yet.**
- `DragCoefficient` — needs a flow solver. Same policy.

`AnchorConnectivity` stands in for structural validity today. It checks that
material actually joins the load to the supports, which is necessary and
nowhere near sufficient — a one-voxel thread passes it and would snap on
contact. It is scaffolding that makes the pipeline runnable and testable, not
physics.

## Install

```bash
git clone https://github.com/greentheo/whittle && cd whittle
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Three dependencies: numpy, scipy, scikit-image. The mesh IO is eighty lines in
`src/whittle/io.py` rather than a mesh library, because we need triangles in and
triangles out and nothing else.

## A first run

```bash
python examples/01_bridge.py
```

Evolves a density field to connect three anchor pads using as little material as
possible, and writes `out/bridge.stl` and `out/bridge.obj`.

Measured on a 32x6x16 design box with a 8x2x4 control grid — 64 free
parameters, 10,009 evaluations, **10.0s on one core**:

```
fitness=0.381150  (material=0.3811, connectivity=0.0000)
mesh:    1254 vertices, 2500 triangles
volume:  642.6 of 3072 in the design box
```

Connectivity reached zero, so the three pads really are joined by material.

**Two things in that result are worth staring at,** because they are what the
proxy costs you:

1. The structure it found is a thin thread. Nothing in the objective rewards
   thickness, so nothing stops it going as thin as the grid allows. Under a real
   load it would fail immediately.
2. Of 772 solid voxels, **35 are a blob floating free of every anchor** —
   connected to nothing, carrying nothing, and unprintable without supports.
   The only thing arguing against it is its own small material cost.

Both are artifacts of scoring geometry instead of physics, and both disappear on
their own once `Compliance` lands: material that carries no load contributes
nothing to stiffness, so the material term drives it out without anyone having
to write a rule against it. That is the argument for ROADMAP.md item 1, stated
in results rather than in prose.

```python
import numpy as np
from whittle import AnchorConnectivity, DensityField, MaterialFraction, WeightedSum, evolve, io

shape = (32, 6, 16)

def pad(x0, x1, z0, z1):
    mask = np.zeros(shape, dtype=bool)
    mask[x0:x1, :, z0:z1] = True
    return mask

anchors = [
    pad(1, 4, 0, 2),      # left foot
    pad(28, 31, 0, 2),    # right foot
    pad(14, 18, 14, 16),  # load pad up top
]
keep = anchors[0] | anchors[1] | anchors[2]

backend = DensityField(shape, control_shape=(8, 2, 4), keep=keep)
objective = WeightedSum({
    MaterialFraction(): 1.0,
    AnchorConnectivity(anchors): 4.0,
})

result = evolve(backend, objective, seed=0, maxiter=25, popsize=6)
print(result)                       # fitness, evaluations, per-objective terms
io.save(result.mesh, "bridge.stl")
```

`keep` is the boundary condition the optimizer cannot argue with: those voxels
stay solid no matter what genome is proposed, which is how a load pad or a bolt
boss survives a search whose whole purpose is removing material.

## Where this is going

`ROADMAP.md`, in the order the items should be built. The short version: the FEA
solver is item 1, and until it lands this is a well-shaped search loop around a
geometric proxy.

---

A tool that tells you how much material you saved, without having checked the
part still holds, is not a tool. It is a flattering picture.
