# Roadmap

In build order. Items 1 and 2 are what turn this from a search loop into a
structural tool; everything after is reach.

## 1. Linear-elastic FEA on the voxel grid

The one that matters. Replaces `AnchorConnectivity` with `Compliance` and makes
every claim in this repo mean something.

- Trilinear hexahedral elements on the existing density grid, so the geometry
  backend needs no changes at all.
- SIMP interpolation, `E(rho) = E_min + rho**p * (E_0 - E_min)` with `p ≈ 3`.
  The penalization is what drives intermediate densities to solid or void
  instead of leaving a grey mush that means nothing physically.
- Matrix-free conjugate gradient, element-by-element. Assembling a global
  stiffness matrix once is fine; assembling one per candidate per generation is
  not, and that is the actual cost structure here.
- Boundary conditions as masks on the same grid as `keep`/`void`, so a problem
  is defined in exactly one coordinate system.

**Done when:** the MBB beam and the cantilever reproduce the shapes from the
topology-optimization literature, and `examples/01_bridge.py` switches to
`Compliance` with a one-line change at the call site.

**Two current artifacts that should vanish rather than be patched around.** The
bridge example today produces a thread thin enough to fail instantly, and leaves
35 voxels of material floating unattached to any anchor. Both are the geometric
proxy showing through: it scores paths, not load. Resist adding a minimum-
thickness rule or a floating-material filter to hide them — under a real
compliance objective, material that carries no load earns no stiffness and the
material term removes it unprompted. If they survive item 1, *then* they are
real problems and worth a rule.

## 2. Warm-started solves

Candidates mostly resemble their parents, so each solve should start from the
parent's displacement field rather than from zero. Expect most of the per-
generation cost to come off.

Depends on item 1, and on the optimizer exposing parent identity — which scipy's
`differential_evolution` does not. This is the first place the EML engine would
earn its keep over the stock one.

## 3. Pareto search instead of a weighted sum

`WeightedSum` can only find one point on the material/stiffness trade-off, and
which point depends on weights you guessed. A proper multi-objective method
(NSGA-II is the obvious starting point) returns the front, and the front is what
you actually want to look at when deciding how light to go.

This is also the item that most directly reuses EML's multi-objective work —
there, loss against free-parameter count; here, compliance against material.

## 4. A flow solver for `DragCoefficient`

Three routes, in increasing order of risk:

1. **OpenFOAM per candidate.** Trustworthy, far too slow inside a generation
   loop except with a tiny population and an overnight budget.
2. **A fast approximate solver** — panel method or lattice Boltzmann. Only needs
   to rank candidates correctly against each other, which is all selection uses.
   Probably where to start.
3. **A surrogate** trained on a few hundred real solves. Fastest by far, and the
   one most likely to produce confident nonsense once the search wanders outside
   its training distribution. Needs a trust region and an honest refresh policy.

Geometry is already settled for this: `FFDLattice` keeps the surface closed and
valid for whatever solver wins.

## 5. Manufacturing constraints as first-class objectives

The real reason to run an evolutionary method rather than adjoint SIMP. All of
these are conditionals rather than smooth penalties, so they fit a population
method naturally and a gradient method badly.

- Minimum feature size and minimum wall thickness
- Overhang and draft angle, for printing or for a two-part mould
- Tool reach and line-of-sight access for subtractive manufacturing
- Snapping internal structure to a catalogue of lattice cells

## 6. Parallel evaluation

The population is embarrassingly parallel and currently runs on one core.
`evolve(workers=...)` passes through to scipy today but has not been profiled,
and the FEA solve from item 1 is what will make it worth doing properly.

## Open questions

- **Control-grid resolution.** Upsampling imposes a length scale, which
  suppresses checkerboarding for free. How coarse can it be before it stops
  being able to express a good structure? Worth a sweep once item 1 can score
  the answer.
- **Where the genome should live.** A density field is one encoding; an implicit
  SDF built from primitives is another, with far fewer parameters and a
  topology that can still change. Worth trying as a third backend once there is
  a real objective to compare them against.
