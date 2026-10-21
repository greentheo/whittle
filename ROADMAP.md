# Roadmap

In build order.

## Done

- **Linear-elastic FEA on the voxel grid** (was item 1). Hex elements on the
  same grid as the geometry, matrix-free CG, hard-kill material, stress at
  element corners. Within 1-4% of beam theory on deflection, 10-15% low on peak
  stress. Hard-kill rather than SIMP because what is printed is solid or
  nothing, and BESO, not a gradient method, drives it.
- **The slider.** `LoadStudy.sweep()` and `FlowStudy.sweep()` return a
  `Sweep`: stations from the original part to the most aggressive design, each
  evolved from the last. This is the part-in, slider-out workflow a GUI wraps.
- **The web GUI.** `python -m whittle.gui`: load a part, click to place
  supports and loads (a whole face or a ball around a point, with the
  selected voxels drawn), or pick the flow and its constraints; run with
  progress and cancel; slide through the stations with stress or drag
  heatmaps, trade-off charts, and STL download. Standard library server,
  three.js page.
- **Both at once.** `DualStudy` nests the two: carve the outside to a few
  drag budgets, lighten inside each shape (keeping its skin) under the user's
  loads plus that shape's own wind -- head-on and in a crosswind, built by the
  crossflow principle -- and refine along the carving axis where neighbours
  differ most. FEA load cases and distributed loads came with it. In the GUI
  as a third mode, with two sliders and a map.
- **A fast approximate drag model** (the "panel method" route of old item 4,
  done as an engineering model rather than a potential-flow solve, because
  potential flow predicts zero drag). Calibrated to textbook high-Reynolds
  values, including the 2D wakes of long sections.

**The two artifacts from the bridge example.** The old roadmap predicted that
the thread-thin structure and the floating blob would vanish once a real
compliance objective existed, without anyone writing a rule against them. Under
BESO they do: the bracket sweep produces members two to three voxels thick at
every station, and floating material is dropped before every solve because it
carries nothing.

**Warm-started solves (old item 2) were measured and do not help.** Starting
CG from the previous design's displacement saved nothing -- 570 iterations
against 578 from zero -- because removing even 2% of the voxels leaves a
residual as large as the load itself. An incomplete-LU preconditioner was also
tried; CG diverges under it, since ILU of this system is not positive definite.

## 1. Multigrid preconditioning

The structural solve's real cost. Jacobi-preconditioned CG needs 300-1000
iterations on a slender part, so a sweep is about 30 s at 48 voxels along the
longest side and 86 s at 60, and the gap widens with every step up. Geometric
multigrid on the voxel grid -- coarsen by two per level, Galerkin coarse
operators, a smoother on each -- is the standard fix in topology optimization
and should make iteration counts nearly independent of resolution.

## 2. A real flow check

The drag model ranks shapes; something should check its rankings.

1. **Export to OpenFOAM** for the two or three stations a person is choosing
   between. Trustworthy, and slow enough that it belongs outside the loop.
2. **A lattice-Boltzmann visual wind tunnel** in numpy, at low resolution:
   too low a Reynolds number to trust its drag at highway speed, but enough to
   show where the flow separates and what the wake looks like -- a view the
   GUI could draw beside the drag heatmap.

Either is also the way to recalibrate the drag model if the rankings disagree.

## 3. Pareto search instead of a weighted sum

A sweep is already a hand-built trade-off curve: removal on one axis,
stiffness or drag on the other. A proper multi-objective method (NSGA-II)
would find the front in one run, and could add a third axis -- printability,
say -- that the sweep cannot. This is where EML's multi-objective work is
most directly reusable.

## 4. Manufacturing constraints as first-class objectives

The real reason to keep a population method around.

- Minimum feature size and wall thickness beyond what the BESO filter imposes
- Overhang angle for printing without supports, draft for a two-part mould
- Tool reach for subtractive manufacturing
- Snapping internal structure to a catalogue of lattice cells

## 5. Parallel evaluation

A flow station runs ten thousand independent drag evaluations; scipy's
`workers=` passes straight through `evolve()` and has not been profiled. BESO
is sequential by nature, so the structural side waits on item 1 instead.

## 6. Open structures in the wind

`DualStudy` keeps the outer skin, which is what makes its coupling one-way:
lightening never changes the outside, so it never changes the drag. An open
structure -- a rack frame, a lattice fairing, a bracket that could lose a
corner -- would rather punch through, and then every lightening step changes
the drag and the two have to alternate. That needs a flow representation that
can hold holes, which the deformation lattice cannot: the drag model scoring
the marching-cubes surface of a density field is the obvious route, and the
terracing it brings is the obvious problem.

## Open questions

- **How far to trust the drag model on a car roof.** It scores bodies in free
  air. A roof box sits in the car's boundary layer and the flow off the
  windscreen. The rankings probably survive; the absolute savings certainly
  shrink. Item 2 is how to find out by how much.
- **Control-lattice detail for flow.** Two divisions per axis can round a
  nose, slope it and taper a tail, and cannot make the wiggles that would game
  the model. Three might find better shapes, or might find exploits.
- **Where the genome should live.** An implicit SDF built from primitives is a
  third representation worth trying -- far fewer parameters than a density
  field, and a topology that can still change.
