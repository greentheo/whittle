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
- **Choosing where material comes from.** `whittle(...)` on every study names
  the surfaces material may come from; every other surface stays as it is --
  a kept skin for strength, zero-weight vertices for airflow, with the carving
  easing in over a few triangles. A "Whittle" tool in the GUI.
- **Tuning a parametric model.** `ParametricStudy` searches a model's own
  dimensions -- any function from a dict of parameters to a mesh, CadQuery
  included -- against the drag model, and gives the same slider, each station
  a set of numbers to build to. CAD tessellations are welded and evened
  before scoring. Its first user is Caliper, which builds the model and
  writes the chosen numbers back into it.
- **A flow study carves the part as sent.** It used to re-mesh every part
  from voxel centres, which put it up to half a voxel proud on any axis not
  a whole number of voxels long -- a 39" box came back an inch bigger, with
  8% more volume and its flat bottom an inch lower -- and rounded every edge,
  which made a sharp box look 15% slipperier than it is. Now a clean closed
  part is carved as it is, only welded and evened, so station 0 is the part
  itself. A messy one is re-meshed from how much of each voxel it fills,
  within a quarter of a voxel of where it was.
- **Strength meshes and masses follow the part's surface.** A load study's
  stations, and a dual study's lightened ones, are meshed from how much of
  each voxel the part filled where they meet its outside, and weighed the
  same way. They used to come back up to half a voxel proud and as much as
  15% heavy -- worst on a lightened foam pod, which is mostly skin. Material
  taken away still meshes on voxel boundaries, which is all there is to know
  about it. `Grid.coverage` measures the fraction exactly along each ray.
- **Frames.** `FrameStudy` models tube structures as beams along their
  centrelines -- 3D frame elements, rigid joints, square tube and bar from a
  catalog -- and sweeps the tube of each group of members up or down until
  the lightest frame that keeps its safety factor. Reports stress, Euler
  buckling, deflection and the reactions at the supports, and flags fatigue
  territory under 3.5x. Built for Caliper's hitch platform frame.
- **Obstacles.** Fixed bodies a part rides in the wake of -- the car in front
  of a hitch box -- shelter it without being scored or moved, in the drag
  model, `FlowStudy` and `ParametricStudy`.
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

## Next for frames

- **Plate mode.** Lighten a gusset as a 2D plate a plasma table can cut: an
  outline and a thickness in, through-holes out as DXF and SVG, with
  minimum hole and web sizes and a margin along the welds. A one-voxel-thick
  `LoadStudy` already gets most of the way; the frame study's end forces at
  a joint are what to load the gusset with (`CaseResult.end_forces`).
- **Legs and the parked case.** How a load splits between a hitch and legs
  depends on how the legs are set and how the car settles on its springs:
  supports with stiffness, and both bounds reported.
- **Dropping members,** not just resizing them: start from every brace that
  could be there and take away the ones that carry least.
- **Effective lengths.** Buckling takes each member's own length; a column
  free at one end buckles at twice it.

## Open questions

- **How far to trust the drag model on a car roof.** It scores bodies in free
  air. A roof box sits in the car's boundary layer and the flow off the
  windscreen. The rankings probably survive; the absolute savings certainly
  shrink. Item 2 is how to find out by how much.
- **Control-lattice detail for flow.** Two divisions per axis can round a
  nose, slope it and taper a tail, and cannot make the wiggles that would game
  the model. Three might find better shapes, or might find exploits.
- **Where the genome should live.** An implicit SDF built from primitives is a
  representation worth trying -- far fewer parameters than a density field,
  and a topology that can still change. A parametric model is the other end
  of that: fewest parameters of all, and exactly the topology its author drew.
- **Curvature by rings, not millimetres.** The drag model smooths curvature
  over four rings of neighbouring faces, so how far that reaches depends on
  the mesh. Both studies split a part's long edges to about the same size,
  and the rankings hold, but across large fillets the absolute number moves
  by 5-10% with mesh density. Smoothing over a fixed distance would fix it,
  at the cost of re-checking the calibration.
- **Parts thinner than two voxels.** Meshes and masses follow the part's
  real surface to within about a tenth of a voxel on anything thicker, but
  a wall one voxel thick has both its faces in the same voxel, and no
  fraction filled says where each one is. It comes out thinner than drawn.
  A finer grid is the fix, as it is for everything else that small.
