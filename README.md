# whittle

Evolutionary optimization of 3D parts, for people who build things. Take
material away until only what carries the load is left, or carve a shape down
until the air slips past it. OBJ and STL at both ends.

Bring a part. Say what pushes on it, or which way the wind blows. Get back a
**slider**: the part with more and more material removed, every step scored,
from conservative to extreme. You pick how far to go.

```python
from whittle import LoadStudy, FlowStudy, io

# Lighten a bracket without losing the strength it needs.
study = LoadStudy(io.load("bracket.stl"), material="pla")
study.support(box=((0, 0, 0), (4, 8, 80)))          # screwed to the wall
study.load((0, 0, -245), box=((70, 0, 76), (80, 8, 80)))   # 25 kg on the tip
sweep = study.sweep(remove_up_to=0.8)
print(sweep)
io.save(sweep.at(0.6).mesh, "bracket-60.stl")

# Carve a roof box until it cuts through the air.
study = FlowStudy(io.load("roof_box.stl"), flow=(-1, 0, 0), mirror="y", flat=("-z",))
sweep = study.sweep(remove_up_to=0.25)
io.save(sweep.slide(0.5).mesh, "roof_box-halfway.stl")
```

## The GUI

```bash
python -m whittle.gui
```

That opens a page in your browser, served from your own machine (`whittle-gui`
does the same once the package is installed). Load an STL or OBJ -- or one of
the four examples -- and then:

- **Strength:** click the part to place a *support* (where it is screwed or
  clamped) and a *load* (kilograms or newtons, down or into the surface).
  "Whole face" takes the face you clicked; "around point" a ball of the
  radius you set. Every placement shows the voxels it actually selected, so
  what the solver will hold or push is never a guess.
- **Airflow:** pick which way the air moves, and optionally keep the part
  symmetric, keep faces flat, protect a box inside it, or mark it as a slice
  of something long.
- **Whittle:** in any mode, click the surfaces material may come from. Name
  none and the physics decides everywhere; name some and every other surface
  stays exactly as it is.
- **Both:** the airflow set-up plus where the part is held, a cargo space
  and its weight, and a crosswind. You get two sliders -- shape for the wind,
  and lighten -- and a map of every design, mass against drag, coloured by
  safety, with the best trade-offs at the safety you ask for marked.
- **Run**, then **slide**. Each station shows its mesh with a heatmap --
  stress for strength, drag per area for airflow -- its numbers, and where it
  sits on the trade-off curve. "Cut in half" slices the part to show what
  lightening did inside a skin. Download any station as an STL.

The page loads three.js from a CDN, so the browser needs to be online. The
server needs nothing beyond the three dependencies, binds to localhost only,
and keeps everything in memory.

## Why a slider and not an answer

Neither physics model here is accurate enough to pick *the* answer, and no
model would be -- the load was estimated, the material idealized, the print
will have layer lines. A single optimum computed from those inputs is false
precision. What the physics *is* good at is the shape of the trade-off: how
much stiffness the first 30% of material costs (almost none), where the safety
factor falls off a cliff, how much of the drag saving the first 10% of carving
already buys. That is a table to read, so that is what a sweep gives you:

```
removed  mass g  stiffness  deflect mm  stress MPa  safety
-------  ------  ---------  ----------  ----------  ------
     0%    37.7       100%       0.177         4.3    9.4x
    10%    33.9        98%       0.179         4.3    9.4x
    20%    30.1        95%       0.184         4.3    9.3x
    30%    26.3        89%       0.193         4.4    9.1x
    40%    22.4        80%       0.210         4.9    8.1x
    50%    18.7        68%       0.242         6.4    6.3x
    60%    14.9        53%       0.308         8.4    4.8x
    70%    11.1        36%       0.439        13.3    3.0x
    80%     7.3        19%       0.850        24.1    1.7x
```

That is `examples/02_shelf_bracket.py`: an 8 mm PLA bracket carrying 25 kg.
Seventy percent of it can go and it still has a 3x margin; the last ten percent
halves that. The designs it passes through are the textbook ones -- a diagonal
strut under compression and a tie along the top in tension -- found by the
solver, not drawn by anyone.

Each station evolves from the one before it, so sliding along a sweep reads as
one part being whittled down, not as unrelated designs, which is what makes
the GUI's slider worth dragging.

## The two studies

| | `LoadStudy` | `FlowStudy` |
|---|---|---|
| you place | supports and loads (`at=` a point or `box=`) | the flow direction |
| it protects | `keep(...)` -- bolt bosses, mating faces | `keep(box)` -- cargo space; `flat=` faces; `mirror=` |
| `whittle(...)` names | the surfaces pockets may open onto | the surfaces the carving may reshape |
| physics | linear-elastic FEA on a voxel grid | engineering drag model on the surface mesh |
| search | BESO: remove the voxels doing the least work | differential evolution over a deformation lattice |
| geometry | can open holes, grow trusses | stays one smooth closed surface, never grows |
| each station reports | mass, stiffness kept, deflection, stress, safety factor | drag vs original, Cd, CdA, newtons at speed |

The split is the old one this library started from. Lightening a part wants a
representation that can open a hole -- no smooth deformation of a solid block
will ever find a truss. Shaping against a flow wants the opposite: a surface
that stays closed and smooth, because drag is decided by its curvature.

## Choosing where material comes from

Left alone, a study takes material from wherever the physics says. Often you
know better: the face of a bracket that shows, the side of a roof box that
has to clear a rail, the mating face that has to stay flat. Name the
surfaces material *may* come from, with the same `at=`/`normal=` a click gives
(or `box=`), and every other surface stays exactly as it is:

```python
study.whittle(at=(40, 8, 60), normal=(0, 1, 0))      # pocket the bracket from its front only
roof.whittle(at=(1000, 0, 200), normal=(1, 0, 0))    # reshape the nose ...
roof.whittle(at=(0, 0, 400), normal=(0, 0, 1))       # ... and the top, nothing else
```

- **Strength:** every surface you did not name keeps a skin, so pockets open
  only onto the named ones. The shelf bracket pocketed from its front alone
  keeps a solid back plate and 65% of its stiffness at half its mass, against
  67% when it may hollow from anywhere -- a printable ribbed bracket for two
  points of stiffness.
- **Airflow:** each vertex of the surface follows the carving with a weight,
  0 on surfaces you did not name and easing up to 1 over a few triangles into
  the ones you did. Unnamed surfaces do not move at all -- zero millimetres,
  which the tests check -- and the carved region bends smoothly where it meets
  them instead of tearing. With a 20% budget, the roof box carved on its nose
  and top only reached 37% of its original drag, against 34% when every
  surface could change.
- **Both:** naming surfaces shapes the carving. Lightening already keeps the
  whole skin, which is what keeps each shape's drag intact, so it works inside
  whatever you name.

## Both at once

A part that sits in the wind and carries a load -- a roof pod on its bars, a
fairing, a light-bar bracket -- wants both: slipperier *and* lighter, without
breaking. `DualStudy` does it without the obvious `n x m` grid of runs, because
the two couple in one direction only. Carving changes the outside; lightening,
as long as it keeps the outer skin, never does. So the study nests them:

1. carve the outside to a few drag budgets, coarse first,
2. lighten inside each carved shape, keeping its skin, under your loads plus
   *that shape's own* wind -- once head-on, once in a crosswind,
3. fill in more shapes wherever neighbouring ones differ most.

Each lightening run yields its whole column of designs at once, so a map of
five shapes by five weights is five carves and five lightening runs: about
two minutes for the 2 m foam pod in `examples/05_dual_roof_box.py`.

```python
from whittle import DualStudy, io

pod = DualStudy(io.load("pod.stl"), flow=(-1, 0, 0), material="xps", mirror="y", flat=("-z",))
pod.support(box=((350, -460, -10), (450, 460, 40)))     # front roof bar
pod.support(box=((-450, -460, -10), (-350, 460, 40)))   # rear roof bar
pod.cargo(((-750, -350, 60), (750, 350, 320)), kg=50)   # stays empty, 50 kg on its floor
sweep = pod.sweep()
for i, j in sweep.front(min_safety=20):
    print(sweep.rows[i].stations[j].metrics)
```

What it found is worth knowing: for that pod, drag and weight do not fight --
carving the nose removes foam too, so slipperier and lighter go together. The
real trade-off is against strength, so how conservative to be is the choice
that moves the answer:

| at least | best design | mass | drag | safety |
|---|---|---|---|---|
| 5x safe | carved 24%, lightened 26% | 7.6 kg | 18% | 25x |
| 30x safe | carved 18%, lightened 32% -- or 24% / 15% | 8.0 / 8.7 kg | 32% / 18% | 34x / 35x |
| 50x safe | carved 6%, lightened 30% -- or 18% / 0% | 9.9 / 12.0 kg | 83% / 32% | 65x / 52x |

From 30x up a genuine choice appears between a lighter pod and a slipperier
one.
The safety factors are high because the foam carries only 50 kg and the wind;
they are bare foam, with no glass or carbon skin, so they are conservative.

The crosswind uses the crossflow principle: the drag at driving speed along
the direction of travel, plus the broadside drag at crosswind speed across it.
Both are head-on cases, which is where the drag model is calibrated -- its own
answers for a yawed flow are not to be trusted.

**One limit to know.** Voxels are about 1/48 of the part, so walls come out
at least that thick: 40 mm on a 2 m pod. That is a real foam wall, which is
why the example is foam. It is not a 4 mm moulded shell, and for one of those
this would tell you where ribs and mounts belong, not what it weighs.

## A parametric model instead of a mesh

A flow study carves whatever STL you give it, which is why it works on
anything -- and why what comes back is a mesh: the part's surface pushed
around by a lattice, as smooth as that allows, with no trace of the
dimensions it was drawn from. When the part already exists as a parametric model -- a CadQuery
script, an OpenSCAD file, a few lines of numpy -- search its own numbers
instead:

```python
from whittle import ParametricStudy

def build(p):                 # every parameter, as a dict -> a mesh, or (vertices, faces)
    ...

study = ParametricStudy(
    build,
    {"tail": (100, 600), "drop": (0, 400), "pinch": (0, 300)},  # what to search
    start={"tail": 300, "drop": 0, "pinch": 0, "width": 1880},  # the design as it stands
    flow=(-1, 0, 0),
    obstacles=[car],          # optional: what it rides in the wake of
)
sweep = study.sweep(remove_up_to=0.2)
sweep.at(0.10).design         # {"tail": 590.0, "drop": 112.0, ...}: numbers to build to
```

The slider is the same -- volume given up against drag saved -- but every
station is a real design from the model, as smooth as the model makes it, and
what it hands back is a handful of numbers to type into the model rather than
a mesh to reverse-engineer. A handful is also far easier to search than a
lattice's eighty-odd moves. The builder sets the pace: it runs about
`6 x parameters x 30 x effort` times per station, so keep the parameters few.

- **CAD exports are evened out first.** They repeat every vertex along the
  seams between faces, and pair big flat triangles with thin strips along
  fillets. Unwelded, a box scores as six separate plates. Every shape is
  welded, and its long edges split to about the size the drag model was
  calibrated on, before the drag model sees it -- as a flow study does with
  the part it is sent.
- **A design that will not build is skipped.** A fillet too big for its edge
  raises, the search moves on, and the sweep's note says how many failed and
  why.
- **The bounds say what may change.** Parameters not in the bounds are passed
  through from `start` untouched. A design may grow if its bounds allow -- a
  longer tail -- and its "removed" is then negative.

**Riding in a wake.** `obstacles=` -- on `ParametricStudy`, `FlowStudy` and
`DragModel` -- takes fixed bodies the part sits behind: the car in front of a
hitch box. A face of the part downstream of an obstacle's silhouette feels no
pressure and no friction, but still counts in the part's frontal area, so Cd
stays a coefficient on the part's own size. Obstacles are never scored and
never move, so keep them out of the part itself: merged into one STL, a flow
study's lattice would drag the car along with the box.

`examples/06_parametric_hitch_box.py` searches five numbers of a hitch box
lofted from rounded sections -- the radius on the front edges, the radius on
the long top edges, and where, how far down and how far in the tail tapers --
once in free air and once behind a slab standing in for the car:

| | drag | front r | top edge r | tail | drop | pinch |
|---|---|---|---|---|---|---|
| as drawn | 1197 N alone, 307 N behind the car | 20 | 20 | -- | 0 | 0 |
| free air, 5% of the volume given up | 70% | 300 | 24 | 330 | 58 | 56 |
| behind the car, 5% given up | 74% | 78 | 46 | 596 | 116 | 122 |

In free air the square front is three quarters of the drag, and the search
spends its first 5% rounding it as far as the bounds allow. Behind the car the
front is mostly in the car's wake, a tenth of the drag, and the same 5% goes
into a long tail taper instead: the top falling 12 cm and each side drawing in
12 cm over the last 60 cm, ending at about 20 degrees -- the steepest the
model lets the flow follow before it lets go. Both run out of gains near 10%,
where the parameters reach their bounds. The model treats everything inside
the car's silhouette as sheltered, which flatters the box behind it: trust the
change of strategy more than the 307 N.

## Frames: tube structures

A 2" square tube with a 1/8" wall is 3 mm of steel on a frame nearly two
metres across -- far below what a voxel grid can see. A frame study models
each member as a beam along its centreline instead, with the real area and
stiffness of its tube, joined rigidly at the nodes, and sizes the tubes:

```python
from whittle import FrameStudy

frame = FrameStudy("steel")
inner, mouth = frame.node((228.6, 0, 228.6)), frame.node((76.2, 0, 228.6))
...                                                   # nodes along the centrelines
frame.member(inner, mouth, "2x2x1/8", group="shank")  # tubes by size, in inches
frame.support(mouth, fix=("ux", "uy", "uz", "rx"))    # a receiver: a socket and a pin
frame.support(inner, fix=("uy", "uz"))
frame.case("driving", factor=2.5)                     # bumps, the frame's own weight included
frame.spread(rail, total=(0, 0, -222), case="driving")
frame.point(cross, 0.4, force=(0, 0, -90), case="driving")  # or at a point along it
result = frame.analyze()                              # stress, safety, deflection per member
sweep = frame.sweep(groups={"shank": ["2x2x1/4", "2x2x5/16", "2x2xsolid"], ...})
sweep[-1].design                                      # {"shank": "2x2xsolid", ...}
```

`TUBES` holds the square tube and bar a fabricator can buy, keyed like
`"2x2x1/8"`, with `tube_size()` turning a key back into millimetres. Its
strength and stiffness match AISC's published tables to within 1%, at
their design wall of 0.93 of nominal, and its weight matches the catalog's,
which is at the nominal wall. `group` ties members cut from the same stock,
so a sweep sizes them together; it steps each group up or down its list of
tubes one size at a time -- first adding material where it closes the most
of the shortfall per kilogram if the frame as drawn falls short, then taking
it away where it costs the least -- and each station is one size step from
the last. Where no one step helps -- a heavier arm fixes the arm but
overloads the shank it hangs from -- it steps two groups together, or jumps
one further. `FrameStudy.from_dict()` takes the whole frame as plain data, as
another program would hand it over.

A member can be something other than tube: a foam-cored wall carrying load
beside the steel, say, given as its section's numbers and a material of its
own (`material=`, or `"material"` in the dict). A sweep leaves those members
as they are unless it is told what they may be.

`examples/07_hitch_frame.py` is the steel frame of a hitch cargo platform:
a shank in a 2" receiver, a 6" riser, an upper shank running back 50", and a
ladder 60" wide on top, 100 lb of gear on it, taken at 2.5x for bumps:

| | shank | riser | upper shank | rails | cross | mass | safety |
|---|---|---|---|---|---|---|---|
| as drawn | 2x2x1/8 | 2x2x1/8 | 2x2x1/8 | 2x2x1/8 | 2x2x1/8 | 31.6 kg | 1.0x |
| where the sweep ends | 2x2 solid | 2x2 solid | 3x3x1/8 | 1x1x1/16 | 2x2x0.083 | 32.8 kg | 3.2x |

As drawn, the shank yields inside the receiver the first time the car hits
a bump hard. The sweep's first moves are free: lighter rails and cross
members, which carry little, take weight off the end of a long lever. Then
it thickens the shank and riser up to solid bar -- the strongest thing that
fits a 2" receiver -- and that caps it at 3.2x, short of the 3.5x a welded
frame shaken on the road wants. The receiver holds 170 lb and 6,600 lb-in
standing still: that, not the 100 lb of gear, is the number to put against
the hitch's rating.

## Plates: gussets for the plasma table

The corners of a welded frame are where it cracks on the road, and a pair of
gusset plates is what keeps it from cracking. A plate study takes a flat
plate -- an outline and a thickness -- with welds that hold it and welds that
load it, lightens it, and hands back plates a plasma table can cut:

```python
plate = frame.gusset("jog_low", riser, shank, thickness=4.7625)  # 3/16" cheek plates
sweep = plate.sweep(remove_up_to=0.5)                             # whole to lightened
sweep[2].design.save("gusset.dxf")                                # R12 DXF in mm, or .svg
sweep[2].note        # how much of the corner's moment the plates take off its weld
plate.placement      # where the plate sits in the frame, to draw it in 3D
```

`FrameStudy.gusset()` puts cheek plates -- one each side of the tubes -- in
the plane two members span where they meet, over a band of each tube, and
loads them with what the frame says one member pushes into the joint.
Lightening only makes holes through the plate, and each station is cleaned
up before it is checked again as it will be cut: no hole narrower than
`min_hole`, no web narrower than `min_web`, a solid rim round the outline and
a margin beside every weld. `PlateStudy` is the same for any plate, with
`fix()`, `load()` and `keep()` on polygons of its own.

`examples/08_hitch_gussets.py` puts 3/16" plates on both corners of example
07's riser. The lightest pair that keeps 3.5x is 4.0x at each corner, with
two holes each -- a strut across the corner between them -- and takes about
80% of the corner's 10,970 lb-in off the tube-to-tube weld. On a 6" riser the
plates at the two corners meet, and the study says to cut each side as one.

## How far to trust it

**Structure -- validated.** Eight-node hexahedra, one per voxel, solved by
matrix-free conjugate gradients. Against beam theory, a voxel cantilever's tip
deflection comes out within 1-4%; its peak stress (the 99th percentile, kept a
voxel clear of the clamp and load) within 10-15% low. Tests hold both. The
honest gaps are in the inputs, not the solver:

- **Printed parts are not isotropic.** The materials table assumes 100% infill
  loaded along the layers. Across layers a print can be half as strong; at
  partial infill, stiffness and strength drop roughly with the infill. Read a
  safety factor under about 3 on a print as "build one and test it."
- **Loads are static.** No impact, no fatigue, no buckling. A thin compression
  strut that the table calls safe can buckle.
- **Resolution sets the finest member.** About 48 voxels along the longest side
  is the default; members come out at least two to three voxels thick.

**Frames -- validated, with one big blind spot.** Members are Timoshenko
beams, so shear counts where it matters -- a deep, foam-cored wall bends in
shear about as much as in bending -- and barely at all in a tube.
Deflection and twist match beam theory exactly and stress to within 1%, and
tube properties and weights match the AISC tables. What a frame model
cannot see is the weld: joints are perfectly rigid and stress is the
nominal stress in the tube, where a real weld concentrates it at its toe.
Welded steel under road vibration fails there, by fatigue, at stresses well
below yield -- which is why a frame study flags anything under 3.5x and why
the margins it asks for look generous. Compression members are checked
against Euler buckling, eased by shear, with an effective length equal to
their own, which flatters a column free at one end. A sandwich wall given
as a section transformed to its skin is checked against the skin's
strength alone -- not core shear, skin wrinkling or the fixings that hold
it on, which are how sandwich panels usually fail.

**Plates -- the voxel solver, a layer thick.** A plate is one layer of
voxels with its loads scaled to the plate's thickness, so the stress is the
real plate's at any voxel size. Against a beam it reads about 12% low, like
the parts above, and the plates are taken to carry all of the joint's
moment, which they do not -- the tube-to-tube weld carries some too. How
much they take off that weld is shared by stiffness: an estimate, and the
number to watch for fatigue. Plates are in plane only: buckling of a long
free edge, and the welds' own strength, are not checked.

**Aero -- an engineering estimate, not CFD.** `DragModel` scores a mesh in
about a millisecond from the few facts that decide most of the drag of a
bluff body at highway speed: faces into the wind are pushed on (Newtonian),
convex shoulders get suction, a surface falling away steeper than about 20
degrees sheds a wake, long thin sections shed a stronger two-dimensional one,
friction scales with wetted area, and whatever sits in another body's wake --
or an obstacle's -- is sheltered. Calibrated against high-Reynolds-number
textbook values:

| shape | model Cd | reference |
|---|---|---|
| flat plate, face on | 1.29 | 1.17-1.28 |
| plate 20x as long as wide | 1.53 | 1.5 |
| endless plate (2D) | 2.01 | 1.98 |
| cube | 1.09 | 1.05 |
| sphere | 0.19 | 0.2 (above drag crisis) |
| streamlined body, L/D 4 | 0.11 | 0.05-0.1 |

Trust its rankings -- rounded nose over square, gentle taper over steep, the
order of the designs in a sweep. Treat its absolute numbers as good to perhaps
30%, and optimistic for anything sitting in a car's own disturbed airflow,
which it does not see. It cannot see vortex shedding, lift, or gaps between
bodies, and its answers for air arriving at an angle are not to be trusted --
which is why crosswind loads are built from two head-on cases instead. When
those matter, take the two or three designs it picked to a real CFD run; that
is what it is for.

It reads curvature over a few rings of neighbouring triangles, so it wants
them roughly even. A CAD export is not, which is why both the flow study and
the parametric one weld it and split its long edges first. Across a big fillet, halving the
triangle size can still move the score by a few percent, so treat two radii
that score within 5% of each other as a tie.

## Why evolutionary

Two different kinds, for two different reasons.

**BESO for structure** (`whittle.beso`). Every FEA solve says how hard every
voxel is working, so the search can remove the idlest few percent, let
material regrow beside the busiest, and repeat -- about a hundred solves for a
whole sweep, where a population search over the same design would need tens
of thousands. Its history *is* the slider.

**Differential evolution for everything else** (`whittle.evolve`). Where the
objective is a black box -- the drag model, a meshing step, a printability
check -- or the constraints are conditionals rather than smooth penalties, a
population method is the right tool. The flow study runs one search per
station, each seeded around the last, which keeps neighbouring stations
alike. The `optimizer` seam is where [EML](https://github.com/greentheo/EML)
can replace scipy's engine without touching anything else.

## Status

**Works, tested (188 tests, ~61 s), runs today**

- `FrameStudy` -- size the tubes of a welded frame: stress, buckling,
  deflection and what the supports hold, from a catalog of real tube
- `PlateStudy` and `FrameStudy.gusset()` -- gusset plates at a frame's
  joints, lightened and written out as DXF and SVG for a plasma table
- `ParametricStudy` -- tune a parametric model's own dimensions for drag,
  CadQuery or otherwise, with the same slider
- Obstacles -- fixed bodies a part rides in the wake of, like the car in
  front of a hitch box
- The web GUI -- load, click to place, run, slide, download; strength,
  airflow, or both
- `DualStudy`, `DualSweep` -- carve for drag and lighten for load together,
  with wind loads from each shape and adaptive refinement
- Load cases and distributed loads in the FEA (`Load(case=...)`, `FieldLoad`)
- `LoadStudy`, `FlowStudy`, `Sweep` -- the part-in, slider-out workflow;
  a flow study carves the part as sent, so its first station is the part
- `VoxelFEA` -- linear-elastic hex FEA, matrix-free CG, stress and safety
  factor, floating material dropped, broken load paths reported
- `beso` -- bi-directional evolutionary structural optimization with stations
- `DragModel` -- the drag estimate above, with a per-face breakdown of where
  the drag comes from
- `whittle(...)` on every study -- choose the surfaces material comes from
- `Grid` -- voxelizing any closed STL, measuring how much of each voxel it
  fills, selecting regions by point, face or box, and meshing results so
  their outside lands where the part's was
- `Compliance` -- the FEA as an objective for population search, replacing
  the geometric proxy `AnchorConnectivity` (which stays, as a cheap pre-filter)
- `FFDLattice` -- now with frozen control points, mirror symmetry, inward-only
  carving
- OBJ and STL in and out

**Not yet**

- Speed on fine grids. Jacobi-preconditioned CG needs hundreds of iterations
  on a slender part; a structural sweep is about 30 s at the default
  resolution (48 voxels along the longest side) and 86 s at 60. Multigrid is
  the fix.
- A real flow solver to check the drag model against.

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

## Examples

Times measured on one core of an M4 laptop. Everything is written to `out/`.

| | what it shows | time |
|---|---|---|
| `01_bridge.py` | population search on a density field, with the geometric proxy -- and the thread-thin result that proxy produces | 10 s |
| `02_shelf_bracket.py` | `LoadStudy`: a solid bracket whittled to a truss, table above | 31 s |
| `03_roof_box.py` | `FlowStudy`: a square 2 m roof box carved around a protected cargo block | 91 s |
| `04_roof_bar.py` | `FlowStudy(extruded=...)`: a rectangular rack bar reshaped, as a 2D section | 42 s |
| `05_dual_roof_box.py` | `DualStudy`: a 2 m foam pod on two bars with 50 kg of cargo, carved and lightened together | 129 s |
| `06_parametric_hitch_box.py` | `ParametricStudy`: five dimensions of a lofted hitch box, tuned in free air and behind the car (`obstacles=`) | 200 s |
| `07_hitch_frame.py` | `FrameStudy`: the steel tube frame of a hitch cargo platform, sized for bumps | 1 s |
| `08_hitch_gussets.py` | `FrameStudy.gusset()`: 3/16" plates for both corners of that frame's riser, as DXF and SVG | 8 s |

The roof box, for a taste of the flow side:

```
removed  drag     Cd  CdA m2  N @110km/h
-------  ----  -----  ------  ----------
     0%  100%  1.071  0.3856       220.5
     5%   83%  0.896  0.3189       182.4
     9%   59%  0.644  0.2282       130.5
    15%   34%  0.370  0.1293        73.9
    20%   18%  0.206  0.0705        40.3
    25%   16%  0.184  0.0620        35.4

80% of the available drag saving by 20% removed
```

The nose rounds, then slopes, then the box narrows towards the front in plan --
the shape commercial roof boxes converge on -- while the cargo block stays
whole. The six-fold drag reduction at the end is the model's free-air
optimism; the ordering and the knee of the curve are the parts to act on.
The square box starts at Cd 1.07 because it is scored as the sharp box it
is. Earlier versions carved a voxel copy whose edges came back rounded, which
started it at 0.80 -- the carved end lands at the same 35 N either way.

## Where this is going

`ROADMAP.md`. The short version: multigrid to make finer grids fast, then a
real flow solve to check the drag model's picks against.

---

A tool that tells you how much material you saved, without having checked the
part still holds, is not a tool. It is a flattering picture. This one checks --
approximately, and says how approximately.
