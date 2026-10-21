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
- **Run**, then **slide**. Each station shows its mesh with a heatmap --
  stress for strength, drag per area for airflow -- its numbers, and where it
  sits on the trade-off curve. Download any station as an STL.

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
     0%    40.2       100%       0.177         4.3    9.4x
    10%    36.2        98%       0.179         4.3    9.4x
    20%    32.2        95%       0.184         4.3    9.3x
    30%    28.2        89%       0.193         4.4    9.1x
    40%    24.1        80%       0.210         4.9    8.1x
    50%    20.1        68%       0.242         6.4    6.3x
    60%    16.1        53%       0.308         8.4    4.8x
    70%    12.1        36%       0.439        13.3    3.0x
    80%     8.0        19%       0.850        24.1    1.7x
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
| physics | linear-elastic FEA on a voxel grid | engineering drag model on the surface mesh |
| search | BESO: remove the voxels doing the least work | differential evolution over a deformation lattice |
| geometry | can open holes, grow trusses | stays one smooth closed surface, never grows |
| each station reports | mass, stiffness kept, deflection, stress, safety factor | drag vs original, Cd, CdA, newtons at speed |

The split is the old one this library started from. Lightening a part wants a
representation that can open a hole -- no smooth deformation of a solid block
will ever find a truss. Shaping against a flow wants the opposite: a surface
that stays closed and smooth, because drag is decided by its curvature.

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

**Aero -- an engineering estimate, not CFD.** `DragModel` scores a mesh in
about a millisecond from the few facts that decide most of the drag of a
bluff body at highway speed: faces into the wind are pushed on (Newtonian),
convex shoulders get suction, a surface falling away steeper than about 20
degrees sheds a wake, long thin sections shed a stronger two-dimensional one,
friction scales with wetted area, and whatever sits in another body's wake is
sheltered. Calibrated against high-Reynolds-number textbook values:

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
which it does not see. It also cannot see crosswind, vortex shedding, lift, or
gaps between bodies. When those matter, take the two or three designs it
picked to a real CFD run; that is what it is for.

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

**Works, tested (124 tests, ~19 s), runs today**

- The web GUI -- load, click to place, run, slide, download
- `LoadStudy`, `FlowStudy`, `Sweep` -- the part-in, slider-out workflow
- `VoxelFEA` -- linear-elastic hex FEA, matrix-free CG, stress and safety
  factor, floating material dropped, broken load paths reported
- `beso` -- bi-directional evolutionary structural optimization with stations
- `DragModel` -- the drag estimate above, with a per-face breakdown of where
  the drag comes from
- `Grid` -- voxelizing any closed STL, selecting regions by point, face or
  box, meshing results with smoothing that keeps volume
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
| `03_roof_box.py` | `FlowStudy`: a square 2 m roof box carved around a protected cargo block | 84 s |
| `04_roof_bar.py` | `FlowStudy(extruded=...)`: a rectangular rack bar reshaped, as a 2D section | 35 s |

The roof box, for a taste of the flow side:

```
removed  drag     Cd  CdA m2  N @110km/h
-------  ----  -----  ------  ----------
     0%  100%  0.796  0.2928       167.4
     5%   81%  0.657  0.2379       136.1
     9%   59%  0.484  0.1721        98.4
    15%   32%  0.262  0.0938        53.6
    20%   23%  0.197  0.0687        39.3
    25%   21%  0.190  0.0609        34.8

80% of the available drag saving by 15% removed
```

The nose rounds, then slopes, then the box narrows towards the front in plan --
the shape commercial roof boxes converge on -- while the cargo block stays
whole. The 5x drag reduction at the end is the model's free-air optimism; the
ordering and the knee of the curve are the parts to act on.

## Where this is going

`ROADMAP.md`. The short version: multigrid to make finer grids fast, then a
real flow solve to check the drag model's picks against.

---

A tool that tells you how much material you saved, without having checked the
part still holds, is not a tool. It is a flattering picture. This one checks --
approximately, and says how approximately.
