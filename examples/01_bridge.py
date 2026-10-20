"""Connect three pads with as little material as possible.

Run:  python examples/01_bridge.py

Read the result for what it is. The objective here is material plus a
*geometric* connectivity proxy, not stiffness, so the optimizer is rewarded for
finding the thinnest thread that technically joins the pads. That is the
correct answer to the question actually being asked, and the wrong answer to
the question you care about -- which is exactly the gap ROADMAP.md item 1
closes. See it for yourself before trusting any number this library prints.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from whittle import (
    AnchorConnectivity,
    DensityField,
    MaterialFraction,
    WeightedSum,
    evolve,
    io,
)

SHAPE = (32, 6, 16)
CONTROL = (8, 2, 4)
OUT = Path(__file__).resolve().parent.parent / "out"


def pads(shape: tuple[int, int, int]) -> list[np.ndarray]:
    """Two feet on the ground and one load pad overhead."""
    nx, ny, nz = shape
    left = np.zeros(shape, dtype=bool)
    left[1:4, :, 0:2] = True
    right = np.zeros(shape, dtype=bool)
    right[nx - 4 : nx - 1, :, 0:2] = True
    top = np.zeros(shape, dtype=bool)
    top[nx // 2 - 2 : nx // 2 + 2, :, nz - 2 : nz] = True
    return [left, right, top]


def main() -> None:
    anchors = pads(SHAPE)
    keep = anchors[0] | anchors[1] | anchors[2]
    backend = DensityField(SHAPE, CONTROL, keep=keep, spacing=1.0)

    objective = WeightedSum(
        {
            MaterialFraction(): 1.0,
            AnchorConnectivity(anchors): 4.0,
        }
    )

    print(f"design box {SHAPE}, {backend.n_params} free parameters")
    print(f"objective:  {objective.name}\n")

    started = time.perf_counter()
    result = evolve(
        backend,
        objective,
        seed=0,
        maxiter=25,
        popsize=6,
        callback=lambda gen, value: print(f"  gen {gen:3d}  {value:.5f}"),
    )
    elapsed = time.perf_counter() - started

    print(f"\n{result}")
    print(f"elapsed: {elapsed:.1f}s")

    mesh = result.mesh
    print(f"mesh:    {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    print(f"volume:  {mesh.volume:.1f} of {backend.design_volume:.0f} in the design box")

    OUT.mkdir(exist_ok=True)
    io.save(mesh, OUT / "bridge.stl")
    io.save(mesh, OUT / "bridge.obj")
    print(f"written: {OUT / 'bridge.stl'}")


if __name__ == "__main__":
    main()
