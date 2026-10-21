"""Simple closed meshes: for tests, examples, and quick what-ifs.

Tessellated finely and evenly on purpose. A deformation lattice moves vertices,
not faces, so a box made of twelve triangles cannot bend at all; and the drag
model reads curvature from neighbouring faces, so it wants many of them.
"""

from __future__ import annotations

import numpy as np

from .io import weld
from .types import FloatArray, IntArray, Mesh, Vector3


def box(size: Vector3, *, center: Vector3 = (0.0, 0.0, 0.0), cell: float | None = None) -> Mesh:
    """An axis-aligned box, its faces gridded into triangles about ``cell`` across."""
    dims = np.asarray(size, dtype=np.float64)
    if np.any(dims <= 0):
        raise ValueError(f"size must be positive, got {size}")
    cell = float(dims.min()) / 4.0 if cell is None else float(cell)
    counts = np.maximum(np.round(dims / cell).astype(np.int64), 1)
    half = dims / 2.0
    corners: list[FloatArray] = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        gu = np.linspace(-half[u], half[u], counts[u] + 1)
        gv = np.linspace(-half[v], half[v], counts[v] + 1)
        mu, mv = np.meshgrid(gu, gv, indexing="ij")
        for side in (-1.0, 1.0):
            points = np.zeros((*mu.shape, 3), dtype=np.float64)
            points[..., axis] = side * half[axis]
            points[..., u] = mu
            points[..., v] = mv
            corners.append(_grid_triangles(points))
    mesh = weld(np.concatenate(corners).reshape(-1, 3))
    return _outward(_translate(mesh, center), center)


def ellipsoid(
    radii: Vector3, *, center: Vector3 = (0.0, 0.0, 0.0), subdivisions: int = 3
) -> Mesh:
    """An ellipsoid, as a scaled icosphere. Three subdivisions is 1280 triangles."""
    vertices, faces = _icosphere(subdivisions)
    scaled = vertices * np.asarray(radii, dtype=np.float64)
    return _outward(_translate(Mesh(scaled, faces), center), center)


def sphere(radius: float, *, center: Vector3 = (0.0, 0.0, 0.0), subdivisions: int = 3) -> Mesh:
    return ellipsoid((radius, radius, radius), center=center, subdivisions=subdivisions)


def _grid_triangles(points: FloatArray) -> FloatArray:
    """Two triangles per cell of a grid of points, as stacked corner triples."""
    p00 = points[:-1, :-1].reshape(-1, 3)
    p10 = points[1:, :-1].reshape(-1, 3)
    p01 = points[:-1, 1:].reshape(-1, 3)
    p11 = points[1:, 1:].reshape(-1, 3)
    first = np.stack([p00, p10, p11], axis=1)
    second = np.stack([p00, p11, p01], axis=1)
    return np.concatenate([first, second])


def _icosphere(subdivisions: int) -> tuple[FloatArray, IntArray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(subdivisions):
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        corners = np.concatenate(
            [
                np.stack([a, ab, ca], 1),
                np.stack([ab, b, bc], 1),
                np.stack([ca, bc, c], 1),
                np.stack([ab, bc, ca], 1),
            ]
        )
        mesh = weld(corners.reshape(-1, 3))
        vertices = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        faces = mesh.faces
    return vertices, faces


def _translate(mesh: Mesh, offset: Vector3) -> Mesh:
    return Mesh(mesh.vertices + np.asarray(offset, dtype=np.float64), mesh.faces)


def _outward(mesh: Mesh, center: Vector3) -> Mesh:
    """Wind every face of a convex mesh so its normal points away from ``center``."""
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    outward = np.einsum("ij,ij->i", normal, (a + b + c) / 3.0 - np.asarray(center)) >= 0
    faces = np.where(outward[:, None], mesh.faces, mesh.faces[:, ::-1])
    return Mesh(mesh.vertices, np.ascontiguousarray(faces))
