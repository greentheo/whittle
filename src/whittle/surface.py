"""Getting a mesh someone else made into a shape the drag model can read.

CAD exports and slicer-bound STLs are tessellated for display: every vertex
repeated along the seams between faces, big flat triangles beside thin strips
along fillets. The drag model reads curvature and shelter from neighbouring
triangles, so it wants the opposite -- one connected surface, triangles of
roughly even size. These turn the first into the second without moving the
surface at all.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from .types import IntArray, Mesh


def weld(mesh: Mesh, tolerance: float) -> Mesh:
    """Merge vertices closer than ``tolerance``, and drop the triangles that collapses.

    :func:`whittle.io.weld` rounds coordinates, which merges exact copies. A
    CAD tessellation's copies along a seam can differ in the last few digits,
    because each face computes its own; this finds them by distance instead.
    """
    if not len(mesh.vertices):
        return mesh
    pairs = cKDTree(mesh.vertices).query_pairs(tolerance, output_type="ndarray")
    n = len(mesh.vertices)
    if len(pairs):
        graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = csgraph.connected_components(graph, directed=False)
        label = np.asarray(labels, dtype=np.int64)
    else:
        label = np.arange(n, dtype=np.int64)
    _, first = np.unique(label, return_index=True)
    faces = label[mesh.faces]
    whole = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
    whole &= faces[:, 2] != faces[:, 0]
    return Mesh(mesh.vertices[first], faces[whole])


def split_long_edges(mesh: Mesh, max_edge: float, *, rounds: int = 64) -> Mesh:
    """Halve edges longer than ``max_edge`` until none is. The surface does not move.

    Each round halves the longest edge of every face that has one too long --
    longest-edge bisection, which keeps triangles from getting thinner as they
    get smaller -- and re-triangulates each face by how many of its edges were
    split that round, one, two or all three, so faces either side of a split
    edge always agree and the mesh stays closed.
    """
    vertices, faces = mesh.vertices, mesh.faces
    for _ in range(rounds):
        if not len(faces):
            break
        sides = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
        ends = np.sort(sides.reshape(-1, 2), axis=1)
        key = ends[:, 0] * len(vertices) + ends[:, 1]
        unique, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        a, b = ends[first, 0], ends[first, 1]
        length = np.linalg.norm(vertices[a] - vertices[b], axis=1)
        each = length[inverse.reshape(-1, 3)]
        longest = inverse.reshape(-1, 3)[np.arange(len(faces)), np.argmax(each, axis=1)]
        long = np.zeros(len(unique), dtype=bool)
        long[longest] = True
        long &= length > max_edge
        if not long.any():
            break
        middle = np.full(len(unique), -1, dtype=np.int64)
        middle[long] = len(vertices) + np.arange(int(long.sum()))
        vertices = np.concatenate([vertices, 0.5 * (vertices[a[long]] + vertices[b[long]])])
        # mids[:, k] is the new vertex halfway along side k, corner k to k + 1.
        mids = middle[inverse.reshape(-1)].reshape(-1, 3)
        split = mids >= 0
        count = split.sum(axis=1)
        pieces = [faces[count == 0]]

        one = count == 1
        turn = np.argmax(split[one], axis=1)
        c, m = _turn(faces[one], turn), _turn(mids[one], turn)
        pieces.append(np.stack([c[:, 0], m[:, 0], c[:, 2]], 1))
        pieces.append(np.stack([m[:, 0], c[:, 1], c[:, 2]], 1))

        two = count == 2
        turn = (np.argmin(split[two], axis=1) + 1) % 3
        c, m = _turn(faces[two], turn), _turn(mids[two], turn)
        pieces.append(np.stack([m[:, 0], c[:, 1], m[:, 1]], 1))
        # The quad left over is cut along its shorter diagonal.
        short = np.linalg.norm(vertices[c[:, 0]] - vertices[m[:, 1]], axis=1) <= np.linalg.norm(
            vertices[m[:, 0]] - vertices[c[:, 2]], axis=1
        )
        pieces.append(
            np.where(
                short[:, None],
                np.stack([c[:, 0], m[:, 0], m[:, 1]], 1),
                np.stack([c[:, 0], m[:, 0], c[:, 2]], 1),
            )
        )
        pieces.append(
            np.where(
                short[:, None],
                np.stack([c[:, 0], m[:, 1], c[:, 2]], 1),
                np.stack([m[:, 0], m[:, 1], c[:, 2]], 1),
            )
        )

        three = count == 3
        c, m = faces[three], mids[three]
        pieces += [
            np.stack([c[:, 0], m[:, 0], m[:, 2]], 1),
            np.stack([m[:, 0], c[:, 1], m[:, 1]], 1),
            np.stack([m[:, 2], m[:, 1], c[:, 2]], 1),
            np.stack([m[:, 0], m[:, 1], m[:, 2]], 1),
        ]
        faces = np.concatenate(pieces).astype(np.int64)
    return Mesh(vertices, faces)


def _turn(rows: IntArray, by: IntArray) -> IntArray:
    """Rotate each row's three entries left by its own amount."""
    index = (np.arange(3)[None, :] + by[:, None]) % 3
    result: IntArray = np.take_along_axis(rows, index, axis=1)
    return result


def open_edges(mesh: Mesh) -> int:
    """Edges with only one face on them: holes in what should be a closed surface."""
    f = mesh.faces
    ends = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(ends[:, 0] * len(mesh.vertices) + ends[:, 1], return_counts=True)
    return int((counts == 1).sum())


def is_closed(mesh: Mesh) -> bool:
    """Whether every edge has exactly two faces: no holes, no fins, no shells sharing an edge."""
    if not len(mesh.faces):
        return False
    f = mesh.faces
    ends = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(ends[:, 0] * len(mesh.vertices) + ends[:, 1], return_counts=True)
    return bool((counts == 2).all())
