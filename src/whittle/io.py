"""OBJ and STL, read and written directly.

Deliberately not a mesh library. We need to get triangles in and triangles out,
and a dependency that can do everything costs more than the eighty lines it
would save. If you need repair, boolean ops or robust loading of someone else's
broken export, run it through a real mesh tool before it gets here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .types import FloatArray, IntArray, Mesh


def load(path: str | Path) -> Mesh:
    """Load a mesh by extension. Supports .obj, and binary or ASCII .stl."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix == ".stl":
        return load_stl(path)
    raise ValueError(f"unsupported mesh format {suffix!r} (expected .obj or .stl)")


def save(mesh: Mesh, path: str | Path) -> None:
    """Save a mesh by extension. STL is written binary."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        save_obj(mesh, path)
    elif suffix == ".stl":
        save_stl(mesh, path)
    else:
        raise ValueError(f"unsupported mesh format {suffix!r} (expected .obj or .stl)")


def load_obj(path: str | Path) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                x, y, z = (float(v) for v in line.split()[1:4])
                vertices.append((x, y, z))
            elif line.startswith("f "):
                # OBJ indices are 1-based and may carry texture/normal fields.
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                # Fan-triangulate anything with more than three corners.
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return Mesh(
        np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )


def save_obj(mesh: Mesh, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# written by whittle\n")
        for v in mesh.vertices:
            handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in mesh.faces + 1:
            handle.write(f"f {f[0]} {f[1]} {f[2]}\n")


def load_stl(path: str | Path) -> Mesh:
    raw = Path(path).read_bytes()
    # An ASCII STL starts with "solid", but so do some binary ones written by
    # careless exporters. The length check is the reliable discriminator.
    if raw[:5] == b"solid" and len(raw) >= 84:
        n_tri = struct.unpack("<I", raw[80:84])[0]
        if len(raw) != 84 + n_tri * 50:
            return _load_stl_ascii(raw)
    elif raw[:5] == b"solid":
        return _load_stl_ascii(raw)
    return _load_stl_binary(raw)


def _load_stl_binary(raw: bytes) -> Mesh:
    n_tri = struct.unpack("<I", raw[80:84])[0]
    record = np.dtype(
        [("normal", "<3f4"), ("corners", "<3,3f4"), ("attr", "<u2")]
    )
    data = np.frombuffer(raw, dtype=record, count=n_tri, offset=84)
    corners = np.asarray(data["corners"], dtype=np.float64).reshape(-1, 3)
    return _weld(corners)


def _load_stl_ascii(raw: bytes) -> Mesh:
    corners: list[tuple[float, float, float]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex":
            corners.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return _weld(np.asarray(corners, dtype=np.float64).reshape(-1, 3))


def _weld(corners: FloatArray, decimals: int = 9) -> Mesh:
    """Collapse per-triangle corners into shared vertices.

    STL stores every triangle's corners independently, so a loaded mesh has
    three times the vertices it needs and no connectivity at all. Rounding
    before the unique pass is what makes coincident corners actually compare
    equal across triangles.
    """
    if not len(corners):
        return Mesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64))
    _, index, inverse = np.unique(
        corners.round(decimals), axis=0, return_index=True, return_inverse=True
    )
    vertices = corners[index]
    faces: IntArray = inverse.reshape(-1, 3).astype(np.int64)
    return Mesh(vertices, faces)


def save_stl(mesh: Mesh, path: str | Path) -> None:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.dtype([("normal", "<3f4"), ("corners", "<3,3f4"), ("attr", "<u2")])
    data = np.zeros(len(mesh.faces), dtype=record)
    data["normal"] = normals
    data["corners"] = np.stack([a, b, c], axis=1)
    with open(path, "wb") as handle:
        handle.write(b"whittle" + b"\0" * 73)
        handle.write(struct.pack("<I", len(mesh.faces)))
        handle.write(data.tobytes())
