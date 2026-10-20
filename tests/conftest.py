from __future__ import annotations

import numpy as np
import pytest

from whittle.types import Mesh


def _unit_cube() -> Mesh:
    """A closed, outward-wound unit cube. Signed volume should be exactly +1."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom, -z
            [4, 5, 6], [4, 6, 7],  # top, +z
            [0, 1, 5], [0, 5, 4],  # front, -y
            [3, 7, 6], [3, 6, 2],  # back, +y
            [0, 4, 7], [0, 7, 3],  # left, -x
            [1, 2, 6], [1, 6, 5],  # right, +x
        ],
        dtype=np.int64,
    )
    return Mesh(vertices, faces)


@pytest.fixture
def unit_cube() -> Mesh:
    return _unit_cube()
