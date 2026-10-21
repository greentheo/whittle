from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from whittle import io, shapes
from whittle.gui.server import App, BadRequest, serve


def _wait(app: App, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = app.status(job_id)
        if status["state"] not in ("queued", "running"):
            return status
        time.sleep(0.05)
    raise AssertionError("study did not finish in time")


def _floats(payload: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload), dtype=np.float32)


@pytest.fixture(scope="module")
def bracket_run() -> tuple[App, dict[str, Any]]:
    app = App()
    part = app.example("shelf-bracket")
    setup = dict(part["setup"])
    setup.update(part=part["id"], resolution=32, stations=3, removeUpTo=0.5)
    status = _wait(app, app.start(setup)["id"])
    return app, status


def test_an_example_arrives_with_a_runnable_setup() -> None:
    app = App()
    part = app.example("shelf-bracket")
    assert part["closed"]
    assert part["triangles"] > 0
    assert part["setup"]["kind"] == "load"
    assert np.allclose(part["max"], [80.0, 8.0, 80.0])


def test_a_load_study_runs_and_reports_every_station(
    bracket_run: tuple[App, dict[str, Any]],
) -> None:
    _, status = bracket_run
    assert status["state"] == "done", status["error"]
    assert status["progress"] == 1.0
    result = status["result"]
    assert [s["removed"] for s in result["stations"]][0] == 0.0
    assert len(result["stations"]) == 3
    assert result["heat"]["unit"] == "MPa"
    assert result["strength"] == 40.0
    keys = {c["key"] for c in result["columns"]}
    assert {"stiffness", "safety_factor"} <= keys


def test_station_meshes_carry_a_stress_heatmap(bracket_run: tuple[App, dict[str, Any]]) -> None:
    app, status = bracket_run
    mesh = app.station_mesh(status["id"], 2)
    positions = _floats(mesh["positions"]).reshape(-1, 3)
    heat = _floats(mesh["heat"])
    assert len(heat) == len(positions)
    assert heat.max() > 0
    # The scale the page colours by spans the whole sweep.
    assert heat.max() <= 2.0 * status["result"]["heatMax"]


def test_a_station_downloads_as_a_readable_stl(bracket_run: tuple[App, dict[str, Any]]) -> None:
    app, status = bracket_run
    data, filename = app.station_stl(status["id"], 1)
    assert filename.startswith("Shelf_bracket-") and filename.endswith("pct.stl")
    mesh = io.parse(data, ".stl")
    assert mesh.volume > 0


def test_a_flow_study_runs_with_a_signed_drag_heatmap() -> None:
    app = App()
    part = app.example("roof-bar")
    setup = dict(part["setup"])
    setup.update(part=part["id"], resolution=20, stations=2, effort=0.2)
    status = _wait(app, app.start(setup)["id"])
    assert status["state"] == "done", status["error"]
    assert status["result"]["heat"]["signed"] is True
    heat = _floats(app.station_mesh(status["id"], 0)["heat"])
    assert heat.max() > 0


def test_previewing_a_placement_returns_the_voxels_it_selects() -> None:
    app = App()
    part = app.example("shelf-bracket")
    where = {"part": part["id"], "resolution": 32, "at": [0, 4, 40]}
    face = app.preview({**where, "normal": [-1, 0, 0]})
    point = app.preview({**where, "radius": 4.0})
    assert face["count"] > point["count"] > 0
    centres = _floats(face["points"]).reshape(-1, 3)
    # The whole wall face, and nothing deeper than a couple of voxels into the part.
    assert centres[:, 0].max() < 3 * face["spacing"]
    assert centres[:, 2].max() - centres[:, 2].min() > 60.0


def test_uploads_are_scaled_by_their_units() -> None:
    app = App()
    data = io.stl_bytes(shapes.box((2.0, 3.0, 4.0), cell=1.0))
    part = app.upload(data, "block.stl", units="cm")
    assert np.allclose(np.subtract(part["max"], part["min"]), [20.0, 30.0, 40.0])
    with pytest.raises(BadRequest, match=r"\.stl or \.obj"):
        app.upload(data, "block.step")
    with pytest.raises(BadRequest, match=r"units"):
        app.upload(data, "block.stl", units="furlongs")
    with pytest.raises(BadRequest, match=r"could not read|no triangles"):
        app.upload(b"solid nothing\nendsolid\n", "empty.stl")


def test_an_open_mesh_is_flagged() -> None:
    box = shapes.box((10.0, 10.0, 10.0), cell=5.0)
    open_box = type(box)(box.vertices, box.faces[2:])
    assert App().upload(io.stl_bytes(open_box), "open.stl")["closed"] is False


def test_bad_studies_are_refused_with_a_reason() -> None:
    app = App()
    part = app.example("shelf-bracket")
    with pytest.raises(BadRequest, match=r"support and one load"):
        app.start({"kind": "load", "part": part["id"], "supports": [], "loads": []})
    with pytest.raises(BadRequest, match=r"no longer loaded"):
        app.start({"kind": "flow", "part": "nope"})
    with pytest.raises(BadRequest, match=r"kind"):
        app.start({"kind": "magic", "part": part["id"]})
    status = _wait(app, app.start({"kind": "flow", "part": part["id"], "flow": [0, 0, 0]})["id"])
    assert status["state"] == "failed" and "non-zero" in status["error"]


def test_a_running_study_can_be_cancelled() -> None:
    app = App()
    part = app.example("shelf-bracket")
    setup = dict(part["setup"])
    setup.update(part=part["id"], resolution=64)
    job = app.start(setup)
    app.cancel(job["id"])
    assert _wait(app, job["id"])["state"] == "cancelled"


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = serve(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _post(url: str, body: bytes) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_and_its_assets_are_served(server: str) -> None:
    status, page = _get(server + "/")
    assert status == 200 and b"<title>whittle</title>" in page
    for asset in ("app.js", "app.css"):
        assert _get(f"{server}/static/{asset}")[0] == 200


def test_files_outside_the_static_folder_are_not(server: str) -> None:
    assert _get(server + "/static/..%2Fserver.py")[0] == 404
    assert _get(server + "/static/missing.js")[0] == 404
    assert _get(server + "/nowhere")[0] == 404


def test_the_api_over_http(server: str) -> None:
    status, info = _get(server + "/api/info")
    assert status == 200 and "pla" in json.loads(info)["materials"]
    data = io.stl_bytes(shapes.box((10.0, 10.0, 10.0), cell=2.0))
    status, part = _post(server + "/api/parts?name=cube.stl&units=mm", data)
    assert status == 200 and part["closed"]
    status, error = _post(server + "/api/jobs", b"{not json")
    assert status == 400 and "JSON" in error["error"]
    status, error = _post(server + "/api/examples/unicorn", b"")
    assert status == 400 and "unicorn" in error["error"]
