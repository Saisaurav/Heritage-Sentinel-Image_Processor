"""
Test suite for Heritage Sentinel's app.py.

Run with:
    pip install pytest flask flask-cors pillow
    pytest test_app.py -v

These tests exercise every HTTP endpoint through Flask's test client and
isolate the filesystem state per test (a fresh tmp_path each time, via the
`client` fixture), so nothing touches your real raw_images/ processed/
manifests/ folders. The Hugin/Meshroom/Blender subprocess calls are
monkeypatched out in the pipeline-endpoint tests, so you do NOT need those
binaries installed to run this suite — only Flask, flask-cors, and Pillow.
"""

import io
import time

import pytest

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Point every storage directory at a fresh tmp_path and hand back a
    Flask test client. Each test gets a clean filesystem."""
    monkeypatch.setattr(app_module, "RAW_DIR", tmp_path / "raw_images" / "tours")
    monkeypatch.setattr(app_module, "PROCESSED_DIR", tmp_path / "processed" / "panoramas")
    monkeypatch.setattr(app_module, "ARTIFACT_DIR", tmp_path / "processed" / "artifacts")
    monkeypatch.setattr(app_module, "RAW_ARTIFACT_IMAGES_DIR", tmp_path / "raw_images" / "artifacts")
    monkeypatch.setattr(app_module, "MESHROOM_CACHE_DIR", tmp_path / "processed" / "photogrammetry_cache")
    monkeypatch.setattr(app_module, "MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr(app_module, "STATUS_DIR", tmp_path / "data" / "status")

    for directory in [
        app_module.RAW_DIR,
        app_module.PROCESSED_DIR,
        app_module.ARTIFACT_DIR,
        app_module.RAW_ARTIFACT_IMAGES_DIR,
        app_module.MESHROOM_CACHE_DIR,
        app_module.MANIFESTS_DIR,
        app_module.STATUS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def wait_for_status(client, kind, tour_id, item_id, target, timeout=5):
    """Poll a status endpoint until it reaches `target` or times out.
    Used for the background-thread pipeline endpoints."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/status/{tour_id}/{kind}/{item_id}")
        last = r.get_json()
        if last["status"] == target:
            return last
        time.sleep(0.05)
    raise AssertionError(f"status never reached {target!r}, last was {last!r}")


def _seed_waypoint(tour_id, waypoint_id, **overrides):
    app_module.ensure_tour_manifest(tour_id)
    waypoints = app_module.load_json(app_module.waypoints_manifest_path(tour_id), {})
    entry = {
        "id": waypoint_id, "image": "", "mapPos": [0, 0], "links": [],
        "isArtifact": False,
    }
    entry.update(overrides)
    waypoints[waypoint_id] = entry
    app_module.save_json(app_module.waypoints_manifest_path(tour_id), waypoints)


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.get_json()["status"] == "running"


# ---------------------------------------------------------------------------
# parse_waypoint_metadata
# ---------------------------------------------------------------------------

def test_parse_waypoint_metadata_plain():
    meta = app_module.parse_waypoint_metadata("waypoint_001")
    assert meta == {"clean_id": "waypoint_001", "is_artifact": False, "artifact_names": []}


def test_parse_waypoint_metadata_artifact():
    meta = app_module.parse_waypoint_metadata("waypoint_002.artifact.roman_bust")
    assert meta["clean_id"] == "waypoint_002"
    assert meta["is_artifact"] is True
    assert meta["artifact_names"] == ["roman_bust"]


# ---------------------------------------------------------------------------
# load_json / save_json
# ---------------------------------------------------------------------------

def test_load_json_missing_returns_default(tmp_path):
    assert app_module.load_json(tmp_path / "nope.json", {"x": 1}) == {"x": 1}


def test_save_then_load_json(tmp_path):
    path = tmp_path / "nested" / "thing.json"
    app_module.save_json(path, {"hello": "world"})
    assert path.exists()
    assert app_module.load_json(path, None) == {"hello": "world"}


def test_load_json_corrupt_file_returns_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    assert app_module.load_json(path, "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Tour manifest endpoint + sync_tour_manifest
# ---------------------------------------------------------------------------

def test_get_tour_not_found(client):
    r = client.get("/api/tours/ghost_tour")
    assert r.status_code == 404
    assert r.get_json() == {"error": "Tour not found"}


def test_sync_tour_manifest_creates_files(client):
    tour_id = "tour_002"

    # Simulate two waypoints already stitched: a plain one and an artifact one
    (app_module.RAW_DIR / tour_id / "waypoint_001").mkdir(parents=True)
    (app_module.RAW_DIR / tour_id / "waypoint_002.artifact.roman_bust").mkdir(parents=True)
    pano_dir = app_module.PROCESSED_DIR / tour_id
    pano_dir.mkdir(parents=True)
    (pano_dir / "waypoint_001.jpg").write_bytes(b"fake")
    (pano_dir / "waypoint_002.jpg").write_bytes(b"fake")

    result = app_module.sync_tour_manifest(tour_id)

    assert result is not None
    assert app_module.tour_manifest_path(tour_id).exists()
    assert app_module.waypoints_manifest_path(tour_id).exists()

    waypoints = result["waypoints"]
    assert waypoints["waypoint_001"]["isArtifact"] is False
    assert waypoints["waypoint_002"]["isArtifact"] is True
    assert waypoints["waypoint_002"]["artifactIds"] == ["roman_bust"]
    assert waypoints["waypoint_002"]["artifactMarkers"] == {"roman_bust": {"yaw": 0, "pitch": 0}}
    assert "waypoint_001" in result["tour"]["waypoints"]
    assert "waypoint_002" in result["tour"]["waypoints"]


def test_sync_tour_manifest_no_raw_dir_returns_none(client):
    assert app_module.sync_tour_manifest("does_not_exist") is None


def test_sync_tour_manifest_preserves_dashboard_edits(client):
    tour_id = "tour_002"
    (app_module.RAW_DIR / tour_id / "waypoint_001").mkdir(parents=True)
    pano_dir = app_module.PROCESSED_DIR / tour_id
    pano_dir.mkdir(parents=True)
    (pano_dir / "waypoint_001.jpg").write_bytes(b"fake")

    app_module.sync_tour_manifest(tour_id)

    # Simulate a dashboard edit to links/mapPos
    waypoints = app_module.load_json(app_module.waypoints_manifest_path(tour_id), {})
    waypoints["waypoint_001"]["links"] = [{"target": "waypoint_002", "yaw": 90}]
    waypoints["waypoint_001"]["mapPos"] = [12, 34]
    app_module.save_json(app_module.waypoints_manifest_path(tour_id), waypoints)

    # Re-syncing (e.g. after reprocessing the panorama) must not clobber those
    app_module.sync_tour_manifest(tour_id)

    waypoints = app_module.load_json(app_module.waypoints_manifest_path(tour_id), {})
    assert waypoints["waypoint_001"]["links"] == [{"target": "waypoint_002", "yaw": 90}]
    assert waypoints["waypoint_001"]["mapPos"] == [12, 34]


def test_get_tour_merges_only_referenced_artifacts(client):
    tour_id = "tour_002"
    app_module.ensure_tour_manifest(tour_id)
    app_module.save_json(app_module.waypoints_manifest_path(tour_id), {
        "waypoint_001": {
            "id": "waypoint_001", "image": "", "mapPos": [0, 0], "links": [],
            "isArtifact": True, "artifactIds": ["roman_bust"],
            "artifactMarkers": {"roman_bust": {"yaw": 0, "pitch": 0}},
        }
    })
    app_module.save_json(app_module.artifacts_db_path(), {
        "roman_bust": {"id": "roman_bust", "name": "Roman Bust"},
        "unrelated_artifact": {"id": "unrelated_artifact", "name": "Not in this tour"},
    })

    r = client.get(f"/api/tours/{tour_id}")
    body = r.get_json()

    assert r.status_code == 200
    assert "roman_bust" in body["artifacts"]
    assert "unrelated_artifact" not in body["artifacts"]


def test_ensure_tour_manifest_does_not_set_invalid_map_image(client):
    tour_id = "tour_003"
    app_module.ensure_tour_manifest(tour_id)

    tour = app_module.load_json(app_module.tour_manifest_path(tour_id), {})
    assert "map_image" not in tour


def test_ensure_tour_manifest_sets_map_image_when_file_exists(client):
    tour_id = "tour_004"
    (app_module.MANIFESTS_DIR / tour_id).mkdir(parents=True, exist_ok=True)
    (app_module.MANIFESTS_DIR / tour_id / "map.png").write_bytes(b"fake map")

    app_module.ensure_tour_manifest(tour_id)

    tour = app_module.load_json(app_module.tour_manifest_path(tour_id), {})
    assert tour["map_image"] == f"/maps/{tour_id}.png"


# ---------------------------------------------------------------------------
# Marker endpoint
# ---------------------------------------------------------------------------

def test_update_marker_success(client):
    _seed_waypoint("tour_002", "waypoint_001")
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/marker",
        json={"yaw": 74.5, "pitch": -3.2},
    )
    assert r.status_code == 200
    assert r.get_json()["artifactMarker"] == {"yaw": 74.5, "pitch": -3.2}  # legacy path for non-artifact waypoints


def test_update_marker_missing_fields(client):
    _seed_waypoint("tour_002", "waypoint_001")
    r = client.post("/api/tours/tour_002/waypoints/waypoint_001/marker", json={"yaw": 1})
    assert r.status_code == 400


def test_update_marker_non_numeric(client):
    _seed_waypoint("tour_002", "waypoint_001")
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/marker",
        json={"yaw": "not a number", "pitch": 1},
    )
    assert r.status_code == 400


def test_update_marker_unknown_waypoint(client):
    app_module.ensure_tour_manifest("tour_002")
    r = client.post(
        "/api/tours/tour_002/waypoints/ghost/marker",
        json={"yaw": 1, "pitch": 1},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Artifact-status endpoint
# ---------------------------------------------------------------------------

def test_set_waypoint_as_artifact(client):
    _seed_waypoint("tour_002", "waypoint_001")
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/artifact",
        json={"isArtifact": True, "artifactIds": ["amphora", "greek_vase"]},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["isArtifact"] is True
    assert body["artifactIds"] == ["amphora", "greek_vase"]
    assert body["artifactMarkers"] == {
        "amphora": {"yaw": 0, "pitch": 0},
        "greek_vase": {"yaw": 0, "pitch": 0},
    }


def test_set_artifact_true_without_id_fails(client):
    _seed_waypoint("tour_002", "waypoint_001")
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/artifact",
        json={"isArtifact": True},
    )
    assert r.status_code == 400


def test_unset_waypoint_artifact_clears_fields(client):
    _seed_waypoint(
        "tour_002", "waypoint_001",
        isArtifact=True, artifactIds=["amphora", "greek_vase"],
        artifactMarkers={"amphora": {"yaw": 1, "pitch": 1}, "greek_vase": {"yaw": 2, "pitch": 2}},
    )
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/artifact",
        json={"isArtifact": False},
    )
    body = r.get_json()
    assert r.status_code == 200
    assert body["isArtifact"] is False
    assert "artifactIds" not in body
    assert "artifactMarkers" not in body



def test_parse_waypoint_metadata_multi_artifact():
    meta = app_module.parse_waypoint_metadata("waypoint_003.artifact.roman_bust,greek_vase")
    assert meta["clean_id"] == "waypoint_003"
    assert meta["is_artifact"] is True
    assert meta["artifact_names"] == ["roman_bust", "greek_vase"]


def test_sync_tour_manifest_multi_artifact(client):
    tour_id = "tour_multi"
    (app_module.RAW_DIR / tour_id / "waypoint_001.artifact.roman_bust,greek_vase").mkdir(parents=True)
    pano_dir = app_module.PROCESSED_DIR / tour_id
    pano_dir.mkdir(parents=True)
    (pano_dir / "waypoint_001.jpg").write_bytes(b"fake")

    result = app_module.sync_tour_manifest(tour_id)

    wp = result["waypoints"]["waypoint_001"]
    assert wp["isArtifact"] is True
    assert wp["artifactIds"] == ["roman_bust", "greek_vase"]
    assert wp["artifactMarkers"] == {
        "roman_bust": {"yaw": 0, "pitch": 0},
        "greek_vase": {"yaw": 0, "pitch": 0},
    }


def test_sync_tour_manifest_preserves_multi_artifact_markers(client):
    tour_id = "tour_multi"
    (app_module.RAW_DIR / tour_id / "waypoint_001.artifact.roman_bust,greek_vase").mkdir(parents=True)
    pano_dir = app_module.PROCESSED_DIR / tour_id
    pano_dir.mkdir(parents=True)
    (pano_dir / "waypoint_001.jpg").write_bytes(b"fake")

    app_module.sync_tour_manifest(tour_id)

    waypoints = app_module.load_json(app_module.waypoints_manifest_path(tour_id), {})
    waypoints["waypoint_001"]["artifactMarkers"] = {
        "roman_bust": {"yaw": 45, "pitch": -5},
        "greek_vase": {"yaw": 200, "pitch": 10},
    }
    app_module.save_json(app_module.waypoints_manifest_path(tour_id), waypoints)

    result = app_module.sync_tour_manifest(tour_id)
    wp = result["waypoints"]["waypoint_001"]
    assert wp["artifactMarkers"]["roman_bust"] == {"yaw": 45, "pitch": -5}
    assert wp["artifactMarkers"]["greek_vase"] == {"yaw": 200, "pitch": 10}


def test_update_marker_multi_artifact(client):
    _seed_waypoint(
        "tour_002", "waypoint_001",
        isArtifact=True, artifactIds=["roman_bust", "greek_vase"],
        artifactMarkers={"roman_bust": {"yaw": 0, "pitch": 0}, "greek_vase": {"yaw": 0, "pitch": 0}},
    )
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/marker",
        json={"artifact_id": "roman_bust", "yaw": 45.0, "pitch": -10.0},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["artifactMarkers"]["roman_bust"] == {"yaw": 45.0, "pitch": -10.0}
    assert body["artifactMarkers"]["greek_vase"] == {"yaw": 0, "pitch": 0}


def test_update_marker_multi_artifact_requires_artifact_id(client):
    _seed_waypoint(
        "tour_002", "waypoint_001",
        isArtifact=True, artifactIds=["roman_bust", "greek_vase"],
        artifactMarkers={"roman_bust": {"yaw": 0, "pitch": 0}, "greek_vase": {"yaw": 0, "pitch": 0}},
    )
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/marker",
        json={"yaw": 45.0, "pitch": -10.0},
    )
    assert r.status_code == 400


def test_update_marker_unknown_artifact_id(client):
    _seed_waypoint(
        "tour_002", "waypoint_001",
        isArtifact=True, artifactIds=["roman_bust"],
        artifactMarkers={"roman_bust": {"yaw": 0, "pitch": 0}},
    )
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/marker",
        json={"artifact_id": "ghost_artifact", "yaw": 1, "pitch": 1},
    )
    assert r.status_code == 404


def test_set_waypoint_artifact_legacy_single_id(client):
    _seed_waypoint("tour_002", "waypoint_001")
    r = client.post(
        "/api/tours/tour_002/waypoints/waypoint_001/artifact",
        json={"isArtifact": True, "artifactId": "amphora"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["artifactIds"] == ["amphora"]
    assert "artifactMarkers" in body


def test_get_tour_merges_multiple_artifacts(client):
    tour_id = "tour_multi"
    app_module.ensure_tour_manifest(tour_id)
    app_module.save_json(app_module.waypoints_manifest_path(tour_id), {
        "waypoint_001": {
            "id": "waypoint_001", "image": "", "mapPos": [0, 0], "links": [],
            "isArtifact": True,
            "artifactIds": ["roman_bust", "greek_vase"],
            "artifactMarkers": {
                "roman_bust": {"yaw": 0, "pitch": 0},
                "greek_vase": {"yaw": 90, "pitch": 5},
            },
        }
    })
    app_module.save_json(app_module.artifacts_db_path(), {
        "roman_bust": {"id": "roman_bust", "name": "Roman Bust"},
        "greek_vase": {"id": "greek_vase", "name": "Greek Vase"},
        "unrelated": {"id": "unrelated", "name": "Not in this tour"},
    })

    r = client.get(f"/api/tours/{tour_id}")
    body = r.get_json()

    assert r.status_code == 200
    assert "roman_bust" in body["artifacts"]
    assert "greek_vase" in body["artifacts"]
    assert "unrelated" not in body["artifacts"]


# ---------------------------------------------------------------------------
# Artifact database endpoints
# ---------------------------------------------------------------------------

def test_get_artifact_not_found(client):
    r = client.get("/api/artifacts/ghost")
    assert r.status_code == 404


def test_create_then_update_artifact(client):
    r = client.post("/api/artifacts/roman_bust", json={
        "name": "Roman Bust", "origin": "Rome",
    })
    assert r.status_code == 201
    body = r.get_json()
    assert body["name"] == "Roman Bust"
    assert body["origin"] == "Rome"
    assert body["glb"] is None  # not generated yet

    r = client.post("/api/artifacts/roman_bust", json={"description": "Marble portrait."})
    assert r.status_code == 200
    body = r.get_json()
    assert body["description"] == "Marble portrait."
    assert body["name"] == "Roman Bust"  # untouched fields persist

    r = client.get("/api/artifacts/roman_bust")
    assert r.status_code == 200
    assert r.get_json()["description"] == "Marble portrait."


def test_artifact_upsert_requires_json_body(client):
    r = client.post("/api/artifacts/roman_bust", data="not json", content_type="text/plain")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Upload endpoints
# ---------------------------------------------------------------------------

def test_upload_image(client):
    r = client.post(
        "/api/upload-image",
        data={
            "tour_id": "tour_002",
            "waypoint_id": "waypoint_001",
            "image_index": "0",
            "image": (io.BytesIO(b"fake-jpeg-bytes"), "photo.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    saved = app_module.RAW_DIR / "tour_002" / "waypoint_001" / "0000.jpg"
    assert saved.exists()
    assert saved.read_bytes() == b"fake-jpeg-bytes"


def test_upload_artifact_image(client):
    r = client.post(
        "/api/upload-artifact-image",
        data={
            "tour_id": "tour_002",
            "artifact_id": "roman_bust",
            "image_index": "2",
            "image": (io.BytesIO(b"fake"), "photo.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    saved = app_module.RAW_ARTIFACT_IMAGES_DIR / "tour_002" / "roman_bust" / "0002.jpg"
    assert saved.exists()


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

def test_get_panorama(client):
    pano_dir = app_module.PROCESSED_DIR / "tour_002"
    pano_dir.mkdir(parents=True)
    (pano_dir / "waypoint_001.jpg").write_bytes(b"fake-jpeg")

    r = client.get("/panoramas/tour_002/waypoint_001.jpg")
    assert r.status_code == 200
    assert r.data == b"fake-jpeg"


def test_get_artifact_model(client):
    art_dir = app_module.ARTIFACT_DIR / "tour_002"
    art_dir.mkdir(parents=True)
    (art_dir / "roman_bust.glb").write_bytes(b"fake-glb")

    r = client.get("/artifacts/tour_002/roman_bust.glb")
    assert r.status_code == 200
    assert r.data == b"fake-glb"


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------

def test_status_defaults_to_idle(client):
    r = client.get("/api/status/tour_002/waypoint/waypoint_001")
    assert r.status_code == 200
    assert r.get_json() == {"status": "idle", "message": ""}


# ---------------------------------------------------------------------------
# Processing pipeline endpoints (Hugin/Meshroom/Blender mocked out)
# ---------------------------------------------------------------------------

def test_process_waypoint_success(client, monkeypatch):
    tour_id, waypoint_id = "tour_002", "waypoint_001"
    (app_module.RAW_DIR / tour_id / waypoint_id).mkdir(parents=True)

    def fake_stitch(tour_id, waypoint_id):
        pano_dir = app_module.PROCESSED_DIR / tour_id
        pano_dir.mkdir(parents=True, exist_ok=True)
        (pano_dir / f"{waypoint_id}.jpg").write_bytes(b"stitched")

    monkeypatch.setattr(app_module, "stitch_waypoint", fake_stitch)

    r = client.post(f"/api/process/{tour_id}/{waypoint_id}")
    assert r.status_code == 202
    assert r.get_json()["status"] == "queued"

    wait_for_status(client, "waypoint", tour_id, waypoint_id, "complete")

    # sync_tour_manifest should have run as a side effect of the job
    waypoints = app_module.load_json(app_module.waypoints_manifest_path(tour_id), {})
    assert waypoint_id in waypoints


def test_process_waypoint_failure_reports_error(client, monkeypatch):
    tour_id, waypoint_id = "tour_002", "waypoint_001"
    (app_module.RAW_DIR / tour_id / waypoint_id).mkdir(parents=True)

    def fake_stitch(tour_id, waypoint_id):
        raise ValueError("Not enough images found")

    monkeypatch.setattr(app_module, "stitch_waypoint", fake_stitch)

    client.post(f"/api/process/{tour_id}/{waypoint_id}")
    status = wait_for_status(client, "waypoint", tour_id, waypoint_id, "error")
    assert "Not enough images" in status["message"]


def test_process_artifact_success(client, monkeypatch):
    tour_id, artifact_id = "tour_002", "roman_bust"

    def fake_photogrammetry(tour_id, artifact_id):
        glb_dir = app_module.ARTIFACT_DIR / tour_id
        glb_dir.mkdir(parents=True, exist_ok=True)
        glb_path = glb_dir / f"{artifact_id}.glb"
        glb_path.write_bytes(b"glb-bytes")
        app_module.update_artifact_glb(tour_id, artifact_id)
        return glb_path

    monkeypatch.setattr(app_module, "process_artifact_photogrammetry", fake_photogrammetry)

    r = client.post(f"/api/process-artifact/{tour_id}/{artifact_id}")
    assert r.status_code == 202

    wait_for_status(client, "artifact", tour_id, artifact_id, "complete")

    artifacts_db = app_module.load_json(app_module.artifacts_db_path(), {})
    assert artifacts_db[artifact_id]["glb"] == f"/artifacts/{tour_id}/{artifact_id}.glb"


# ---------------------------------------------------------------------------
# Pure validation logic in the real (unmocked) pipeline functions
# ---------------------------------------------------------------------------

def test_stitch_waypoint_requires_two_images(client):
    tour_id, waypoint_id = "tour_002", "waypoint_001"
    source = app_module.RAW_DIR / tour_id / waypoint_id
    source.mkdir(parents=True)
    (source / "0000.jpg").write_bytes(b"only one image")

    with pytest.raises(ValueError, match="Not enough images"):
        app_module.stitch_waypoint(tour_id, waypoint_id)


def test_process_artifact_photogrammetry_requires_three_images(client):
    tour_id, artifact_id = "tour_002", "roman_bust"
    images_dir = app_module.RAW_ARTIFACT_IMAGES_DIR / tour_id / artifact_id
    images_dir.mkdir(parents=True)
    (images_dir / "0000.jpg").write_bytes(b"one")
    (images_dir / "0001.jpg").write_bytes(b"two")

    with pytest.raises(ValueError, match="Not enough images"):
        app_module.process_artifact_photogrammetry(tour_id, artifact_id)


def test_process_artifact_photogrammetry_missing_dir(client):
    with pytest.raises(ValueError, match="No uploaded images"):
        app_module.process_artifact_photogrammetry("tour_002", "ghost_artifact")