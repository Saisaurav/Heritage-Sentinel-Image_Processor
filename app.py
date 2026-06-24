import json
import re
import shutil
import subprocess
import time
import traceback
import uuid as _uuid

from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sympy import reduced

app = Flask(__name__)
CORS(app)

RAW_DIR = Path("raw_images/tours")
PROCESSED_DIR = Path("processed/panoramas")
ARTIFACT_DIR = Path("processed/artifacts")

RAW_ARTIFACT_IMAGES_DIR = Path("raw_images/artifacts")
MESHROOM_CACHE_DIR = Path("processed/photogrammetry_cache")

MANIFESTS_DIR = Path("manifests")
STATUS_DIR = Path("data/status")

BLENDER_CONVERT_SCRIPT = Path(__file__).resolve().parent / "scripts" / "obj_to_glb.py"

# Placeholder GLB used when Meshroom/Blender are unavailable (no GPU).
# Drop any valid .glb file here and it will be copied into the artifact output.
PLACEHOLDER_GLB = Path(__file__).resolve().parent / "assets" / "placeholder.glb"

for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    ARTIFACT_DIR,
    RAW_ARTIFACT_IMAGES_DIR,
    MESHROOM_CACHE_DIR,
    MANIFESTS_DIR,
    STATUS_DIR,
    PLACEHOLDER_GLB.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)


from concurrent.futures import ThreadPoolExecutor
_job_queue = ThreadPoolExecutor(max_workers=2)          # panorama stitching
_artifact_queue = ThreadPoolExecutor(max_workers=2)     # photogrammetry — separate pool
# ---------------------------------------------------------------------------
# Generic JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Manifest paths
# ---------------------------------------------------------------------------

def tour_manifest_path(tour_id: str) -> Path:
    return MANIFESTS_DIR / tour_id / "tour.json"


def waypoints_manifest_path(tour_id: str) -> Path:
    return MANIFESTS_DIR / tour_id / "waypoints.json"


def artifacts_db_path() -> Path:
    return MANIFESTS_DIR / "artifacts.json"


def ensure_tour_manifest(tour_id: str) -> None:
    tpath = tour_manifest_path(tour_id)
    if not tpath.exists():
        save_json(tpath, {
            "tour_id": tour_id,
            "name": tour_id.replace("_", " ").title(),
            "waypoints": [],
        })
    wpath = waypoints_manifest_path(tour_id)
    if not wpath.exists():
        save_json(wpath, {})


# ---------------------------------------------------------------------------
# Waypoint folder naming convention
# ---------------------------------------------------------------------------

def parse_waypoint_metadata(waypoint_id: str) -> Dict[str, Any]:
    match = re.match(
        r"^(?P<id>.+?)\.artifact\.(?P<names>.+)$",
        waypoint_id,
        re.IGNORECASE,
    )
    if match:
        artifact_names = [n.strip() for n in match.group("names").split(",") if n.strip()]
        return {
            "clean_id": match.group("id"),
            "is_artifact": True,
            "artifact_names": artifact_names,
        }
    return {
        "clean_id": waypoint_id,
        "is_artifact": False,
        "artifact_names": [],
    }


# ---------------------------------------------------------------------------
# Manifest sync (filesystem -> manifests/<tour_id>/*.json)
# ---------------------------------------------------------------------------

def sync_tour_manifest(tour_id: str) -> Optional[Dict[str, Any]]:
    raw_tour_dir = RAW_DIR / tour_id
    pano_dir = PROCESSED_DIR / tour_id

    if not raw_tour_dir.exists():
        return None

    ensure_tour_manifest(tour_id)

    tour = load_json(tour_manifest_path(tour_id), {})
    waypoints = load_json(waypoints_manifest_path(tour_id), {})

    tour.setdefault("waypoints", [])

    for raw_waypoint_dir in sorted(raw_tour_dir.iterdir()):
        if not raw_waypoint_dir.is_dir():
            continue

        meta = parse_waypoint_metadata(raw_waypoint_dir.name)
        clean_id = meta["clean_id"]
        pano_name = f"{clean_id}.jpg"

        if not (pano_dir / pano_name).exists():
            continue

        entry = waypoints.get(clean_id, {
            "id": clean_id,
            "mapPos": [0, 0],
            "links": [],
        })

        entry["id"] = clean_id
        panorama_path = f"/panoramas/{tour_id}/{pano_name}"
        entry["image"] = panorama_path
        entry["panoramaUrl"] = panorama_path
        entry["isArtifact"] = meta["is_artifact"]

        if meta["is_artifact"]:
            entry["artifactIds"] = meta["artifact_names"]
            existing_markers = entry.get("artifactMarkers", {})
            entry["artifactMarkers"] = {
                aid: existing_markers.get(aid, {"yaw": 0, "pitch": 0})
                for aid in meta["artifact_names"]
            }
            entry.pop("artifactId", None)
            entry.pop("artifactMarker", None)
        else:
            entry.pop("artifactIds", None)
            entry.pop("artifactMarkers", None)
            entry.pop("artifactId", None)
            entry.pop("artifactMarker", None)

        waypoints[clean_id] = entry

        if clean_id not in tour["waypoints"]:
            tour["waypoints"].append(clean_id)

    save_json(tour_manifest_path(tour_id), tour)
    save_json(waypoints_manifest_path(tour_id), waypoints)

    return {"tour": tour, "waypoints": waypoints}


# ---------------------------------------------------------------------------
# Artifact database (manifests/artifacts.json)
# ---------------------------------------------------------------------------

def default_artifact_entry(artifact_id: str) -> Dict[str, Any]:
    return {
        "id": artifact_id,
        "name": artifact_id.replace("_", " ").title(),
        "description": "",
        "origin": "",
        "material": "",
        "acquisition": "",
        "glb": None,
    }


def update_artifact_glb(tour_id: str, artifact_id: str) -> None:
    artifacts_db = load_json(artifacts_db_path(), {})
    entry = artifacts_db.get(artifact_id, default_artifact_entry(artifact_id))
    entry["glb"] = f"/artifacts/{tour_id}/{artifact_id}.glb"
    artifacts_db[artifact_id] = entry
    save_json(artifacts_db_path(), artifacts_db)


# ---------------------------------------------------------------------------
# Job status tracking
# ---------------------------------------------------------------------------

def status_path(kind: str, tour_id: str, item_id: str) -> Path:
    return STATUS_DIR / tour_id / kind / f"{item_id}.json"


def set_status(kind, tour_id, item_id, state, message="", session_id=None):
    save_json(status_path(kind, tour_id, item_id), {"status": state, "message": message, "session_id": session_id})

def get_status(kind: str, tour_id: str, item_id: str) -> Dict[str, str]:
    return load_json(status_path(kind, tour_id, item_id), {"status": "idle", "message": ""})


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

def run_waypoint_job(tour_id: str, waypoint_id: str) -> None:
    set_status("waypoint", tour_id, waypoint_id, "processing")
    try:
        stitch_waypoint(tour_id, waypoint_id)
        sync_tour_manifest(tour_id)
        set_status("waypoint", tour_id, waypoint_id, "complete")
    except Exception as e:
        traceback.print_exc()
        set_status("waypoint", tour_id, waypoint_id, "error", str(e))


def run_artifact_job(tour_id: str, artifact_id: str, session_id: Optional[str] = None) -> None:
    """Run photogrammetry for one artifact.

    If Meshroom or Blender fail (e.g. no GPU), fall back to copying the
    placeholder GLB so the rest of the pipeline still works.
    """
    set_status("artifact", tour_id, artifact_id, "processing")

    def _push_event(level: str, msg: str) -> None:
        if not session_id:
            return
        try:
            sess = _load_artifact_session(session_id)
            if sess:
                _artifact_session_append_event(sess, level, msg)
                _save_artifact_session(sess)
        except Exception:
            pass

    def _push_step(step: str) -> None:
        if not session_id:
            return
        try:
            sess = _load_artifact_session(session_id)
            if sess:
                sess["step"] = step
                _save_artifact_session(sess)
        except Exception:
            pass

    try:
        _push_step("meshroom")
        _push_event("info", "Starting Meshroom photogrammetry…")
        glb_path = process_artifact_photogrammetry(tour_id, artifact_id)
        _push_step("complete")
        _push_event("success", f"GLB ready: /artifacts/{tour_id}/{artifact_id}.glb")
        set_status("artifact", tour_id, artifact_id, "complete")

    except Exception as e:
        traceback.print_exc()
        _push_event("warning", f"Photogrammetry failed ({e}). Using placeholder GLB.")

        # ── No-GPU fallback: copy placeholder ────────────────────────────────
        glb_out = ARTIFACT_DIR / tour_id / f"{artifact_id}.glb"
        glb_out.parent.mkdir(parents=True, exist_ok=True)

        if PLACEHOLDER_GLB.exists():
            shutil.copy2(PLACEHOLDER_GLB, glb_out)
            _push_event("info", "Placeholder GLB copied to output.")
        else:
            # Write a minimal valid GLB (binary glTF magic + empty JSON chunk)
            _write_minimal_glb(glb_out)
            _push_event("info", "Minimal placeholder GLB written.")

        update_artifact_glb(tour_id, artifact_id)
        _push_step("complete")
        set_status("artifact", tour_id, artifact_id, "complete", "placeholder")


def _write_minimal_glb(path: Path) -> None:
    """Write the smallest valid GLB file (empty scene) so model-viewer
    doesn't crash when there's no real mesh."""
    import struct

    gltf = json.dumps({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
    }).encode("utf-8")

    # Pad JSON chunk to 4-byte boundary
    pad = (4 - len(gltf) % 4) % 4
    gltf += b" " * pad

    # GLB header (12 bytes) + JSON chunk header (8 bytes) + JSON chunk data
    total_len = 12 + 8 + len(gltf)
    header = struct.pack("<III", 0x46546C67, 2, total_len)        # magic, version, length
    chunk_header = struct.pack("<II", len(gltf), 0x4E4F534A)      # chunkLength, chunkType JSON

    path.write_bytes(header + chunk_header + gltf)


# ---------------------------------------------------------------------------
# Hugin panorama stitching
# ---------------------------------------------------------------------------

def stitch_waypoint(tour_id: str, waypoint_id: str) -> None:
    source = (RAW_DIR / tour_id / waypoint_id).resolve()
    output_dir = (PROCESSED_DIR / tour_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pto = source / "project.pto"
    images = sorted(source.glob("*.jpg"))
    if len(images) < 2:
        raise ValueError(f"Not enough images found in {source}")
    images = [img.resolve() for img in images]
    subprocess.run(["pto_gen", "-o", str(pto), *map(str, images)], check=True)
    subprocess.run(["cpfind", "--multirow", "-o", str(pto), str(pto)], check=True)
    subprocess.run(["cpclean", "-o", str(pto), str(pto)], check=True)
    subprocess.run(["autooptimiser", "-a", "-l", "-s", "-m", "-o", str(pto), str(pto)], check=True)
    subprocess.run(["pano_modify", "--canvas=AUTO", "--crop=AUTO", "-o", str(pto), str(pto)], check=True)
    meta = parse_waypoint_metadata(waypoint_id)
    prefix = output_dir / meta["clean_id"]
    subprocess.run(["hugin_executor", "--stitching", "--prefix", str(prefix), str(pto)], check=True)
    tif_file = prefix.with_suffix(".tif")
    jpg_file = prefix.with_suffix(".jpg")
    img = Image.open(tif_file)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    img.save(jpg_file, "JPEG", quality=95, optimize=True)
    tif_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Photogrammetry: Meshroom + Blender
# ---------------------------------------------------------------------------

def _debug_list_cache_dir(cache_dir: Path) -> None:
    print(f"DEBUG: listing cache dir {cache_dir}")
    if not cache_dir.exists():
        print("DEBUG: cache dir does not exist")
        return

    entries = sorted(cache_dir.rglob("*"), key=lambda p: (p.is_file(), str(p)))
    if not entries:
        print("DEBUG: cache dir is empty")
        return

    for path in entries:
        rel = path.relative_to(cache_dir)
        if path.is_dir():
            print(f"DEBUG:   DIR  {rel}")
        else:
            print(f"DEBUG:   FILE {rel} ({path.stat().st_size} bytes)")


def find_textured_mesh(cache_dir: Path, output_dir: Path) -> Path:
    for root_dir, label in [(output_dir, "output"), (cache_dir, "cache")]:
        if root_dir.exists():
            matches = sorted(root_dir.glob("**/texturedMesh.obj"))
            if matches:
                print(f"DEBUG: found texturedMesh.obj under {label} directory {root_dir}")
                return max(matches, key=lambda p: p.stat().st_mtime)
            else:
                print(f"DEBUG: no texturedMesh.obj under {label} directory {root_dir}")
        else:
            print(f"DEBUG: {label} directory does not exist: {root_dir}")

    print(f"DEBUG: No texturedMesh.obj found under cache={cache_dir} or output={output_dir}")
    _debug_list_cache_dir(cache_dir)
    _debug_list_cache_dir(output_dir)
    candidates = []
    for root_dir, label in [(output_dir, "output"), (cache_dir, "cache")]:
        for pattern in ["**/*.obj", "**/*.gltf", "**/*.glb", "**/*.sfm", "**/*.json"]:
            found = sorted(root_dir.glob(pattern))
            if found:
                candidates.extend(found)
                print(f"DEBUG: found {len(found)} files matching {pattern} in {label} directory")
        if found:
            print(f"DEBUG: candidate files under {label} directory:")
            for path in sorted(found):
                print(f"DEBUG:   {path.relative_to(root_dir)}")
    raise FileNotFoundError(f"No textured mesh found under {cache_dir} or {output_dir}")


def run_meshroom(images_dir: Path, cache_dir: Path) -> Path:
    output_dir = cache_dir / "output"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: running meshroom_batch with images_dir={images_dir} cache_dir={cache_dir} output_dir={output_dir}")
    print("DEBUG: meshroom_batch command:", "meshroom_batch", "-i", str(images_dir), "--cache", str(cache_dir), "-o", str(output_dir), "--pipeline", "photogrammetryDraft")
    subprocess.run(
        [
            "meshroom_batch",
            "-i", str(images_dir),
            "--cache", str(cache_dir),
            "-o", str(output_dir),
            "--pipeline", "photogrammetryDraft",  # CPU-only, skips DepthMap
        ],
        check=True,
    )
    print("DEBUG: meshroom_batch finished")
    _debug_list_cache_dir(cache_dir)
    _debug_list_cache_dir(output_dir)
    return find_textured_mesh(cache_dir, output_dir)


def convert_obj_to_glb(obj_path: Path, glb_path: Path) -> None:
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["blender", "-b", "--python", str(BLENDER_CONVERT_SCRIPT), "--", str(obj_path), str(glb_path)],
        check=True,
    )


def process_artifact_photogrammetry(tour_id: str, artifact_id: str) -> Path:
    images_dir = (RAW_ARTIFACT_IMAGES_DIR / tour_id / artifact_id).resolve()
    if not images_dir.exists():
        raise ValueError(f"No uploaded images found for {tour_id}/{artifact_id}")
    images = sorted(images_dir.glob("*.jpg"))
    if len(images) < 3:
        raise ValueError(f"Need ≥3 images, found {len(images)} in {images_dir}")
    cache_dir = (MESHROOM_CACHE_DIR / tour_id / artifact_id).resolve()
    obj_path = run_meshroom(images_dir, cache_dir)
    glb_path = (ARTIFACT_DIR / tour_id / f"{artifact_id}.glb").resolve()
    convert_obj_to_glb(obj_path, glb_path)
    update_artifact_glb(tour_id, artifact_id)
    return glb_path


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/panoramas/<tour_id>/<filename>")
def get_panorama(tour_id, filename):
    return send_from_directory(PROCESSED_DIR / tour_id, filename)


@app.route("/artifacts/<tour_id>/<filename>")
def get_artifact_model(tour_id, filename):
    return send_from_directory(ARTIFACT_DIR / tour_id, filename)


# ---------------------------------------------------------------------------
# Tour manifest endpoints
# ---------------------------------------------------------------------------

@app.get("/api/tours/<tour_id>/manifest")
@app.get("/api/tours/<tour_id>")
def get_tour(tour_id):
    tpath = tour_manifest_path(tour_id)
    if not tpath.exists():
        return jsonify({"error": "Tour not found"}), 404

    tour = load_json(tpath, {})
    waypoints = load_json(waypoints_manifest_path(tour_id), {})
    if isinstance(waypoints, list):
        try:
            waypoints = {w.get("id"): w for w in waypoints if w and w.get("id")}
        except Exception:
            waypoints = {}

    artifacts_db = load_json(artifacts_db_path(), {})
    referenced_ids = set()
    for wp in (waypoints.values() if isinstance(waypoints, dict) else []):
        try:
            if wp.get("isArtifact"):
                for aid in (wp.get("artifactIds") or []):
                    referenced_ids.add(aid)
        except Exception:
            continue
    artifacts = {aid: artifacts_db[aid] for aid in referenced_ids if aid in artifacts_db}

    try:
        if request.path.endswith("/manifest"):
            wp_list = []
            if isinstance(waypoints, dict):
                ordered_ids = tour.get("waypoints") or list(waypoints.keys())
                for wid in ordered_ids:
                    if wid in waypoints:
                        wp_list.append(waypoints[wid])
            else:
                wp_list = waypoints
            return jsonify({"tour": tour, "waypoints": wp_list, "artifacts": artifacts}), 200
    except Exception:
        pass

    return jsonify({"tour": tour, "waypoints": waypoints, "artifacts": artifacts}), 200


@app.put("/api/tours/<tour_id>")
def save_tour(tour_id):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    ensure_tour_manifest(tour_id)
    tour = load_json(tour_manifest_path(tour_id), {})
    waypoints = load_json(waypoints_manifest_path(tour_id), {})

    incoming_wps = data.get("waypoints", [])
    wp_ids = []
    for w in incoming_wps:
        wid = w.get("id")
        if not wid:
            continue
        entry = waypoints.get(wid, {"id": wid, "mapPos": [0, 0], "links": []})
        entry["id"] = wid
        entry["name"] = w.get("name", wid)
        entry["mapPos"] = [w.get("x", 0), w.get("y", 0)]
        entry["links"] = w.get("links", [])
        if "isArtifact" in w:
            entry["isArtifact"] = w["isArtifact"]
        if "artifactIds" in w:
            entry["artifactIds"] = w["artifactIds"]
        waypoints[wid] = entry
        wp_ids.append(wid)

    tour["waypoints"] = wp_ids
    if "name" in data:
       tour["name"] = data["name"]
    if "map_image" in data:
        tour["map_image"] = data["map_image"]
    else:
        tour.pop("map_image", None)  # don't persist the placeholder path

    save_json(tour_manifest_path(tour_id), tour)
    save_json(waypoints_manifest_path(tour_id), waypoints)

    return jsonify({"ok": True, "waypoints": len(wp_ids)}), 200


@app.post("/api/tours/<tour_id>/waypoints/<waypoint_id>/marker")
def update_waypoint_marker(tour_id, waypoint_id):
    data = request.get_json(silent=True)
    if not data or "yaw" not in data or "pitch" not in data:
        return jsonify({"error": "yaw and pitch are required"}), 400

    try:
        yaw = float(data["yaw"])
        pitch = float(data["pitch"])
    except (TypeError, ValueError):
        return jsonify({"error": "yaw and pitch must be numbers"}), 400

    ensure_tour_manifest(tour_id)
    waypoints = load_json(waypoints_manifest_path(tour_id), {})

    if waypoint_id not in waypoints:
        return jsonify({"error": "Waypoint not found"}), 404

    entry = waypoints[waypoint_id]
    artifact_ids = entry.get("artifactIds", [])

    if len(artifact_ids) > 1 and not data.get("artifact_id"):
        return jsonify({"error": "artifact_id is required when waypoint has multiple artifacts"}), 400

    artifact_id = data.get("artifact_id") or (artifact_ids[0] if artifact_ids else None)

    if artifact_id and artifact_id not in artifact_ids:
        return jsonify({"error": f"artifact_id '{artifact_id}' not found on this waypoint"}), 404

    markers = entry.setdefault("artifactMarkers", {})
    if artifact_id:
        markers[artifact_id] = {"yaw": yaw, "pitch": pitch}
    else:
        entry["artifactMarker"] = {"yaw": yaw, "pitch": pitch}

    save_json(waypoints_manifest_path(tour_id), waypoints)
    return jsonify(entry), 200


@app.post("/api/tours/<tour_id>/waypoints/<waypoint_id>/artifact")
def update_waypoint_artifact(tour_id, waypoint_id):
    data = request.get_json(silent=True)
    if not data or "isArtifact" not in data:
        return jsonify({"error": "isArtifact is required"}), 400

    is_artifact = bool(data["isArtifact"])
    artifact_ids = data.get("artifactIds", [])
    if not artifact_ids and data.get("artifactId"):
        artifact_ids = [data["artifactId"]]

    if is_artifact and not artifact_ids:
        return jsonify({"error": "artifactIds is required when isArtifact is true"}), 400

    ensure_tour_manifest(tour_id)
    waypoints = load_json(waypoints_manifest_path(tour_id), {})

    if waypoint_id not in waypoints:
        return jsonify({"error": "Waypoint not found"}), 404

    entry = waypoints[waypoint_id]
    entry["isArtifact"] = is_artifact

    if is_artifact:
        entry["artifactIds"] = artifact_ids
        existing_markers = entry.get("artifactMarkers", {})
        entry["artifactMarkers"] = {
            aid: existing_markers.get(aid, {"yaw": 0, "pitch": 0})
            for aid in artifact_ids
        }
        entry.pop("artifactId", None)
        entry.pop("artifactMarker", None)
    else:
        entry.pop("artifactIds", None)
        entry.pop("artifactMarkers", None)
        entry.pop("artifactId", None)
        entry.pop("artifactMarker", None)

    waypoints[waypoint_id] = entry
    save_json(waypoints_manifest_path(tour_id), waypoints)
    return jsonify(entry), 200


# ---------------------------------------------------------------------------
# Artifact database endpoints
# ---------------------------------------------------------------------------

@app.get("/api/artifacts/<artifact_id>")
def get_artifact(artifact_id):
    artifacts_db = load_json(artifacts_db_path(), {})
    artifact = artifacts_db.get(artifact_id)
    if artifact is None:
        return jsonify({"error": "Artifact not found"}), 404
    return jsonify(artifact), 200


@app.post("/api/artifacts/<artifact_id>")
def upsert_artifact(artifact_id):
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    artifacts_db = load_json(artifacts_db_path(), {})
    is_new = artifact_id not in artifacts_db
    entry = artifacts_db.get(artifact_id, default_artifact_entry(artifact_id))

    for field in ("name", "description", "origin", "period", "material", "dimensions", "acquisition", "tags"):
        if field in data:
            entry[field] = data[field]

    entry["id"] = artifact_id
    artifacts_db[artifact_id] = entry
    save_json(artifacts_db_path(), artifacts_db)
    return jsonify(entry), (201 if is_new else 200)


# ---------------------------------------------------------------------------
# Processing pipeline endpoints
# ---------------------------------------------------------------------------

@app.post("/api/process/<tour_id>/<waypoint_id>")
def process_waypoint(tour_id, waypoint_id):
    _job_queue.submit(run_waypoint_job, tour_id, waypoint_id)
    set_status("waypoint", tour_id, waypoint_id, "queued")
    return jsonify({"status": "queued"}), 202


@app.post("/api/upload-image")
def upload_image():
    tour_id = request.form["tour_id"]
    waypoint_id = request.form["waypoint_id"]
    image_index = int(request.form["image_index"])
    image = request.files["image"]
    target = RAW_DIR / tour_id / waypoint_id
    target.mkdir(parents=True, exist_ok=True)
    image.save(target / f"{image_index:04d}.jpg")
    return jsonify({"status": "ok"})


@app.post("/api/upload-artifact-image")
def upload_artifact_image():
    tour_id = request.form["tour_id"]
    artifact_id = request.form["artifact_id"]
    image_index = int(request.form["image_index"])
    image = request.files["image"]
    target = RAW_ARTIFACT_IMAGES_DIR / tour_id / artifact_id
    target.mkdir(parents=True, exist_ok=True)
    image.save(target / f"{image_index:04d}.jpg")
    return jsonify({"status": "ok"})

@app.post("/api/process-artifact/<tour_id>/<artifact_id>")
def process_artifact_endpoint(tour_id, artifact_id):
    session_id = request.args.get("session_id") or (request.get_json(silent=True) or {}).get("session_id")
    _artifact_queue.submit(run_artifact_job, tour_id, artifact_id, session_id)
    set_status("artifact", tour_id, artifact_id, "queued")
    return jsonify({"status": "queued"}), 202


@app.get("/api/status/<tour_id>/waypoint/<waypoint_id>")
def get_waypoint_status(tour_id, waypoint_id):
    return jsonify(get_status("waypoint", tour_id, waypoint_id))


@app.get("/api/status/<tour_id>/artifact/<artifact_id>")
def get_artifact_status(tour_id, artifact_id):
    return jsonify(get_status("artifact", tour_id, artifact_id))


@app.get("/")
def home():
    return jsonify({"status": "running", "service": "Heritage Sentinel Image Processor"})


# ---------------------------------------------------------------------------
# Panorama scan sessions (existing, unchanged in contract)
# ---------------------------------------------------------------------------

SESSION_STORE: Dict[str, Any] = {}
SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> Optional[Dict]:
    p = _session_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return SESSION_STORE.get(session_id)


def _save_session(session: Dict) -> None:
    sid = session["session_id"]
    SESSION_STORE[sid] = session
    _session_path(sid).write_text(json.dumps(session, indent=2))


@app.post("/session/start")
def session_start():
    data = request.get_json(silent=True) or {}
    initial_waypoints = [
        {
            "id": w.get("id", ""),
            "name": w.get("name", w.get("id", "")),
            "status": "pending",
            "images_captured": 0,
            "images_total": 0,
            "images_uploaded": 0,
        }
        for w in data.get("waypoints", [])
    ]
    session_id = f"run_{_uuid.uuid4().hex[:10]}"
    session = {
        "session_id": session_id,
        "tour_id": data.get("tour_id"),
        "started_at": time.time() * 1000,
        "paused": False,
        "current_index": 0,
        "waypoints": initial_waypoints,
        "connection": {"online": True, "rssi": -60},
        "battery": None,
        "camera": "ready",
        "events": [
            {
                "id": _uuid.uuid4().hex,
                "ts": time.time() * 1000,
                "level": "info",
                "msg": f"Session {session_id} started",
            }
        ],
        "last_error": None,
    }
    _save_session(session)
    return jsonify(session), 201


@app.post("/session/end")
def session_end():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid:
        return jsonify({"error": "session_id required"}), 400
    SESSION_STORE.pop(sid, None)
    p = _session_path(sid)
    if p.exists():
        p.unlink()
    return jsonify({"ok": True}), 200


@app.get("/session/status/<session_id>")
def session_status(session_id):
    session = _load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    try:
        tour_id = session.get("tour_id")
        wps = session.get("waypoints", [])
        for w in wps:
            wid = w.get("id")
            if not wid or not tour_id:
                continue
            st = get_status("waypoint", tour_id, wid)
            backend_state = st.get("status") if isinstance(st, dict) else None
            if backend_state:
                if backend_state in ("queued", "processing"):
                    w["status"] = "processing"
                elif backend_state in ("complete", "completed"):
                    w["status"] = "complete"
                elif backend_state in ("error", "failed"):
                    w["status"] = "failed"
                else:
                    w["status"] = backend_state
        session["waypoints"] = wps
    except Exception:
        pass
    return jsonify(session), 200


@app.get("/session/status/<session_id>/waypoints")
def session_waypoints(session_id):
    session = _load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "session_id": session_id,
        "tour_id": session.get("tour_id", "tour_001"),
        "waypoints": session.get("waypoints", []),
    }), 200


@app.post("/session/pause")
def session_pause():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    session = _load_session(sid) if sid else None
    if not session:
        return jsonify({"error": "Session not found"}), 404
    session["paused"] = True
    _save_session(session)
    return jsonify(session), 200


@app.post("/session/resume")
def session_resume():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    session = _load_session(sid) if sid else None
    if not session:
        return jsonify({"error": "Session not found"}), 404
    session["paused"] = False
    _save_session(session)
    return jsonify(session), 200


@app.post("/session/<session_id>/telemetry")
def session_telemetry(session_id):
    session = _load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    if "battery" in data:
        session["battery"] = float(data["battery"])
    if "rssi" in data:
        session["connection"]["rssi"] = int(data["rssi"])
        session["connection"]["online"] = True
    if "camera" in data:
        session["camera"] = data["camera"]
    if "current_index" in data:
        session["current_index"] = int(data["current_index"])
    wp_update = data.get("waypoint_status")
    if wp_update and "id" in wp_update:
        wps = session.get("waypoints", [])
        existing = next((w for w in wps if w["id"] == wp_update["id"]), None)
        if existing:
            existing.update(wp_update)
        else:
            wps.append(wp_update)
        session["waypoints"] = wps
    _save_session(session)
    return jsonify({"ok": True}), 200


@app.post("/session/<session_id>/event")
def session_event(session_id):
    session = _load_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    event = {
        "id": _uuid.uuid4().hex,
        "ts": time.time() * 1000,
        "level": data.get("level", "info"),
        "msg": str(data.get("msg", "")),
    }
    events = session.get("events", [])
    events.append(event)
    session["events"] = events[-200:]
    if data.get("level") == "error":
        session["last_error"] = event["msg"]
    _save_session(session)
    return jsonify(event), 201


# ---------------------------------------------------------------------------
# Artifact scan sessions  ← NEW
#
# Mirrors the panorama session pattern but for photogrammetry runs.
# One artifact session = one artifact being scanned by the robot arm.
#
# Schema:
# {
#   "session_id": "art_<hex>",
#   "tour_id": "tour_TEST69",
#   "artifact_id": "roman_bust",
#   "waypoint_id": "waypoint_003",
#   "started_at": <ms>,
#   "step": "capturing" | "uploading" | "meshroom" | "generating" | "complete" | "error",
#   "images_captured": 12,
#   "images_total": 144,
#   "images_uploaded": 10,
#   "ring": 1,          # which capture ring (1-4)
#   "events": [...],
#   "glb_url": null | "/artifacts/<tour>/<artifact>.glb"
# }
# ---------------------------------------------------------------------------

ARTIFACT_SESSIONS_DIR = Path("data/artifact_sessions")
ARTIFACT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_SESSION_STORE: Dict[str, Any] = {}


def _artifact_session_path(session_id: str) -> Path:
    return ARTIFACT_SESSIONS_DIR / f"{session_id}.json"


def _load_artifact_session(session_id: str) -> Optional[Dict]:
    p = _artifact_session_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return ARTIFACT_SESSION_STORE.get(session_id)


def _save_artifact_session(session: Dict) -> None:
    sid = session["session_id"]
    ARTIFACT_SESSION_STORE[sid] = session
    _artifact_session_path(sid).write_text(json.dumps(session, indent=2))


def _artifact_session_append_event(session: Dict, level: str, msg: str) -> None:
    event = {
        "id": _uuid.uuid4().hex,
        "ts": time.time() * 1000,
        "level": level,
        "msg": msg,
    }
    events = session.get("events", [])
    events.append(event)
    session["events"] = events[-200:]
    if level == "error":
        session["last_error"] = msg


@app.post("/artifact-session/start")
def artifact_session_start():
    """Dashboard calls this after Save Draft to start a photogrammetry run.

    Body:
    {
        "tour_id": "tour_TEST69",
        "artifact_id": "roman_bust",
        "waypoint_id": "waypoint_003",
        "images_total": 144
    }
    Returns the full session object including session_id.
    """
    
    data = request.get_json(silent=True) or {}
    tour_id = data.get("tour_id")
    artifact_id = data.get("artifact_id")
    waypoint_id = data.get("waypoint_id")

    if not tour_id or not artifact_id:
        return jsonify({"error": "tour_id and artifact_id are required"}), 400
    
    session_id = f"art_{_uuid.uuid4().hex[:10]}"
    session = {
        "session_id": session_id,
        "tour_id": tour_id,
        "artifact_id": artifact_id,
        "waypoint_id": waypoint_id,
        "started_at": time.time() * 1000,
        "step": "capturing",
        "images_captured": 0,
        "images_total": data.get("images_total", 144),
        "images_uploaded": 0,
        "ring": 1,
        "glb_url": None,
        "last_error": None,
        "events": [
            {
                "id": _uuid.uuid4().hex,
                "ts": time.time() * 1000,
                "level": "info",
                "msg": f"Artifact session {session_id} started for {artifact_id}",
            }
        ],
    }
    set_status("artifact", tour_id, artifact_id, "idle")
    _save_artifact_session(session)
    return jsonify(session), 201


@app.get("/artifact-session/status/<session_id>")
def artifact_session_status(session_id):
    """Dashboard polls this every 2 s."""
    session = _load_artifact_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    # Merge backend job status so dashboard sees meshroom/complete without polling separately
    try:
        tour_id = session.get("tour_id")
        artifact_id = session.get("artifact_id")
        if tour_id and artifact_id:
            st = get_status("artifact", tour_id, artifact_id)
            backend = st.get("status") if isinstance(st, dict) else None
            if backend == "complete" and session.get("step") not in ("complete", "error"):
                session["step"] = "complete"
                # Attach the glb url once done
                arts_db = load_json(artifacts_db_path(), {})
                art = arts_db.get(artifact_id, {})
                session["glb_url"] = art.get("glb")
                _save_artifact_session(session)
            elif backend == "error" and session.get("step") != "error":
                session["step"] = "error"
                _save_artifact_session(session)
    except Exception:
        pass

    return jsonify(session), 200


@app.post("/artifact-session/<session_id>/telemetry")
def artifact_session_telemetry(session_id):
    """PowerShell script pushes image capture progress here.

    Body (all fields optional):
    {
        "images_captured": 36,
        "images_uploaded": 34,
        "images_total": 144,
        "ring": 2,
        "step": "capturing" | "uploading" | "meshroom" | "generating"
    }
    """
    session = _load_artifact_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    for field in ("images_captured", "images_uploaded", "images_total", "ring"):
        if field in data:
            session[field] = int(data[field])
    if "step" in data:
        session["step"] = data["step"]

    _save_artifact_session(session)
    return jsonify({"ok": True}), 200


@app.post("/artifact-session/<session_id>/event")
def artifact_session_event(session_id):
    """PowerShell / background job pushes log events here."""
    session = _load_artifact_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    _artifact_session_append_event(session, data.get("level", "info"), str(data.get("msg", "")))
    _save_artifact_session(session)
    return jsonify({"ok": True}), 201


@app.post("/artifact-session/end")
def artifact_session_end():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid:
        return jsonify({"error": "session_id required"}), 400
    ARTIFACT_SESSION_STORE.pop(sid, None)
    p = _artifact_session_path(sid)
    if p.exists():
        p.unlink()
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# Startup sync
# ---------------------------------------------------------------------------

for tour_dir in RAW_DIR.iterdir():
    if tour_dir.is_dir():
        sync_tour_manifest(tour_dir.name)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)