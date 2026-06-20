import json
import re
import subprocess
import threading
import traceback

from pathlib import Path
from PIL import Image

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RAW_DIR = Path("raw_images/tours")
PROCESSED_DIR = Path("processed/panoramas")
ARTIFACT_DIR = Path("processed/artifacts")

RAW_ARTIFACT_IMAGES_DIR = Path("raw_images/artifacts")
MESHROOM_CACHE_DIR = Path("processed/photogrammetry_cache")

TOURS_DIR = Path("data/tours")
ARTIFACT_META_DIR = Path("data/artifacts")
STATUS_DIR = Path("data/status")

BLENDER_CONVERT_SCRIPT = Path(__file__).resolve().parent / "scripts" / "obj_to_glb.py"

for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    ARTIFACT_DIR,
    RAW_ARTIFACT_IMAGES_DIR,
    MESHROOM_CACHE_DIR,
    TOURS_DIR,
    ARTIFACT_META_DIR,
    STATUS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def parse_waypoint_metadata(waypoint_id: str):
    match = re.match(
        r"^(?P<id>.+?)\.artifact\.(?P<name>.+)$",
        waypoint_id,
        re.IGNORECASE,
    )

    if match:
        return {
            "clean_id": match.group("id"),
            "is_artifact": True,
            "artifact_name": match.group("name"),
        }

    return {
        "clean_id": waypoint_id,
        "is_artifact": False,
        "artifact_name": None,
    }


def build_tour_manifest(tour_id):
    raw_tour_dir = RAW_DIR / tour_id
    pano_dir = PROCESSED_DIR / tour_id

    if not raw_tour_dir.exists():
        return None

    waypoints = []

    for raw_waypoint_dir in sorted(raw_tour_dir.iterdir()):
        if not raw_waypoint_dir.is_dir():
            continue

        meta = parse_waypoint_metadata(raw_waypoint_dir.name)

        pano_name = f"{meta['clean_id']}.jpg"

        if not (pano_dir / pano_name).exists():
            continue

        waypoint = {
            "id": meta["clean_id"],
            "image": f"http://localhost:5000/panoramas/{tour_id}/{pano_name}",
            "isArtifact": meta["is_artifact"],
        }

        if meta["is_artifact"]:
            artifact_id = meta["artifact_name"]

            waypoint["artifactId"] = artifact_id
            waypoint["model"] = (
                f"http://localhost:5000/artifacts/{tour_id}/{artifact_id}.glb"
            )

        waypoints.append(waypoint)

    manifest = {
        "tour_id": tour_id,
        "waypoints": waypoints,
    }

    manifest_path = TOURS_DIR / f"{tour_id}.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_artifact_metadata(artifact_id):
    path = ARTIFACT_META_DIR / f"{artifact_id}.json"

    if not path.exists():
        return {
            "id": artifact_id,
            "name": artifact_id.replace("_", " ").title(),
            "description": "",
            "origin": "",
            "material": "",
            "acquisition": "",
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def status_path(kind: str, tour_id: str, item_id: str) -> Path:
    # kind is "waypoint" or "artifact" — keeps the two namespaces separate
    # since a waypoint id and an artifact id could theoretically collide.
    return STATUS_DIR / tour_id / kind / f"{item_id}.json"


def set_status(kind: str, tour_id: str, item_id: str, state: str, message: str = ""):
    path = status_path(kind, tour_id, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"status": state, "message": message}, f, indent=2)


def get_status(kind: str, tour_id: str, item_id: str):
    path = status_path(kind, tour_id, item_id)

    if not path.exists():
        return {"status": "idle", "message": ""}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_waypoint_job(tour_id, waypoint_id):
    set_status("waypoint", tour_id, waypoint_id, "processing")

    try:
        stitch_waypoint(tour_id, waypoint_id)
        build_tour_manifest(tour_id)

        set_status("waypoint", tour_id, waypoint_id, "complete")

    except Exception as e:
        traceback.print_exc()
        set_status("waypoint", tour_id, waypoint_id, "error", str(e))


def run_artifact_job(tour_id, artifact_id):
    set_status("artifact", tour_id, artifact_id, "processing")

    try:
        process_artifact_photogrammetry(tour_id, artifact_id)

        set_status("artifact", tour_id, artifact_id, "complete")

    except Exception as e:
        traceback.print_exc()
        set_status("artifact", tour_id, artifact_id, "error", str(e))


    except Exception as e:
        traceback.print_exc()
        set_status("artifact", tour_id, artifact_id, "error", str(e))


def stitch_waypoint(tour_id, waypoint_id):
    source = (RAW_DIR / tour_id / waypoint_id).resolve()

    output_dir = (PROCESSED_DIR / tour_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pto = source / "project.pto"

    images = sorted(source.glob("*.jpg"))

    if len(images) < 2:
        raise ValueError(f"Not enough images found in {source}")

    images = [img.resolve() for img in images]

    subprocess.run(
        ["pto_gen", "-o", str(pto), *map(str, images)],
        check=True
    )

    subprocess.run(
        ["cpfind", "--multirow", "-o", str(pto), str(pto)],
        check=True
    )

    subprocess.run(
        ["cpclean", "-o", str(pto), str(pto)],
        check=True
    )

    subprocess.run(
        ["autooptimiser", "-a", "-l", "-s", "-m", "-o", str(pto), str(pto)],
        check=True
    )

    subprocess.run(
        ["pano_modify", "--canvas=AUTO", "--crop=AUTO", "-o", str(pto), str(pto)],
        check=True
    )

    meta = parse_waypoint_metadata(waypoint_id)

    prefix = output_dir / meta["clean_id"]

    subprocess.run(
        [
            "hugin_executor",
            "--stitching",
            "--prefix",
            str(prefix),
            str(pto),
        ],
        check=True
    )

    tif_file = prefix.with_suffix(".tif")
    jpg_file = prefix.with_suffix(".jpg")

    img = Image.open(tif_file)

    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")

    img.save(
        jpg_file,
        "JPEG",
        quality=95,
        optimize=True,
    )

    tif_file.unlink(missing_ok=True)


def find_textured_mesh(cache_dir: Path) -> Path:
    """Locate the textured OBJ Meshroom produced inside its cache dir."""
    matches = sorted(cache_dir.glob("**/texturedMesh.obj"))

    if not matches:
        raise FileNotFoundError(
            f"No textured mesh (texturedMesh.obj) found under {cache_dir}"
        )

    # If Meshroom re-ran, take the most recently modified one.
    return max(matches, key=lambda p: p.stat().st_mtime)


def run_meshroom(images_dir: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "meshroom_batch",
            "-i", str(images_dir),
            "-o", str(cache_dir),
        ],
        check=True,
    )

    return find_textured_mesh(cache_dir)


def convert_obj_to_glb(obj_path: Path, glb_path: Path):
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "blender",
            "-b",
            "--python", str(BLENDER_CONVERT_SCRIPT),
            "--",
            str(obj_path),
            str(glb_path),
        ],
        check=True,
    )


def process_artifact_photogrammetry(tour_id: str, artifact_id: str) -> Path:
    images_dir = (RAW_ARTIFACT_IMAGES_DIR / tour_id / artifact_id).resolve()

    if not images_dir.exists():
        raise ValueError(f"No uploaded images found for {tour_id}/{artifact_id}")

    images = sorted(images_dir.glob("*.jpg"))

    if len(images) < 3:
        raise ValueError(
            f"Not enough images found in {images_dir} "
            f"(need at least 3 for photogrammetry, found {len(images)})"
        )

    cache_dir = (MESHROOM_CACHE_DIR / tour_id / artifact_id).resolve()
    obj_path = run_meshroom(images_dir, cache_dir)

    glb_path = (ARTIFACT_DIR / tour_id / f"{artifact_id}.glb").resolve()
    convert_obj_to_glb(obj_path, glb_path)

    return glb_path


@app.route("/panoramas/<tour_id>/<filename>")
def get_panorama(tour_id, filename):
    return send_from_directory(PROCESSED_DIR / tour_id, filename)


@app.route("/artifacts/<tour_id>/<filename>")
def get_artifact_model(tour_id, filename):
    return send_from_directory(ARTIFACT_DIR / tour_id, filename)


@app.get("/api/tours/<tour_id>/manifest")
def get_manifest(tour_id):
    manifest_path = TOURS_DIR / f"{tour_id}.json"

    if not manifest_path.exists():
        manifest = build_tour_manifest(tour_id)

        if manifest is None:
            return jsonify({"error": "Tour not found"}), 404

        return jsonify(manifest)

    with open(manifest_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.get("/api/artifacts/<artifact_id>")
def get_artifact_metadata(artifact_id):
    return jsonify(load_artifact_metadata(artifact_id))


@app.post("/api/process/<tour_id>/<waypoint_id>")
def process_waypoint(tour_id, waypoint_id):
    # Stitching can take a while, so kick it off in the background and
    # return immediately — the robot (or whatever called this) shouldn't
    # have to sit on the connection waiting for Hugin to finish. Poll
    # /api/status/<tour_id>/waypoint/<waypoint_id> for progress.
    thread = threading.Thread(
        target=run_waypoint_job,
        args=(tour_id, waypoint_id),
        daemon=True,
    )
    thread.start()

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

    filename = f"{image_index:04d}.jpg"

    image.save(target / filename)

    return jsonify({"status": "ok"})


@app.post("/api/upload-artifact-image")
def upload_artifact_image():
    tour_id = request.form["tour_id"]
    artifact_id = request.form["artifact_id"]
    image_index = int(request.form["image_index"])

    image = request.files["image"]

    target = RAW_ARTIFACT_IMAGES_DIR / tour_id / artifact_id
    target.mkdir(parents=True, exist_ok=True)

    filename = f"{image_index:04d}.jpg"

    image.save(target / filename)

    return jsonify({"status": "ok"})


@app.post("/api/process-artifact/<tour_id>/<artifact_id>")
def process_artifact_endpoint(tour_id, artifact_id):
    # Same deal as waypoint processing: Meshroom + Blender can easily take
    # several minutes per artifact, so this returns immediately and runs
    # the pipeline in the background. Poll
    # /api/status/<tour_id>/artifact/<artifact_id> to know when it's done.
    thread = threading.Thread(
        target=run_artifact_job,
        args=(tour_id, artifact_id),
        daemon=True,
    )
    thread.start()

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
    return jsonify({
        "status": "running",
        "service": "Heritage Sentinel Image Processor"
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True)