"""
Run headless via:
    blender -b --python obj_to_glb.py -- <input.obj> <output.glb>

Imports the textured OBJ that Meshroom produces and exports it as GLB.
"""

import sys
import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) != 2:
    raise SystemExit(
        "Usage: blender -b --python obj_to_glb.py -- <input.obj> <output.glb>"
    )

obj_path, glb_path = argv

# Start from a clean scene (no default cube/camera/light).
bpy.ops.wm.read_factory_settings(use_empty=True)

# NOTE: the OBJ import/export operator names differ between Blender
# versions. Blender 4.0+ renamed the OBJ importer to wm.obj_import;
# Blender <=3.6 uses import_scene.obj. Handle both so this keeps working
# whichever Blender build ends up on the processing box.
if hasattr(bpy.ops.wm, "obj_import"):
    bpy.ops.wm.obj_import(filepath=obj_path)
else:
    bpy.ops.import_scene.obj(filepath=obj_path)

bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format="GLB",
    export_yup=True,
)

print(f"Wrote {glb_path}")
