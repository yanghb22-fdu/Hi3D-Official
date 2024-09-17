"""Blender script to render images of 3D models.

** Modified version of SyncDreamer to fit Hi3D dataset needs. 

This version is much faster since it doesn't need to recreate the blender on each model switch.
It can also use the bpy standalone library directly without the need to install the whole blender application.
This can be runned directly on the training machine (tested on a h100) to save time on generating the dataset.

pip install bpy==4.1 fake-bpy-module-4.1

This script is used to render images of 3D models. 
It takes a path containing unsorted .glb/.usdz/.fbx files and renders images of each model. 
Make sure every models have a unique name. The images are from rotating theobject around the origin. 
The images are saved to the output directory.

Basic Usage (see script args for more render options):

python blender_script.py -- --models_path [models_path] --output_path [output_path] 
        
"""

import argparse
import math
import os
from fnmatch import fnmatch
import sys
from pathlib import Path

import numpy as np

import bpy
from mathutils import Vector

from contextlib import contextmanager

parser = argparse.ArgumentParser()
parser.add_argument("--models_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--num_images", type=int, default=16) #per elevation
parser.add_argument("--distance", type=float, default=1.5)
parser.add_argument("--elevation_start", type=float, default=-10)
parser.add_argument("--elevation_end", type=float, default=50)
parser.add_argument("--elevation_step", type=float, default=10)
parser.add_argument("--device", type=str, default='CUDA')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

scene = bpy.data.scenes.new("New Scene")
bpy.context.window.scene = scene

cam_data = bpy.data.cameras.new(name='CameraData')
cam_data.lens = 35
cam_data.sensor_width = 32

cam = bpy.data.objects.new('CameraObject', cam_data)
cam.location = (0, args.distance, 0)

scene.camera = cam

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

bpy.context.scene.collection.objects.link(cam)

render = scene.render
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 1024
render.resolution_y = 1024
render.resolution_percentage = 100
render.film_transparent = True

cycles = scene.cycles
cycles.device = "GPU"
cycles.samples = 128
cycles.diffuse_bounces = 1
cycles.glossy_bounces = 1
cycles.transparent_max_bounces = 3
cycles.transmission_bounces = 3
cycles.filter_width = 0.01
cycles.use_denoising = True
cycles.film_transparent = True
cycles.tile_size = 1024

world = bpy.data.worlds.new("World")
world.use_nodes = True

world_nodes = world.node_tree.nodes
world_nodes.clear()

env_light = 0.5
background_node = world_nodes.new(type='ShaderNodeBackground')
background_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
background_node.inputs['Strength'].default_value = 1.0

world_output = world_nodes.new(type='ShaderNodeOutputWorld')
world_links = world.node_tree.links
world_links.new(background_node.outputs['Background'], world_output.inputs['Surface'])

scene.world = world

bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device # or "OPENCL"

distances = np.asarray([args.distance for _ in range(args.num_images)])
azimuths = (np.arange(args.num_images)/args.num_images*np.pi*2).astype(np.float32)

@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()
    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K

def reset_scene():
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
        
def load_model(object_path: str) -> None:
    reset_scene()
    
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".usdz"):
        bpy.ops.wm.usd_import(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def save_images(object_file: str) -> None:
    load_model(object_file)
    
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    img_number = 0
    object_uid = os.path.basename(object_file).split(".")[0]
    (Path(args.output_path) / object_uid).mkdir(exist_ok=True, parents=True)
    
    for r in range(args.elevation_start, args.elevation_end, args.elevation_step):
        elevations = np.deg2rad(np.asarray([r] * args.num_images).astype(np.float32))    
        cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
        for i in range(args.num_images):
            cam.location = cam_pts[i]
            render_path = os.path.join(args.output_path, object_uid, f"{img_number:03d}.png")
            img_number += 1
            if not os.path.exists(render_path):
                scene.render.filepath = os.path.abspath(render_path)
                with stdout_redirected():
                    bpy.ops.render.render(animation=False, write_still=True)

if __name__ == "__main__":
    c_model = 1
    types = ('*.fbx', '*.glb', '*.usd*')
    files = [f.path for f in os.scandir(args.models_path) if any(fnmatch(f, p) for p in types)]
    f_number = len(files)
    
    for file in files:
        print('Processing: {} ({}/{})'.format(file, c_model, f_number))
        save_images(file)
        c_model += 1