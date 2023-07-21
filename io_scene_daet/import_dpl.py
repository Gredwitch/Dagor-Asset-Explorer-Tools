import bpy
import bmesh
from os import path
from io import BufferedReader, BytesIO
from struct import unpack
from math import radians
from mathutils import Matrix, Vector
from .common import readInt, readString, get_file_name
from traceback import format_exc
from .import_dmf import load as load_dmf

MERGE_DISTANCE = 0.001

def get_assets_dir(filepath:str):
	return path.join(path.dirname(filepath), get_file_name(filepath) + "_props")


def optimize_object(ob:bpy.types.Object, decimate_ratio:float):
	bpy.context.view_layer.objects.active = ob
	
	bpy.ops.object.mode_set(mode = "EDIT")
	
	bm = bmesh.from_edit_mesh(ob.data)
	bmesh.ops.remove_doubles(bm, verts = bm.verts, dist = MERGE_DISTANCE)
	bmesh.update_edit_mesh(ob.data)
	
	bpy.ops.object.mode_set(mode = "OBJECT")

	if decimate_ratio != 1.0:
		decimateMod = ob.modifiers.new(name = "Decimate", type = "DECIMATE")
		decimateMod.ratio = decimate_ratio
		
		bpy.ops.object.modifier_apply(modifier = decimateMod.name)

def optimize_objects(objects:list, decimate_ratio:float):
	for k, ob in enumerate(objects):
		ob:bpy.types.Object

		if ob.type == "MESH":
			print(f"[D] \tOptimizing {k + 1}/{len(objects)}: {ob.name}")

			optimize_object(ob, decimate_ratio)


def place_instance(objects:list, pos:tuple, matrix:tuple):
	for ob in objects:
		ob:bpy.types.Object

		if ob.parent is not None: # only modify the armature's pos and matrix
			continue

		new_object:bpy.types.Object = ob.copy()
		new_object.data = ob.data.copy()

		# new_object.scale.x = -new_object.scale.x
		# new_object.rotation_euler = -OBJECT_ROT
		
		new_object.matrix_world = Matrix(matrix)
		new_object.location = Vector(pos)

		# new_object.scale.x = new_object.scale.x
		# new_object.rotation_euler += OBJECT_ROT
		
		bpy.context.collection.objects.link(new_object)

def remove_objects(objects:list):
	for ob in objects:
		bpy.data.objects.remove(ob)

def get_pos_matrix(f:BufferedReader):
	pos = unpack("fff", f.read(0xC))

	matrix = (
		unpack("ffff", f.read(0x10)),
		unpack("ffff", f.read(0x10)),
		unpack("ffff", f.read(0x10)),
		unpack("ffff", f.read(0x10))
	)

	return pos, matrix

def skip_model(f:BufferedReader, pos_matrix_cnt:int):
	f.seek(0x4C * pos_matrix_cnt, 1)


def load(filepath:str, 
	 max_occurences_cnt:int = 10000,
	 skip_max_occurences:bool = True,
	 optimize_models:bool = True,
	 random_viewport_color:bool = True, 
	 recreate_materials:bool = False, 
	 import_textures:bool = True,
	 disable_autosave:bool = True,
	 decimate_ratio:float = 0.4):

	if disable_autosave:
		pref = bpy.context.preferences
		
		pref.use_preferences_save = False
		pref.filepaths.use_auto_save_temporary_files = False
	
	assets_dir = get_assets_dir(filepath)

	view_layer = bpy.context.view_layer
	
	with open(filepath, "rb") as file:
		# if file.read(4) != DMF_MAGIC:
		# 	raise Exception("Invalid magic, not a DMF?")
		# else:
		# 	file.seek(0, 0)
		
		f = BytesIO(file.read())
		# f.seek(4, 1)
		
		ent_cnt = readInt(f)

		for k in range(ent_cnt):
			idx = k + 1

			ent_name = readString(f)
			pos_matrix_cnt = readInt(f)

			ent_path = path.join(assets_dir, ent_name + "_0.dmf")

			print(f"[D] ({idx}/{ent_cnt}) ", end = "")

			if pos_matrix_cnt == 0:
				print(f"Skipping {ent_name}")
		
				continue
			elif not path.exists(ent_path):
				print(f"Skipping {ent_name}: model file does not exist")

				skip_model(f, pos_matrix_cnt)

				continue
			elif pos_matrix_cnt >= max_occurences_cnt and skip_max_occurences:
				print(f"Skipping {ent_name}: model has {pos_matrix_cnt} occurences (max {max_occurences_cnt})")

				skip_model(f, pos_matrix_cnt)

				continue
			
			
			print(f"Loading entity {ent_name}")
			
			selected_objects = None
			
			try:
				load_dmf(ent_path, 
	     				apply_scale = False, 
						random_viewport_color = random_viewport_color,
						recreate_materials = recreate_materials,
						import_textures = import_textures,
						update_viewlayer = False)
				
				selected_objects = bpy.context.selected_objects
			except Exception as e:
				print(f"[E] \tImport failed: {e}")
				format_exc()

				skip_model(f, pos_matrix_cnt)

				continue
			
			if optimize_models:
				optimize_objects(selected_objects, decimate_ratio)
	
			for i in range(pos_matrix_cnt):
				print(f"[D] \tProcessing {i + 1}/{pos_matrix_cnt} ({idx}/{ent_cnt})") 
				
				pos, matrix = get_pos_matrix(f)

				place_instance(selected_objects, pos, matrix)

				if i >= max_occurences_cnt:
					print(f"[D] \tSkipping other occurences (max occurences reached)")

					skip_model(f, max_occurences_cnt - i) 
			
			bpy.ops.object.select_all(action='DESELECT')

	view_layer.update()


	return {"FINISHED"}