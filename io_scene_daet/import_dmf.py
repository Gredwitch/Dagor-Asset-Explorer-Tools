import bpy
from os import path
from io import BufferedReader, BytesIO
from struct import unpack
from math import radians
from mathutils import Matrix, Vector
from .common import get_file_name, readInt, readString, deselect_all
from random import random

DMF_MAGIC = b"DMF\0"
DMF_NO_PARENT = 0

MATRIX_TRANSLATE_ZERO = Matrix.Translation((0, 0, 0)).to_translation()
MATRIX_TRANSLATE_ONE = Matrix.Translation((0, 0, 1 / 40)).to_translation()

OBJECT_SCALE = Vector((-40, 40, 40))
OBJECT_ROT = Vector((radians(90), radians(0), radians(-90)))
# OBJECT_ROT = Vector((radians(0), radians(0), radians(0)))

TEXTURE_EXTENSION = ".dds"

def get_texture_dir(filepath:str):
	return path.join(path.dirname(filepath), "textures")

def get_texture_path(texture_dir:str, filename:str):
	return path.join(texture_dir, filename)

def get_texture(texture_dir:str, texture_name:str):
	filename = texture_name.split("*")[0] + TEXTURE_EXTENSION
	
	image = bpy.data.images.get(filename)

	if image is None:
		texture_path = get_texture_path(texture_dir, filename)
		
		if path.exists(texture_path):
			image = bpy.data.images.load(texture_path)
	
	return image

class Skeleton:
	def __init__(self, skeleton_name:str):
		self.__bones_by_name:dict[str, tuple[bpy.types.Bone, Matrix]] = {}

		self.__armature = bpy.data.armatures.new(f"{skeleton_name}_Armature")
		self.__armature_obj = bpy.data.objects.new(f"{skeleton_name}_ArmatureObject", self.__armature)
	
	def create_bones(self, f:BufferedReader):
		armature = self.__armature
		armature_obj = self.__armature_obj

		bpy.context.collection.objects.link(armature_obj)
		bpy.context.view_layer.objects.active = armature_obj
		armature_obj.select_set(True)

		bpy.ops.object.mode_set(mode='EDIT')

		node_count = readInt(f)

		bones:list[bpy.types.Bone] = []
		parents:list[bpy.types.Bone] = []

		for idx in range(node_count):
			parent_idx = readInt(f) - 1
			name = readString(f)
			wtm = Matrix((
				unpack("4f", f.read(0x10)),
				unpack("4f", f.read(0x10)),
				unpack("4f", f.read(0x10)),
				unpack("4f", f.read(0x10))
			))

			
			bone = armature.edit_bones.new(name)
			bone.head = wtm @ MATRIX_TRANSLATE_ZERO
			bone.tail = wtm @ MATRIX_TRANSLATE_ONE

			bones.append(bone)
			parents.append(parent_idx)

			self.__bones_by_name[name] = (bone, wtm)
		
		for idx, parent_idx in enumerate(parents):
			if parent_idx != -1:
				bone = bones[idx]
				parent_bone = bones[parent_idx]

				bone.parent = parent_bone
		
		bpy.ops.object.mode_set(mode='OBJECT')
	
	
	def getBone(self, name:str):
		if not self.hasBone(name):
			return None
		
		return self.__bones_by_name[name]

	def hasBone(self, name:str):
		return name in self.__bones_by_name

	@property
	def armature(self):
		return self.__armature
	
	@property
	def armature_obj(self):
		return self.__armature_obj


def load_skeleton(f:BufferedReader, ofs:int, name:str):
	if ofs != 0:
		f.seek(ofs, 0)

		skeleton = Skeleton(name)
		skeleton.create_bones(f)

		return skeleton
	else:
		return None

def set_skeleton_transform(skeleton:Skeleton):
	skeleton.armature_obj.scale = OBJECT_SCALE
	skeleton.armature_obj.rotation_euler = OBJECT_ROT


def create_camo_node(nodes:bpy.types.Nodes, node_tree:bpy.types.NodeTree):
	texcoor_node = nodes.new("ShaderNodeTexCoord")
	mapping_node = nodes.new("ShaderNodeMapping")
	node_tree.links.new(texcoor_node.outputs["UV"], mapping_node.inputs["Vector"])

	camo_node = nodes.new("ShaderNodeTexImage")

	node_tree.links.new(mapping_node.outputs["Vector"], camo_node.inputs["Vector"])

	return camo_node

def create_materials(f:BufferedReader, 
		     texture_dir:str, 
			 random_viewport_color:bool, 
			 recreate_materials:bool,
			 import_textures:bool):
	material_count = readInt(f)
	
	for _ in range(material_count):
		material_name = readString(f)
		material_class = readString(f)

		masked = material_class.find("masked") != -1

		two_sided = readInt(f) == 1
		diff = unpack("ffff", f.read(0x10))
		amb = unpack("ffff", f.read(0x10))
		emis = unpack("ffff", f.read(0x10))
		spec = unpack("ffff", f.read(0x10))

		diffuse = readString(f)
		normal = readString(f)
		ambient_occlusion = readString(f)
		mask = readString(f)

		detail = [readString(f) for _ in range(readInt(f))]
		detail_normal = [readString(f) for _ in range(readInt(f))]

		params = {}
		param_count = readInt(f)

		for _ in range(param_count):
			param_key = readString(f)
			param_value = readString(f)

			params[param_key] = param_value
		
		# print(f"{material_name}:{material_class}")
		# print(f"\tdiffuse={diffuse}")
		# print(f"\tnormal={normal}")
		# print(f"\tao={ambient_occlusion}")
		# print(f"\tmask={mask}")
		# print(f"\tdetail={detail}")
		# print(f"\tdetail_normal={detail_normal}")
		# print(f"\tparams={params}")

		if bpy.data.materials.get(material_name) is not None and not recreate_materials:
			continue

		if material_class == "rendinst_layered":
			if len(detail) >= 2:
				masked = True

				diffuse = detail[0]
				mask = detail[1]

				if len(detail_normal) >= 1:
					normal = detail_normal[1]

					# TODO: mask normal?

		material = bpy.data.materials.new(name=material_name)

		if random_viewport_color:
			material.diffuse_color = (random(), random(), random(), 1)

		material.use_nodes = True
		# material.use_backface_culling = two_sided

		node_tree = material.node_tree
		nodes = node_tree.nodes

		nodes.clear()

		material_output = nodes.new("ShaderNodeOutputMaterial")
		principled_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
		principled_bsdf.inputs["Base Color"].default_value = diff
		principled_bsdf.inputs["Emission"].default_value = emis
		principled_bsdf.inputs["Roughness"].default_value = 1.0

		diffuse_texture = get_texture(texture_dir, diffuse) if import_textures else None
		
		diffuse_node = nodes.new("ShaderNodeTexImage")
		if diffuse_texture is not None:
			diffuse_node.image = diffuse_texture
		

		node_tree.links.new(diffuse_node.outputs["Color"], principled_bsdf.inputs["Base Color"])
		
		normal_texture = get_texture(texture_dir, normal) if import_textures else None
		

		camo_node = None
		
		if masked:
			camo_node = create_camo_node(nodes, node_tree)
			camo_texture = get_texture(texture_dir, mask) if import_textures else None
			
			if camo_texture is not None:
				camo_node.image = camo_texture
		
		# if normal_texture is not None:
		normal_node = nodes.new("ShaderNodeTexImage")
		
		if normal_texture is not None:
			normal_node.image = normal_texture

		color_node = nodes.new("ShaderNodeValToRGB")
		color_node.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
		color_node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

		separate_rgb_node = nodes.new("ShaderNodeSeparateRGB")
		node_tree.links.new(normal_node.outputs["Color"], separate_rgb_node.inputs["Image"])
		node_tree.links.new(separate_rgb_node.outputs["B"], principled_bsdf.inputs["Metallic"])

		combine_rgb_node = nodes.new("ShaderNodeCombineRGB")
		node_tree.links.new(normal_node.outputs["Alpha"], combine_rgb_node.inputs["R"])
		node_tree.links.new(separate_rgb_node.outputs["G"], combine_rgb_node.inputs["G"])
		node_tree.links.new(color_node.outputs["Color"], combine_rgb_node.inputs["B"])
		
		normalmap_node = nodes.new("ShaderNodeNormalMap")
		node_tree.links.new(combine_rgb_node.outputs["Image"], normalmap_node.inputs["Color"])
		node_tree.links.new(normalmap_node.outputs["Normal"], principled_bsdf.inputs["Normal"])

		invert_node = nodes.new("ShaderNodeInvert")
		node_tree.links.new(normal_node.outputs["Alpha"], invert_node.inputs["Color"])
		node_tree.links.new(invert_node.outputs["Color"], principled_bsdf.inputs["Specular"])

		if masked:
			invert_node = nodes.new("ShaderNodeInvert")
			node_tree.links.new(separate_rgb_node.outputs["B"], invert_node.inputs["Color"])

			div_node = nodes.new("ShaderNodeMath")
			div_node.operation = "DIVIDE"
			
			node_tree.links.new(invert_node.outputs["Color"], div_node.inputs[0])
			div_node.inputs[1].default_value = 2.0

			mult_node = nodes.new("ShaderNodeMath")
			mult_node.operation = "MULTIPLY"
			node_tree.links.new(div_node.outputs["Value"], mult_node.inputs[0])
			node_tree.links.new(diffuse_node.outputs["Alpha"], mult_node.inputs[1])

			invert2_node = nodes.new("ShaderNodeInvert")
			node_tree.links.new(mult_node.outputs["Value"], invert2_node.inputs["Color"])
		# elif masked:
		# 	invert2_node = nodes.new("ShaderNodeInvert")
		# 	node_tree.links.new(diffuse_node.outputs["Alpha"], invert_node.inputs["Color"])

		if masked:
			mix_node = nodes.new("ShaderNodeMixShader")
			node_tree.links.new(invert2_node.outputs["Color"], mix_node.inputs["Fac"])
			node_tree.links.new(camo_node.outputs["Color"], mix_node.inputs[1])
			node_tree.links.new(principled_bsdf.outputs["BSDF"], mix_node.inputs[2])

			node_tree.links.new(mix_node.outputs["Shader"], material_output.inputs["Surface"])
		else:
			node_tree.links.new(diffuse_node.outputs["Alpha"], principled_bsdf.inputs["Alpha"])
			
			node_tree.links.new(principled_bsdf.outputs["BSDF"], material_output.inputs["Surface"])

		if material_class.find("atest") != -1:
			... # TODO: make atest work


def load_materials(f:BufferedReader, 
		   ofs:int, 
		   texture_dir:str, 
		   random_viewport_color:bool, 
		   recreate_materials:bool,
		   import_textures:bool):
	if ofs != 0:
		f.seek(ofs, 0)

		create_materials(f, 
		   texture_dir, 
		   random_viewport_color, 
		   recreate_materials,
		   import_textures)



class VertexRemapper:
	def __init__(self, verts:tuple, uvs:tuple):
		self.vert_remap:dict[int, int] = {}
		self.verts = []
		self.uvs = []
		self.__vert_idx = 0

		self.__verts = verts
		self.__uvs = uvs
	
	def get_remapped_idx(self, idx:int):
		if not idx in self.vert_remap:
			self.vert_remap[idx] = self.__vert_idx
			self.verts.append(self.__verts[idx])
			self.uvs.append(self.__uvs[idx])

			self.__vert_idx += 1
		
		return self.vert_remap[idx]
	
	def get_remapped_face(self, face:tuple):
		return (self.get_remapped_idx(face[0]), 
	  			self.get_remapped_idx(face[1]), 
				self.get_remapped_idx(face[2]))
	
	def remap_object(self, faces:tuple):
		return tuple(self.get_remapped_face(face) for face in faces), self.verts, self.uvs


def create_mesh(f:BufferedReader, name:str, global_verts:tuple, global_uvs:tuple):
	face_count = readInt(f)
	faces = tuple(unpack("III", f.read(0xC)) for _ in range(face_count))

	remapper = VertexRemapper(global_verts, global_uvs)
	
	obj_faces, obj_verts, obj_uvs = remapper.remap_object(faces)

	mesh = bpy.data.meshes.new(name)
	mesh.from_pydata(obj_verts, [], obj_faces)
	
	uv_layer = mesh.uv_layers.new().data
	for loop in mesh.loops:
		uv_layer[loop.index].uv = obj_uvs[loop.vertex_index]
	
	return mesh

def assign_mesh_materials(f:BufferedReader, mesh:bpy.types.Mesh):
	mat_count = readInt(f)
	
	prev_idx = 0
	prev_mat = None

	for i in range(mat_count):
		start_idx = readInt(f)
		mat_name = readString(f)

		material = bpy.data.materials.get(mat_name)

		if material is None:
			material = bpy.data.materials.new(mat_name)
		
		mesh.materials.append(material)

		assign_material(mesh, prev_mat, prev_idx, start_idx)

		prev_idx = start_idx
		prev_mat = i
	
	assign_material(mesh, prev_mat, prev_idx, len(mesh.polygons))

def assign_material(mesh:bpy.types.Mesh, material:bpy.types.Material, start_idx:int, end_idx:int):
	if material is not None:
		for k in range(start_idx, end_idx):
			mesh.polygons[k].material_index = material

def create_object(f:BufferedReader, global_verts:tuple, global_uvs:tuple):
	obj_name = readString(f)
	
	mesh = create_mesh(f, obj_name, global_verts, global_uvs)
	assign_mesh_materials(f, mesh)
	
	mesh.validate()
	mesh.update()
	
	ob = bpy.data.objects.new(obj_name, mesh)
	
	return ob

def set_object_transform(ob:bpy.types.Object, skeleton:Skeleton, apply_scale:bool):
	obj_name = ob.name
	
	if skeleton is not None:
		armature_obj = skeleton.armature_obj

		if skeleton.hasBone(obj_name):
			armature = skeleton.armature

			bone, wtm = skeleton.getBone(obj_name)
			ob.matrix_world = wtm


			bpy.ops.object.mode_set(mode='OBJECT')
			deselect_all()

			armature_obj.select_set(True)
			bpy.context.view_layer.objects.active = armature_obj
			
			bpy.ops.object.mode_set(mode='EDIT')
			armature.edit_bones.active = armature.edit_bones[obj_name]
			
			bpy.ops.object.mode_set(mode='OBJECT')

			deselect_all()
			ob.select_set(True)
			armature_obj.select_set(True)
			bpy.context.view_layer.objects.active = armature_obj

			bpy.ops.object.parent_set(type='BONE', keep_transform=True)
		else:
			ob.parent = armature_obj
	elif apply_scale:
		ob.scale = OBJECT_SCALE
		ob.rotation_euler = OBJECT_ROT



def load(filepath:str, 
	 apply_scale:bool = True, 
	 random_viewport_color = True, 
	 recreate_materials = False, 
	 import_textures = True,
	 update_viewlayer = True):
	model_name = get_file_name(filepath)
	texture_dir = get_texture_dir(filepath)

	view_layer = bpy.context.view_layer
	collection = view_layer.active_layer_collection.collection
	
	with open(filepath, "rb") as file:
		if file.read(4) != DMF_MAGIC:
			raise Exception("Invalid magic, not a DMF?")
		else:
			file.seek(0, 0)
		
		f = BytesIO(file.read())
		f.seek(4, 1)
		
		data_ofs, mat_ofs, ske_ofs = unpack("III", f.read(0xC))

		load_materials(f, 
		 mat_ofs, 
		 texture_dir, 
		 random_viewport_color, 
		 recreate_materials,
		 import_textures)
		skeleton = load_skeleton(f, ske_ofs, model_name)

		# main part
		
		f.seek(data_ofs, 0)

		vert_scale = unpack("fff", f.read(0xC)) # unused, scale is already applied

		vert_count = readInt(f)

		verts = tuple(unpack("fff", f.read(0xC)) for _ in range(vert_count))
		uvs = tuple(unpack("ff", f.read(0x8)) for _ in range(vert_count))
		
		obj_count = readInt(f)

		if skeleton is None and obj_count > 1:
			skeleton = Skeleton(model_name) # create an empty skeleton to append our dynmodel's objects to
			# if we are a rendinst then do not make a skeleton

		for _ in range(obj_count):
			ob = create_object(f, verts, uvs)
			
			collection.objects.link(ob)
			ob.select_set(True)
			
			if skeleton is not None:
				set_object_transform(ob, skeleton, apply_scale)

		if skeleton is not None and apply_scale:
			set_skeleton_transform(skeleton)

	if update_viewlayer:
		view_layer.update()


	return {"FINISHED"}