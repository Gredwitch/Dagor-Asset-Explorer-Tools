
bl_info = {
	"name": "Dagor Asset Explorer Tools",
	"author": "Quentin H.",
	"version": (1, 1, 7),
	"blender": (2, 92, 0),
	"category": "Import",
	"location": "File > Import",
	"wiki_url": "https://github.com/Gredwitch/Dagor-Asset-Explorer-Tools",
	"tracker_url": "https://github.com/Gredwitch/Dagor-Asset-Explorer-Tools/issues",
	"description": "Importer for Dagor Model Files (DMF) and Dagor Prop Layout (DPL) files generated by the Dagor Asset Explorer"
}

import bpy
from bpy.props import StringProperty, IntProperty, BoolProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper

if "bpy" in locals():
	import importlib
	
	if "import_dmf" in locals():
		importlib.reload(import_dmf)
	
	if "import_dpl" in locals():
		importlib.reload(import_dpl) 



class ImportDMF(bpy.types.Operator, ImportHelper):
	bl_idname = "import_scene.dmf"
	bl_label = "Import DMF"
	bl_options = {'UNDO'}
 
	filename_ext = ".dmf"
	filter_glob: StringProperty(
		default = "*.dmf",
		options = {"HIDDEN"}, 
	)
	
	apply_scale: BoolProperty(
		name = "Fix transform",
		description = "Scale model by (-40, 40, 40) and rotate them by (90, 0, -90)",
		default = True,
	)
	
	relative_parenting: BoolProperty(
		name = "Relative parenting",
		description = "Parent objects to 'Bone Relative' instead of 'Bone'",
		default = True,
	)
	
	random_viewport_color: BoolProperty(
		name = "Random viewport color",
		description = "Give each material a random viewport color",
		default = True,
	)

	recreate_materials: BoolProperty(
		name = "Recreate material",
		description = "Create a new material if the name is already taken",
		default = False,
	)

	import_textures: BoolProperty(
		name = "Import textures",
		description = "Import textures from the 'textures' directory relative to the model's directory",
		default = True,
	)

	def execute(self, context):
		from . import import_dmf

		# add the following options:
		#  - recreate_materials
		#  - import_textures

		keywords = self.as_keywords(ignore = ("filter_glob", ))  

		return import_dmf.load(**keywords)

	def draw(self, context):
		pass


class DMF_DPL_PT_import_material(bpy.types.Panel):
	bl_space_type = 'FILE_BROWSER'
	bl_region_type = 'TOOL_PROPS'
	bl_label = "Materials"
	bl_parent_id = "FILE_PT_operator"

	@classmethod
	def poll(cls, context):
		sfile = context.space_data
		operator = sfile.active_operator

		return operator.bl_idname == "IMPORT_SCENE_OT_dmf" or operator.bl_idname == "IMPORT_SCENE_OT_dpl"

	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		layout.use_property_decorate = False  # No animation.

		sfile = context.space_data
		operator = sfile.active_operator

		layout.prop(operator, "random_viewport_color")
		layout.prop(operator, "recreate_materials")
		layout.prop(operator, "import_textures")

class DMF_PT_import_transform(bpy.types.Panel):
	bl_space_type = 'FILE_BROWSER'
	bl_region_type = 'TOOL_PROPS'
	bl_label = "Transform"
	bl_parent_id = "FILE_PT_operator"

	@classmethod
	def poll(cls, context):
		sfile = context.space_data
		operator = sfile.active_operator

		return operator.bl_idname == "IMPORT_SCENE_OT_dmf"

	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		layout.use_property_decorate = False  # No animation.

		sfile = context.space_data
		operator = sfile.active_operator

		layout.prop(operator, "apply_scale")
		layout.prop(operator, "relative_parenting")



class ImportDPL(bpy.types.Operator, ImportHelper):
	bl_idname = "import_scene.dpl"
	bl_label = "Import DPL"
	bl_options = {"UNDO"}
 
	filename_ext = ".dpl"
	filter_glob: StringProperty(
		default = "*.dpl",
		options = {"HIDDEN"}, 
	)

	
	random_viewport_color: BoolProperty(
		name = "Random viewport color",
		description = "Give each material a random viewport color",
		default = True,
	)

	recreate_materials: BoolProperty(
		name = "Recreate material",
		description = "Create a new material if the name is already taken",
		default = False,
	)

	import_textures: BoolProperty(
		name = "Import textures",
		description = "Import textures from the 'textures' directory relative to the model's directory",
		default = True,
	)

	
	max_occurences_cnt: IntProperty(
		name = "Max instance occurences",
		description = "Do not create instances above this limit",
		min = 2, max = 1_000_000_000,
		soft_min = 2, soft_max = 1_000_000_000,
		default = 10_000
	)

	skip_max_occurences: BoolProperty(
		name = "Skip max occurences",
		description = "Skip entirely models that exceed the occurences limit",
		default = True,
	)

	disable_autosave: BoolProperty(
		name = "Disable auto-save",
		description = "Disable auto-saving the current project",
		default = True,
	)

	optimize_models: BoolProperty(
		name = "Optimize models",
		description = "Merge vertices by distance",
		default = True,
	)

	decimate_ratio: FloatProperty(
		name= "Decimate ratio",
		description = "Value used for the decimate modifier. Set to 1 to disable",
		min = 0.0, max = 1.0,
		soft_min = 0.0, soft_max = 1.0,
		default = 0.4
	)

	def execute(self, context):
		from . import import_dpl

		# add the following options:
		#  - import_textures
		#  - optimize_models
 
		keywords = self.as_keywords(ignore = ("filter_glob", ))  

		return import_dpl.load(**keywords) 

	def draw(self, context):
		layout = self.layout

		layout.label(text = "You should kill Blender instead of", icon = "ERROR")
		layout.label(text = "exiting it normaly. Otherwise, huge", icon = "ERROR")
		layout.label(text = "temporary files will be created.", icon = "ERROR")
		
		layout.separator()


class DPL_PT_import_misc(bpy.types.Panel):
	bl_space_type = 'FILE_BROWSER'
	bl_region_type = 'TOOL_PROPS'
	bl_label = "Misc"
	bl_parent_id = "FILE_PT_operator"
	
	@classmethod
	def poll(cls, context):
		sfile = context.space_data
		operator = sfile.active_operator

		return operator.bl_idname == "IMPORT_SCENE_OT_dpl"

	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		layout.use_property_decorate = False  # No animation.

		sfile = context.space_data
		operator = sfile.active_operator

		layout.prop(operator, "max_occurences_cnt")
		layout.prop(operator, "skip_max_occurences")
		layout.prop(operator, "disable_autosave")

class DPL_PT_import_optimize_geometry(bpy.types.Panel):
	bl_space_type = 'FILE_BROWSER'
	bl_region_type = 'TOOL_PROPS'
	bl_label = "Optimize geometry"
	bl_parent_id = "FILE_PT_operator"
	
	@classmethod
	def poll(cls, context):
		sfile = context.space_data
		operator = sfile.active_operator

		return operator.bl_idname == "IMPORT_SCENE_OT_dpl"

	def draw_header(self, context):
		sfile = context.space_data
		operator = sfile.active_operator

		self.layout.prop(operator, "optimize_models", text="")

	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		layout.use_property_decorate = False  # No animation.

		sfile = context.space_data
		operator = sfile.active_operator

		layout.enabled = operator.optimize_models

		layout.prop(operator, "decimate_ratio")



def menu_func_import(self, context):
	self.layout.operator(ImportDMF.bl_idname, text="Dagor Model File (.dmf)")
	self.layout.operator(ImportDPL.bl_idname, text="Dagor Prop Layout (.dpl)")

classes = (
	ImportDMF,
	DMF_DPL_PT_import_material,
	DMF_PT_import_transform,
	ImportDPL,
	DPL_PT_import_misc,
	DPL_PT_import_optimize_geometry
)

  
def register():
	for cls in classes:
		bpy.utils.register_class(cls)

	bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

   
def unregister():
	bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

	for cls in classes:
		bpy.utils.unregister_class(cls)


if __name__ == "__main__":
	register()
