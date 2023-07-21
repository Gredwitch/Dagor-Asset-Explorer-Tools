
import bpy
from os import path
from io import BufferedReader

LITTLE = "little"

def readInt(f:BufferedReader):
	return int.from_bytes(f.read(0x4), byteorder = LITTLE)

def readString(f:BufferedReader):
	return f.read(readInt(f)).decode()

def get_file_name(filepath:str):
	return path.splitext(path.basename(filepath))[0]

def deselect_all():
	for ob in bpy.context.selected_objects:
		ob.select_set(False)