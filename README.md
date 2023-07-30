# Dagor Asset Explorer Tools

Importer for Dagor Model Files (DMF) and Dagor Prop Layout (DPL) files generated by the Dagor Asset Explorer.

*Currently only officially supports Blender 2.92*

Installation instructions are available in the [Releases tab](https://github.com/Gredwitch/Dagor-Asset-Explorer-Tools/releases).

<div align="center">
	<img src="https://i.imgur.com/ZMqcL1z.png" alt="Screenshot of the main window in action" width="80%" />
</div>

## Features

- Skeleton support
- Model scaling to the Source engine
- Disabled auto-save when loading DPL (so you don't end up with 10GB large temporary files)
- Automatic shader graph generation
	- Pseudo-PBR magic: 50% of the normal's blue channel is multiplied onto the diffuse's alpha channel
	- Inverts the alpha channel whe necessary for camo support
	- Takes advantage of the normal map suitably:
		1) Turns the blue channel into solid white
		2) Recomposes the normal map into AGB
		3) Inverts the alpha channel and sends it to the *Specular* input of the Principled BSFD
		4) Sends the untouched blue channel into the *Metallic* input of Princinpled BSFD

<br/>

<div align="center">
	<img src="https://i.imgur.com/7AKhQGL.png" alt="Screenshot of the main window in action" width="90%" />
</div>