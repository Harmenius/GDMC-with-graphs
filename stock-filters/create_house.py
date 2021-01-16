import numpy as np

from pymclevel import materials, BoundingBox, MCInfdevOldLevel

template = np.array(
	[
		[
			[1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		]
	] * 2
	+
	[
		[
			[1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
		],
		[
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		],
		[
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		],
		[
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		],
		[
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		],
	]
)


def perform(level, box, options):
	"""Build a house in the box

	Args:
		level (MCInfdevOldLevel): Full level to generate the house
		box (BoundingBox): Box that indicates where the house will be built
		options (dict): Options given to the Filter from MCEdit2
	"""
	material = materials.alphaMaterials.WoodPlanks

	for x in range(template.shape[2]):
		xw = x + box.minx
		for z in range(template.shape[1]):
			zw = z + box.minz
			for y in range(template.shape[0]):
				yw = y + box.miny
				level.setBlockAt(xw, yw, zw, material.ID * template[y, z, x])
				level.setBlockDataAt(xw, yw, zw, 0)
