import numpy as np
from typing import Tuple, cast

from pymclevel import materials
from village_generation.conversion.np_mc import build_level_array, export_level
from village_generation.edit_level.edit_level import place_house
from village_generation.support.utils import traverse_diagonally


displayName = "Build House"
inputs = (
	("Build House", "label"),
	("Creator: Pluralis", "label"),
)

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
).transpose([2,1,0])  # Transpose from y,x,z (readable template) to x,z,y (canon coordinate order)


def perform(level, box, options):
	"""Build a house in the box

    Args:
        level (MCInfdevOldLevel): Full level to generate the house
        box (BoundingBox): Box that indicates where the house will be built
        options (dict): Options given to the Filter from MCEdit2
    """
	level_array = build_level_array(level, box)

	house_slice = cast(Tuple[slice, slice, slice], tuple(slice(0, s) for s in template.shape))
	new_level_array = place_house(level_array.copy(), template, house_slice, materials.alphaMaterials.WoodPlanks.ID)

	export_level(level, box, new_level_array, level_array)


def _determine_house_height(surface_height_map, house_slice):
	# type: (np.ndarray, Tuple[slice, slice]) -> int
	heights = surface_height_map[house_slice[0].start:house_slice[0].stop,
			  house_slice[1].start:house_slice[1].stop]
	return heights.min()


def fill_with_houses(level_array, build_map, surface_height_map):
	# type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
	"""Place houses in the level and box within the area indicated by build map at height of surface_height_map

	Args:
		level_array: 3D np.array(int) representing the blocks in a (part of) a minecraft level
		build_map: 2D np.array(bool) with each coordinate denoting a 16x16 square in level_array
        	Indicate whether this is the area where a house can be built
		surface_height_map: 2D np.array(int) same size as level_arrays X and Z
            Indicate the surface height of every column. Used to determine the placement height of a house

    Returns: a new 3D np.array(int) same size as level_array, the same as level_array but with houses
    """
	new_level_array = level_array.copy()

	subrectangle = __grab_subrectangle(build_map)

	for chunk_coord in traverse_diagonally(subrectangle):
		if build_map[chunk_coord[0], chunk_coord[1]]:
			coord = chunk_coord[0] << 4, chunk_coord[1] << 4  # Builds houses at the north-western corner of a chunk
			house_slice = slice(coord[0], coord[0] + template.shape[0]), \
						  slice(coord[1], coord[1] + template.shape[1])
			floor = _determine_house_height(surface_height_map, house_slice)
			new_level_array = place_house(new_level_array,
										  template,
										  house_slice + (slice(floor, floor + template.shape[2]),))
	return new_level_array


def __grab_subrectangle(build_map):
	# type: (np.ndarray) -> Tuple[int, int, int, int]
	"""Grab a rectangle from build_map which covers all True coordinates to improve performance"""
	lr = build_map.any(1)
	x_min, x_max = lr.argmax(), len(lr) - 1 - lr[::-1].argmax()
	td = build_map.any(0)
	y_min, y_max = td.argmax(), len(td) - 1 - td[::-1].argmax()
	return x_min, x_max, y_min, y_max
