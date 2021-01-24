"""Functions for editing the level_array"""
import itertools

import numpy as np
from typing import Tuple

from pymclevel import materials


def set_chunk_height_with_bricks(agg_height_map, level_array):
	# type: (np.ndarray, np.ndarray) -> np.ndarray
	new_level_array = level_array.copy()
	for xc, zc in itertools.product(xrange(agg_height_map.shape[0]), xrange(agg_height_map.shape[1])):
		y = int(agg_height_map[xc, zc])
		x, z = (xc << 4), (zc << 4)
		new_level_array[x:x+16, z:z+16, y] = materials.alphaMaterials.Brick.ID
	return new_level_array


def place_house(level_array, template, house_slice, material = materials.alphaMaterials.WoodPlanks.ID):
	# type: (np.ndarray, np.ndarray, Tuple[slice, slice, slice], int) -> np.ndarray
	level_array[house_slice] = template * material
	return level_array
