from collections import namedtuple

import numpy as np

from pymclevel import MCInfdevOldLevel, BoundingBox
from village_generation.interpreter.interpreter import RawLevelChunkInterpreter

Coord = namedtuple('Coord', 'x z y')


def build_level_array(level, box):
	# type: (MCInfdevOldLevel, BoundingBox) -> np.ndarray
	interpreter = RawLevelChunkInterpreter()
	array = interpreter.interpret(level, box)
	return array


def export_level(level, box, new_level_array, level_array):
	# type: (MCInfdevOldLevel, BoundingBox, np.ndarray, np.ndarray) -> None
	changes = np.where(new_level_array != level_array)
	for array_coord in zip(*changes):
		coord = Coord(box.minx + array_coord[0],
					   		box.minz + array_coord[1],
					   		box.miny + array_coord[2])
		block = new_level_array[array_coord]
		level.setBlockAt(x=coord.x, z=coord.z, y=coord.y, blockID=block)
		level.setBlockDataAt(x=coord.x, z=coord.z, y=coord.y, newdata=0)
