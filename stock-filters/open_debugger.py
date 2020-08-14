import itertools
from math import ceil

import utilityFunctions as utilityFunctions
from mcplatform import *

import numpy as np

from pymclevel import MCLevel, BoundingBox, MCInfdevOldLevel

inputs = (
	("Open Debugger", "label"),
	("Creator: Pluralis", "label"),
	)


def calculate_height_map(level, box):
	chunk_box = box.chunkBox(level)
	height_map = np.zeros((chunk_box.width, chunk_box.length))
	for chunk_coord in chunk_box.chunkPositions:
		map_coord = (chunk_coord[0] - chunk_box.mincx) << 4, (chunk_coord[1] - chunk_box.mincz) << 4
		chunk_tensor = level.getChunk(*chunk_coord).Blocks  # type:np.ndarray
		air_tensor = (chunk_tensor == level.materials.Air.ID)
		# argmin grabs the first non-air (False) it encounters.
		# Since we want to search from the TOP, we first flip the tensor in the y dimension
		height_map[map_coord[0]:map_coord[0]+16, map_coord[1]:map_coord[1]+16] = 255 - air_tensor[:,:,::-1].argmin(2)

	# extract box from chunk_box
	m_left = box.minx - chunk_box.minx
	m_right = chunk_box.maxx - box.maxx
	m_top = box.minz - chunk_box.minz
	m_bot = chunk_box.maxz - box.maxz
	return height_map[m_left:height_map.shape[0]-m_right, m_top:height_map.shape[1]-m_bot]


def calculate_relief_map(height_map, r=16):
	"""Create a map to represent variance in terrain height per 10x10 chunk

	Args:
		height_map (np.ndarray): The terrain heights
		r (int): The chunk width and height

	Returns (np.ndarray):

	"""
	# TODO: Rewrite to iterate over chunks, not r
	width, length = height_map.shape
	relief_map = np.zeros((int(ceil(width/float(r))), int(ceil(length/float(r)))))
	for x in range(0, width, r):
		for y in range(0, length, r):
			chunk = height_map[x:x+r, y:y+r]  # type:np.ndarray
			relief_map[int(x / r), int(y / r)] = chunk.var()
	return relief_map


def aggregate_height_per_chunk(height_map):
	r=16
	width, length = height_map.shape
	agg_height_map = np.zeros((int(ceil(width / float(r))), int(ceil(length / float(r)))))
	for x in range(0, width, r):
		for y in range(0, length, r):
			chunk = height_map[x:x + r, y:y + r]  # type:np.ndarray
			agg_height_map[int(x / r), int(y / r)] = chunk.mean()
	return agg_height_map


def perform(level, box, options):
	"""

	Args:
		level (MCInfdevOldLevel): Full level to generate the village in
		box (BoundingBox): Box that limits where the village can be placed
		options (dict): Options given to the Filter from MCEdit2
	"""
	import ipdb
	# ipdb.set_trace()
	print("TIME TO DEBUG")

	height_map = calculate_height_map(level, box)  # type: np.ndarray
	relief_map = calculate_relief_map(height_map)  # type: np.ndarray
	agg_height_map = aggregate_height_per_chunk(height_map)  # type: np.ndarray
	for xc, zc in itertools.product(xrange(agg_height_map.shape[0]), xrange(agg_height_map.shape[1])):
		y = int(agg_height_map[xc, zc])
		xc, zc = (xc << 4) + box.minx, (zc << 4) + box.minz
		for x in range(16):
			for z in range(16):
				level.setBlockAt(xc + x, y, zc + z, level.materials.Brick.ID)
				level.setBlockDataAt(xc + x, y, zc + z, 0)
