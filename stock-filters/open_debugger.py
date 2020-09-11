import itertools
from math import ceil

import utilityFunctions as utilityFunctions
from mcplatform import *

import numpy as np

from pymclevel import MCLevel, BoundingBox, MCInfdevOldLevel, materials

from village_generation.interpreter.interpreter import LevelColumnInterpreter, Interpreter, LevelChunkInterpreter, \
	ChunkInterpreter, ColumnInterpreter

inputs = (
	("Open Debugger", "label"),
	("Creator: Pluralis", "label"),
)


def calculate_relief_map(height_map, r=16):
	"""Create a map to represent variance in terrain height per 10x10 chunk

	Args:
		height_map (np.ndarray): The terrain heights
		r (int): The chunk width and height

	Returns (np.ndarray):

	"""
	# TODO: Rewrite to iterate over chunks, not r
	width, length = height_map.shape
	relief_map = np.zeros((int(ceil(width / float(r))), int(ceil(length / float(r)))))
	for x in range(0, width, r):
		for y in range(0, length, r):
			chunk = height_map[x:x + r, y:y + r]  # type:np.ndarray
			relief_map[int(x / r), int(y / r)] = chunk.std()
	return relief_map


def aggregate_height_per_chunk(height_map):
	r = 16
	width, length = height_map.shape
	agg_height_map = np.zeros((int(ceil(width / float(r))), int(ceil(length / float(r)))))
	for x in range(0, width, r):
		for y in range(0, length, r):
			chunk = height_map[x:x + r, y:y + r]  # type:np.ndarray
			agg_height_map[int(x / r), int(y / r)] = chunk.mean()
	return agg_height_map


class HeightInterpreter(Interpreter):
	surface_blocks = [block.ID for block in (  # Convert to ID so we can match blocks independent of state
		materials.alphaMaterials.Stone,
		materials.alphaMaterials.Grass,
		materials.alphaMaterials.Dirt,
		materials.alphaMaterials.Cobblestone,
		materials.alphaMaterials.Bedrock,
		materials.alphaMaterials.WaterActive,
		materials.alphaMaterials.Water,
		materials.alphaMaterials.LavaActive,
		materials.alphaMaterials.Lava,
		materials.alphaMaterials.Sand,
		materials.alphaMaterials.Gravel,
		materials.alphaMaterials.GoldOre,
		materials.alphaMaterials.IronOre,
		materials.alphaMaterials.CoalOre,
		materials.alphaMaterials.Sandstone,
		materials.alphaMaterials.MossStone,
		materials.alphaMaterials.DiamondOre,
		materials.alphaMaterials.RedstoneOre,
		materials.alphaMaterials.Ice,
		materials.alphaMaterials.Snow,
		materials.alphaMaterials.Clay,
		materials.alphaMaterials.SoulSand,
		materials.alphaMaterials.Glowstone,
		materials.alphaMaterials.FrostedIce,
		materials.alphaMaterials.get('magma'),
	)]

	def interpret(self, obj):
		surface_height = 255 - np.isin(obj, self.surface_blocks)[::-1].argmax()
		return surface_height


def center_level(level, box, height_map):
	"""Shift columns in level such that the surface is flat"""
	new_level = _clone_level(level)
	for cx, cz in box.chunkPositions:
		c = new_level.getChunk(cx, cz)
		for x, z in itertools.product(xrange(16), xrange(16)):
			x_map = (cx << 4) + x - box.minx
			z_map = (cz << 4) + z - box.minz
			offset = height_map[x_map, z_map] - 128
			c.Blocks[x, z, -offset:] = c.Blocks[x, z, :offset]  # Assuming negative offset since surface is always below half
			c.Blocks[x, z, :-offset] = new_level.materials.Bedrock.ID
	return new_level


def _clone_level(level):
	new_dirname = os.path.dirname(level.filename) + "_Cloned"
	new_filename = os.path.basename(level.filename)
	from shutil import copytree, rmtree
	rmtree(new_dirname, ignore_errors=True)
	copytree(os.path.dirname(level.filename), new_dirname)
	new_level = MCInfdevOldLevel(os.path.join(new_dirname, new_filename))
	return new_level


def perform(level, box, options):
	"""

	Args:
		level (MCInfdevOldLevel): Full level to generate the village in
		box (BoundingBox): Box that limits where the village can be placed
		options (dict): Options given to the Filter from MCEdit2
	"""
	print("TIME TO DEBUG")

	height_mapper = LevelColumnInterpreter(HeightInterpreter())
	height_map = height_mapper.interpret((level, box)).astype(int)

	relief_map = calculate_relief_map(height_map)  # type: np.ndarray
	# agg_height_map = aggregate_height_per_chunk(height_map)  # type: np.ndarray
	centered_level = center_level(level, box, height_map)

	build_coords = find_build_area(relief_map, centered_level, box, n=100)
	build_map = np.zeros_like(relief_map, dtype=bool)
	build_map[build_coords] = True
	set_height_with_bricks(100 * build_map, box, level)


def _out_of_bounds(coord, area_shape):
	return (coord[0] < 0 or
			coord[0] >= area_shape[0] or
			coord[1] < 0 or
			coord[1] >= area_shape[1])


def _calculate_relief_scores(relief_map):
	area_map = np.zeros_like(relief_map, dtype=int)
	cutoffs = np.sort(relief_map.flatten())

	base_factor = 0.9
	factor = 1
	for score in range(1, 21):
		factor *= base_factor
		cutoff = cutoffs[int(len(cutoffs) * factor)]
		area_map[relief_map <= cutoff] = score
	return area_map


def overlay_on_map(
		grid  # type: np.ndarray
):
	image_size = 792., 792.
	square_size = (image_size[0] / grid.shape[0],
				   image_size[1] / grid.shape[1])

	import matplotlib.pyplot as plt
	import scipy.sparse as sps
	img = plt.imread("SecondWorldMap.png")
	fig, ax = plt.subplots()
	ax.imshow(img)

	coordinate_grid = sps.coo_matrix(grid)
	x = coordinate_grid.col * square_size[0] + square_size[0] / 2
	y = coordinate_grid.row * square_size[1] + square_size[1] / 2
	v = coordinate_grid.data
	plt.scatter(x, y, c=v, s=100, marker="+", cmap="cool")
	plt.show()


def _build_material_map(level, box, materials, search_depth, blocks_needed=1):
	material_map = np.zeros((box.maxcx - box.mincx, box.maxcz - box.mincz), dtype=int)

	for chunk_coord in box.chunkPositions:
		chunk_tensor = level.getChunk(*chunk_coord).Blocks  # type:np.ndarray
		chunk_surface_tensor = chunk_tensor[:, :, 128-search_depth:129]  # surface is expected to be centered around 128
		chunk_material_count_tensor = np.isin(chunk_surface_tensor, [m.ID for m in materials])
		# TODO: transpose. Maps should be indexed (x,z)
		material_map[chunk_coord[1] - box.mincz, chunk_coord[0] - box.mincx] = chunk_material_count_tensor.sum() >= blocks_needed
	return material_map


def _spread_scores_over_map(
		score_map,  # type: np.ndarray
		subtract=True
):
	neighbour_scores = np.zeros(score_map.shape + (4,))
	neighbour_scores[1:, :, 0] = score_map[:-1, :]
	neighbour_scores[:-1, :, 1] = score_map[1:, :]
	neighbour_scores[:, 1:, 2] = score_map[:, :-1]
	neighbour_scores[:, :-1, 3] = score_map[:, 1:]
	max_neighbour_score = neighbour_scores.max(2) - subtract
	return np.maximum(score_map, max_neighbour_score)


def _calculate_resource_scores(level, box):
	tree_scores = _calculate_tree_scores(level, box)
	rock_scores = _calculate_rock_scores(level, box)
	return np.maximum(tree_scores, rock_scores) + np.minimum(tree_scores, rock_scores)/2


class MaterialCountInterpreter(ColumnInterpreter):
	def __init__(self, materials_to_count, search_depth, search_height=0):
		super(MaterialCountInterpreter, self).__init__()
		self.__materials = materials_to_count
		self.__search_depth = search_depth
		self.__search_height = search_height

	def interpret(self, obj):
		materials_to_count = self.__materials
		search_depth = self.__search_depth
		search_height = self.__search_height
		surface_slice = obj[128 - search_depth:129 + search_height]  # surface is expected to be centered around 128
		material_count_tensor = np.isin(surface_slice, [m.ID for m in materials_to_count])
		return material_count_tensor.sum()


def find_build_area(
		relief_map,  # type: np.ndarray
		level,
		box,
		n=None
):
	"""Find all areas where the first buildings will be made

	The algorithm attributes scores to every chunk
		- 0 - 20 for relief
		- 0 - 15 for proximity to water
		- 0 - 15 for proximity to resources
			- 0 - 10 for proximity to trees
			- 0 - 10 for proximity to rocks
			score is 100% of highest + 50% of lowest

	If n is None (default), return scores for all chunks in a grid shape
	If n is an integer, return a 2xn tuple with coordinates of the top n chunks (order not guaranteed)
	"""
	relief_scores = _calculate_relief_scores(relief_map)
	# TODO: water scores is transposed from the original ones. How to handle?
	water_scores = _calculate_water_scores(level, box)
	resource_scores = _calculate_resource_scores(level, box)

	scores = relief_scores + water_scores.T + resource_scores.T
	scores = np.argsort(scores, axis=None)
	if n is not None:
		return np.unravel_index(scores[:n], relief_scores.shape)
	# TODO: investigate: find biggest rectangle that fits within True chunks for every cutoff, find optimal point on
	#  this curve (hopefully it is a sigmoid)
	return scores


def _calculate_water_scores(level, box):
	water = [materials.alphaMaterials.Water, materials.alphaMaterials.WaterActive]
	return _calculate_material_scores(level, box, water, 10, 5)


def _calculate_tree_scores(level, box):
	trees = [level.materials.Wood]
	return _calculate_material_scores(level, box, trees, 15, 0, 3)


def _calculate_rock_scores(level, box):

	ores = (materials.alphaMaterials.GoldOre,
			materials.alphaMaterials.IronOre,
			materials.alphaMaterials.CoalOre)
	return _calculate_material_scores(level, box, ores, 5, 15)


def _calculate_material_scores(level, box, material, blocks_needed, search_depth, search_height=0):
	def check_count(arr): return np.sum(arr) >= blocks_needed

	column_scorer = MaterialCountInterpreter(material, search_depth, search_height)
	chunk_scorer = ChunkInterpreter(column_scorer, check_count)
	level_scorer = LevelChunkInterpreter(chunk_scorer)
	material_scores = level_scorer.interpret((level, box))

	material_scores *= 10  # Calculate closeness to trees (10 - manhattan distance)
	for i in range(10):
		material_scores = _spread_scores_over_map(material_scores)
	return material_scores


def set_height_with_bricks(agg_height_map, box, level):
	for xc, zc in itertools.product(xrange(agg_height_map.shape[0]), xrange(agg_height_map.shape[1])):
		y = int(agg_height_map[xc, zc])
		xc, zc = (xc << 4) + box.minx, (zc << 4) + box.minz
		for x in range(16):
			for z in range(16):
				level.setBlockAt(xc + x, y, zc + z, level.materials.Brick.ID)
				level.setBlockDataAt(xc + x, y, zc + z, 0)
