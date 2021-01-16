import itertools

import dill as dill
import numpy as np
from typing import Tuple, List

from mcplatform import *
from pymclevel import BoundingBox, MCInfdevOldLevel, materials, AnvilChunk
from village_generation.interpreter.convolution import Convolution, ConvolutionInterpreter, FunctionConvolution
from village_generation.interpreter.interpreter import LevelColumnInterpreter, Interpreter, LevelChunkInterpreter, \
	ColumnInterpreter, ChunkInterpreter
from create_house import fill_with_houses

inputs = (
	("Interpret Level", "label"),
	("Creator: Pluralis", "label"),
)


def calculate_relief_map(height_map, r=16):
	"""Create a map to represent variance in terrain height per rxr area

	h is the height (3rd dimension) of height_map
	Args:
		height_map (np.ndarray): The terrain heights
		r (int): The chunk width and height

	Returns (np.ndarray):

	"""
	return aggregate_height_per_chunk(height_map, np.std, r)


def aggregate_height_per_chunk(height_map, aggregator=np.mean, r=16):
	c = FunctionConvolution(aggregator, (r, r))
	return ConvolutionInterpreter(c, (r,r)).interpret(height_map)


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
		materials.alphaMaterials.LapisLazuliOre,
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


def center_level(level, height_map):
	"""Shift columns in level such that the surface is flat

	All blocks that fall out of the level are removed. Blocks shifted into the level are bedrock.

	Args:
		level (np.ndarray): Level-size array
		height_map (np.ndarray): Offsets to define how much to shift each column, level width by level height
	Returns:
		(np.ndarray) of same shape as level with all columns (third dimension) shifted
	"""
	shifted_level = np.copy(level)
	for (x, z), height in np.ndenumerate(height_map):
		offset = level.shape[-1]/2 - height
		shifted_level[x, z, offset:] = level[x, z, :-offset]
		shifted_level[x, z, :offset] = materials.alphaMaterials.Bedrock.ID
	return shifted_level


def _clone_level(level):
	new_dirname = os.path.dirname(level.filename) + "_Cloned"
	new_filename = os.path.basename(level.filename)
	from shutil import copytree, rmtree
	rmtree(new_dirname, ignore_errors=True)
	copytree(os.path.dirname(level.filename), new_dirname)
	new_level = MCInfdevOldLevel(os.path.join(new_dirname, new_filename))
	return new_level


class IdentityInterpreter(ColumnInterpreter):
	def interpret(self, obj):
		return obj


def _build_level_array(level, box):
	interpreter = LevelColumnInterpreter(IdentityInterpreter(), 256)
	array = interpreter.interpret((level, box))
	return array


class ChunkHeightInterpreter(ChunkInterpreter):
	def interpret(
			self,
			obj  # type:AnvilChunk
	):
		return obj.HeightMap


def _neighbor_coords(
		coord  # type: Tuple[int, int]
):
	coords = []
	for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
		coords.append((coord[0] + dx, coord[1] + dy))
	return coords


def within(coord, shape):
	return all(0 <= c < s for c, s in zip(coord, shape))


def get_all_connected(group_i, i, unvisited, build_map, build_coords):
	group = [i]  # type: List[int]
	coord = build_coords[i]
	neighbors = _neighbor_coords(coord)
	for neighbor in neighbors:
		if not within(neighbor, build_map.shape):
			continue
		neighbor_i = build_map[neighbor[0], neighbor[1]]
		if neighbor_i in unvisited:
			unvisited.remove(neighbor_i)
			group.extend(get_all_connected(group_i, neighbor_i, unvisited, build_map, build_coords))
	return group


def find_largest_buildable_area(
		relief_map,  # type: np.array
		build_coords,  # type: Tuple[np.array, np.array]
):
	build_coords = zip(*build_coords)  # type: List[Tuple[int, int]]
	build_map = np.zeros_like(relief_map, dtype=int) - 1  # -1 unbuildable
	n = len(build_coords)
	for i, (x, y) in enumerate(build_coords):
		build_map[x, y] = i  # TODO: revert y/x?

	unvisited = set(range(n))
	groups = []  # type: List[List[int]]
	group_i = 0
	while unvisited:
		i = unvisited.pop()
		groups.append(get_all_connected(group_i, i, unvisited, build_map, build_coords))
		group_i += 1

	biggest_group = max(groups, key=len)
	group_map = np.zeros_like(build_map, dtype=bool)
	for i in biggest_group:
		group_map[build_coords[i]] = 1

	return group_map


# TODO: make level / box shape agnostic
def perform(level, box, options):
	"""

	Args:
		level (MCInfdevOldLevel): Full level to generate the village in
		box (BoundingBox): Box that limits where the village can be placed
		options (dict): Options given to the Filter from MCEdit2
	"""
	# TODO: Move into _perform by using level_array in stead of level
	height_mapper = LevelColumnInterpreter(HeightInterpreter(), 1)
	surface_height_map = height_mapper.interpret((level, box)).astype(int).reshape((400,400))

	top_height_map = LevelChunkInterpreter(ChunkHeightInterpreter((16, 16))).interpret((level, box)).reshape(400, 400)

	relevant_box = _find_relevant_box(box, surface_height_map, top_height_map)
	# TODO: split into build array and slice relevant
	#  that way we can move the the functions above into a convolution
	level_array = _build_level_array(level, relevant_box)

	build_map = _perform(level_array, surface_height_map)
	fill_with_houses(level, box, options, build_map, surface_height_map)

	set_chunk_height_with_bricks(100 * build_map, box, level)


def _perform(level_array, surface_height_map):
	relief_map = calculate_relief_map(surface_height_map)  # type: np.ndarray
	centered_level = center_level(level_array, surface_height_map)
	build_coords = find_buildable_area(relief_map, centered_level, n=100)
	village_area = find_largest_buildable_area(relief_map, build_coords)
	# agg_height_map = aggregate_height_per_chunk(height_map)  # type: np.ndarray
	# build_map = np.zeros_like(relief_map, dtype=bool)
	# build_map[build_coords] = True
	return village_area


def _find_relevant_box(box, surface_height_map, top_height_map, max_depth=10, max_height=10):
	height = top_height_map.max() + max_height - (surface_height_map.min() - max_depth)
	relevant_origin = (box.origin.x, surface_height_map.min() - max_depth, box.origin.z)
	relevant_size = (box.size.x, height, box.size.z)
	relevant_box = BoundingBox(relevant_origin, relevant_size)
	return relevant_box


def _out_of_bounds(coord, area_shape):
	return (coord[0] < 0 or
			coord[0] >= area_shape[0] or
			coord[1] < 0 or
			coord[1] >= area_shape[1])


def _calculate_clear_scores(level):
	material = [
		materials.alphaMaterials.Sapling.ID,
		materials.alphaMaterials.Water.ID,
		materials.alphaMaterials.WaterActive.ID,
		materials.alphaMaterials.Lava.ID,
		materials.alphaMaterials.LavaActive.ID,
		materials.alphaMaterials.Wood.ID,
		materials.alphaMaterials.Sponge.ID,
		materials.alphaMaterials.Cactus.ID,
		materials.alphaMaterials.MobHead.ID,
	]
	counts = _calculate_material_counts(level, material, 1, 10)
	return 10 - np.clip((counts / 5).astype(int), 0, 10)


def _calculate_flatness_scores(relief_map, level):
	relief_scores = _calculate_relief_scores(relief_map)
	clear_scores = _calculate_clear_scores(level)
	return np.minimum(relief_scores, clear_scores)


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


# TODO: not used
def _build_material_map(level, box, material, search_depth, blocks_needed=1):
	material_map = np.zeros((box.maxcx - box.mincx, box.maxcz - box.z), dtype=int)

	for chunk_coord in box.chunkPositions:
		chunk_tensor = level.getChunk(*chunk_coord).Blocks  # type:np.ndarray
		# surface is expected to be centered around 128
		chunk_surface_tensor = chunk_tensor[:, :, 128 - search_depth:129]
		chunk_material_count_tensor = np.isin(chunk_surface_tensor, [m.ID for m in material])
		# TODO: transpose. Maps should be indexed (x,z)
		x = chunk_coord[0] - box.mincx
		z = chunk_coord[1] - box.z
		material_map[z, x] = chunk_material_count_tensor.sum() >= blocks_needed
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


def _calculate_resource_scores(level):
	tree_scores = _calculate_tree_scores(level)
	rock_scores = _calculate_rock_scores(level)
	return np.maximum(tree_scores, rock_scores) + np.minimum(tree_scores, rock_scores) / 2


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


# TODO: do on a block level or convolve a house-sized space, not per chunk
def find_buildable_area(
		relief_map,  # type: np.ndarray
		level,  # type: np.ndarray
		n=None
):
	"""Find all areas where the first buildings will be made

	The algorithm attributes scores to every chunk
		- 0 - 20 for relief
		- 0 - 10 for proximity to water
		- 0 - 15 for proximity to resources
			- 0 - 10 for proximity to trees
			- 0 - 10 for proximity to rocks
			score is 100% of highest + 50% of lowest

	If n is None (default), return scores for all chunks in a grid shape
	If n is an integer, return a 2xn tuple with coordinates of the top n chunks (order not guaranteed)
	"""
	relief_scores = _calculate_flatness_scores(relief_map, level)  # TODO: make buildable score, e.g. no trees, water, etc
	# TODO: water scores is transposed from the original ones. How to handle?
	# TODO: Bonus points for rivers and lakes (connected body of water of at least X squares)
	# TODO: Extra bonus points if rivers and lakes hit the or multiple edges (trade route)
	water_scores = _calculate_water_scores(level)
	resource_scores = _calculate_resource_scores(level)

	scores = relief_scores.T + water_scores.T + resource_scores.T
	scores = np.argsort(scores, axis=None)
	if n is not None:
		return np.unravel_index(scores[:n], relief_scores.shape)
	# TODO: investigate: find biggest rectangle that fits within True chunks for every cutoff, find optimal point on
	#  this curve (hopefully it is a sigmoid)
	return scores


def _calculate_water_scores(level):
	water = [materials.alphaMaterials.Water.ID, materials.alphaMaterials.WaterActive.ID]
	return _calculate_material_scores(level, water, 60, 5)


def _calculate_tree_scores(level):
	trees = [materials.alphaMaterials.Wood.ID]
	return _calculate_material_scores(level, trees, 9, 0, 1)


def _calculate_rock_scores(level):
	ores = (materials.alphaMaterials.GoldOre.ID,
			materials.alphaMaterials.IronOre.ID,
			materials.alphaMaterials.CoalOre.ID)
	return _calculate_material_scores(level, ores, 10, 5)


class MaterialCountConvolution(Convolution):
	def __init__(self, material, bounds, convolution_shape):
		"""

		Args:
			material (Iterable[Block]): Materials to count
			bounds (slice): Slice within column to count the materials in
			convolution_shape: Shape of array expected as first parameter to call
		"""
		super(MaterialCountConvolution, self).__init__(convolution_shape)
		self.material = material
		self.bounds = bounds  # TODO: Is 1D now, generify to ND?

	def __call__(self, arr):
		return np.isin(arr[:, :, self.bounds], self.material).sum()


def _calculate_material_scores(level, material, blocks_needed, search_depth, search_height=0):
	material_scores = _calculate_material_counts(level, material, search_depth, search_height)
	material_scores = np.greater_equal(material_scores, blocks_needed)

	material_scores = material_scores.astype(int) * 10  # Calculate closeness to trees (10 - manhattan distance)
	for i in range(10):
		material_scores = _spread_scores_over_map(material_scores)
	return material_scores


def _calculate_material_counts(level, material, search_depth, search_height):
	level_height = level.shape[2]
	search_bounds = slice(level_height / 2 - search_depth,
						  level_height / 2 + 1 + search_height)  # TODO: handle odd level_height
	material_count_convolution = MaterialCountConvolution(material, search_bounds, (16, 16, level.shape[2]))
	count_interpreter = ConvolutionInterpreter(material_count_convolution, (16, 16, 1))
	material_scores = count_interpreter.interpret(level)
	return material_scores


def set_column_height_with_bricks(
		height_map,  # type: np.ndarray
		box, level):
	for (x, z), v in np.ndenumerate(height_map):
		x, z = x + box.minx, z + box.minz
		level.setBlockAt(x, v, z, level.materials.Brick.ID)
		level.setBlockDataAt(x, v, z, 0)


def set_chunk_height_with_bricks(agg_height_map, box, level):
	for xc, zc in itertools.product(xrange(agg_height_map.shape[0]), xrange(agg_height_map.shape[1])):
		y = int(agg_height_map[xc, zc])
		xc, zc = (xc << 4) + box.minx, (zc << 4) + box.minz
		for x in range(16):
			for z in range(16):
				level.setBlockAt(xc + x, y, zc + z, level.materials.Brick.ID)
				level.setBlockDataAt(xc + x, y, zc + z, 0)


def store_level(level_array, surface_height_map, name="level"):
	d = {"level_array": level_array, "surface_height_map": surface_height_map}
	dill.dump(d, open(name+".pickle", "wb"))


def load_level(name="level"):
	d = dill.load(open(name+".pickle", "rb"))
	return d["level_array"], d["surface_height_map"]


if __name__ == '__main__':
	level_array, surface_height_map = load_level()
	build_map = _perform(level_array, surface_height_map)
