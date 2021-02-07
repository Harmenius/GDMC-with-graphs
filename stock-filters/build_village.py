import numpy as np
from typing import Tuple, List, Union

from create_house import fill_with_houses
from pymclevel import BoundingBox, MCInfdevOldLevel, materials
from village_generation.conversion.np_mc import build_level_array, export_level
from village_generation.convolution.functions import spread_scores_over_map, calculate_relief_map, \
	calculate_material_counts
from village_generation.edit_level.edit_level import set_chunk_height_with_bricks
from village_generation.interpret.level_interpreter import SurfaceHeightInterpreter, TopHeightInterpreter
from village_generation.support.serialisation import load_level


displayName = "Build Village"
inputs = (
	("Build village", "label"),
	("Creator: Pluralis", "label"),
)


def center_level(level, height_map):
	# type: (np.ndarray, np.ndarray) -> np.ndarray
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
		offset = level.shape[-1] / 2 - height
		shifted_level[x, z, offset:] = level[x, z, :-offset]
		shifted_level[x, z, :offset] = materials.alphaMaterials.Bedrock.ID
	return shifted_level


def _neighbor_coords(
		coord  # type: Tuple[int, int]
):
	coords = []
	for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
		coords.append((coord[0] + dx, coord[1] + dy))
	return coords


def _within(coord, shape):
	return all(0 <= c < s for c, s in zip(coord, shape))


def get_all_connected(group_i, i, unvisited, build_map, build_coords):
	group = [i]  # type: List[int]
	coord = build_coords[i]
	neighbors = _neighbor_coords(coord)
	for neighbor in neighbors:
		if not _within(neighbor, build_map.shape):
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


def _change_level(level_array, build_map, surface_height_map):
	# type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
	level_array = fill_with_houses(level_array, build_map, surface_height_map)
	return set_chunk_height_with_bricks(100 * build_map, level_array)


# TODO: make level / box shape agnostic
def perform(level, box, options):
	"""

	Args:
		level (MCInfdevOldLevel): Full level to generate the village in
		box (BoundingBox): Box that limits where the village can be placed
		options (dict): Options given to the Filter from MCEdit2
	"""
	level_array = build_level_array(level, box)

	build_map, surface_height_map = _perform(level_array)
	new_level_array = _change_level(level_array, build_map, surface_height_map)

	export_level(level, box, new_level_array, level_array)



def _perform(level_array):
	# type: (np.ndarray) -> (np.ndarray, np.ndarray)
	surface_height_map = SurfaceHeightInterpreter().interpret(level_array)
	relief_map = calculate_relief_map(surface_height_map)  # type: np.ndarray
	centered_level = center_level(level_array, surface_height_map)
	top_height = TopHeightInterpreter().interpret(centered_level).max()
	sliced_level, surface_height = _slice_relevant_level(centered_level, top_height)
	build_coords = find_buildable_area(relief_map, sliced_level, surface_height, n=100)
	village_area = find_largest_buildable_area(relief_map, build_coords)
	return village_area, surface_height_map


def _slice_relevant_level(level, top_height, surface_height_map=None, pad_depth=10, pad_height=10):
	# type: (np.ndarray, int, np.ndarray, int, int) -> Tuple[np.ndarray, int]
	"""Reduce level size by grabbing a vertical slice from the lowest surface to the highest surface plus padding"""
	if surface_height_map is None:
		surface_height = level.shape[2] // 2
	else:
		surface_height = surface_height_map.min()
	depth = surface_height - pad_depth
	return level[:, :, depth: top_height + pad_height], pad_depth


def _calculate_clear_scores(level, surface_height):
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
	counts = calculate_material_counts(level, material, 1, 10, surface_height=surface_height)
	return 10 - np.clip((counts / 5).astype(int), 0, 10)


def _calculate_flatness_scores(relief_map, level, surface_height):
	relief_scores = _calculate_relief_scores(relief_map)
	clear_scores = _calculate_clear_scores(level, surface_height)
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


def _calculate_resource_scores(level, surface_height):
	tree_scores = _calculate_tree_scores(level, surface_height)
	rock_scores = _calculate_rock_scores(level, surface_height)
	return np.maximum(tree_scores, rock_scores) + np.minimum(tree_scores, rock_scores) / 2


# TODO: do on a block level or convolve a house-sized space, not per chunk
def find_buildable_area(
		relief_map, level, surface_height, n=None):
	# type: (np.ndarray, np.ndarray, int, int) -> Union[np.ndarray, Tuple[Tuple[int, ...], Tuple[int, ...]]]
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
	relief_scores = _calculate_flatness_scores(relief_map, level, surface_height)
	# TODO: make buildable score, e.g. no trees, water, etc
	# TODO: water scores is transposed from the original ones. How to handle?
	# TODO: Bonus points for rivers and lakes (connected body of water of at least X squares)
	# TODO: Extra bonus points if rivers and lakes hit the or multiple edges (trade route)
	water_scores = _calculate_water_scores(level, surface_height)
	resource_scores = _calculate_resource_scores(level, surface_height)

	scores = relief_scores.T + water_scores.T + resource_scores.T
	scores = np.argsort(scores, axis=None)
	if n is not None:
		return np.unravel_index(scores[:n	], relief_scores.shape)
	# TODO: investigate: find biggest rectangle that fits within True chunks for every cutoff, find optimal point on
	#  this curve (hopefully it is a sigmoid)
	return scores


def _calculate_water_scores(level, surface_height):
	water = [materials.alphaMaterials.Water.ID, materials.alphaMaterials.WaterActive.ID]
	return _calculate_material_scores(level, water, 60, 5, surface_height)


def _calculate_tree_scores(level, surface_height):
	trees = [materials.alphaMaterials.Wood.ID]
	return _calculate_material_scores(level, trees, 9, 0, 1, surface_height)


def _calculate_rock_scores(level, surface_height):
	ores = (materials.alphaMaterials.GoldOre.ID,
			materials.alphaMaterials.IronOre.ID,
			materials.alphaMaterials.CoalOre.ID)
	return _calculate_material_scores(level, ores, 10, 5, surface_height)


def _calculate_material_scores(level, material, blocks_needed, search_depth, surface_height, search_height=0):
	material_scores = calculate_material_counts(level, material, search_depth, search_height, surface_height)
	material_scores = np.greater_equal(material_scores, blocks_needed)

	material_scores = material_scores.astype(int) * 10  # Calculate closeness to trees (10 - manhattan distance)
	for i in range(10):
		material_scores = spread_scores_over_map(material_scores)
	return material_scores


if __name__ == '__main__':
	level_array_ = load_level()
	build_map_, surface_height_map_ = _perform(level_array_)
	new_level_array_ = _change_level(level_array_, build_map_, surface_height_map_)
