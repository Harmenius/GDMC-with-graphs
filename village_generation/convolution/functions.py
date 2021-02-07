"""Functions using a Convolution to interpret a tensor"""
import numpy as np
from typing import Callable, Iterable

from village_generation.convolution.convolution import FunctionConvolution, ConvolutionInterpreter
from village_generation.interpret.level_interpreter import MaterialCountConvolution


def spread_scores_over_map(score_map):
	# type: (np.ndarray) -> np.ndarray
	"""Given a 2D score array, spread these scores to their neighbours.

	The maximum neighbour wins. Deduct by 1."""
	neighbour_scores = np.zeros(score_map.shape + (4,))
	neighbour_scores[1:, :, 0] = score_map[:-1, :]
	neighbour_scores[:-1, :, 1] = score_map[1:, :]
	neighbour_scores[:, 1:, 2] = score_map[:, :-1]
	neighbour_scores[:, :-1, 3] = score_map[:, 1:]
	max_neighbour_score = neighbour_scores.max(2) - 1
	return np.maximum(score_map, max_neighbour_score)


def calculate_relief_map(height_map, r=16):
	# type: (np.ndarray, int) -> np.ndarray
	"""Create a map to represent variance in terrain height per rxr area

	h is the height (3rd dimension) of height_map
	Args:
		height_map (np.ndarray): The terrain heights
		r (int): The chunk width and height

	Returns (np.ndarray):

	"""
	return aggregate_height_per_chunk(height_map, np.std, r)


def aggregate_height_per_chunk(height_map, aggregator=np.mean, r=16):
	# type: (np.ndarray, Callable[[np.ndarray], int], int) -> np.ndarray
	c = FunctionConvolution(aggregator, (r, r))
	return ConvolutionInterpreter(c, (r, r)).interpret(height_map)


def calculate_material_counts(level, material, search_depth, search_height, surface_height=None):
	# type: (np.ndarray, Iterable[int], int, int, int) -> np.ndarray
	if surface_height is None:
		surface_height = level.shape[2] / 2
	search_bounds = slice(surface_height - search_depth,
						  surface_height + 1 + search_height)
	material_count_convolution = MaterialCountConvolution(material, search_bounds, (16, 16, level.shape[2]))
	count_interpreter = ConvolutionInterpreter(material_count_convolution, (16, 16, 1))
	material_scores = count_interpreter.interpret(level)
	return material_scores