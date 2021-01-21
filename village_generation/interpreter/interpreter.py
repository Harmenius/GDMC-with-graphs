from abc import ABCMeta, abstractmethod

import numpy as np

from pymclevel import MCInfdevOldLevel, BoundingBox


class Interpreter:
	__metaclass__ = ABCMeta

	@abstractmethod
	def interpret(self, obj):
		# type: (np.ndarray) -> np.ndarray
		"""Transform the given object into a grid, classifying each subjection.

		Returns: A np.ndarray corresponding to the object with an additional optional dimension for multiple variables
		of interpretation

		"""
		pass


class RawLevelChunkInterpreter:
	def __init__(self):
		pass

	def interpret(self, level, box):
		# type: (MCInfdevOldLevel, BoundingBox) -> np.ndarray
		output_shape = self._calculate_output_shape(box)
		output = np.empty(output_shape)
		for chunk_position in box.chunkPositions:
			output_position = self._to_output_position(chunk_position, box)
			output[output_position[0], output_position[1], :] = level.getChunk(*chunk_position).Blocks

		return output

	@staticmethod
	def _calculate_output_shape(box):
		# TODO: Adapt to chunk size in case one or more box corners fall within a chunk
		#  otherwise the chunk returned by the chunk interpretation is too big
		#  Then later we can slice it back to the true box size
		return box.maxx - box.minx, box.maxz - box.minz, box.maxy - box.miny

	@staticmethod
	def _to_output_position(chunk_position, box):
		x_start, y_start = (chunk_position[0] << 4) - box.minx, (chunk_position[1] << 4) - box.minz
		return slice(x_start, x_start + (1 << 4)), slice(y_start, y_start + (1 << 4))
