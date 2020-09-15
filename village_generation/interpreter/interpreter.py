import itertools
from abc import ABCMeta, abstractmethod

import numpy as np


CHUNK_WIDTH = CHUNK_HEIGHT = 16


class Interpreter:
	__metaclass__ = ABCMeta

	@abstractmethod
	def interpret(self, obj):
		"""Transform the given object into a grid, classifying each subjection.

		Args:
			obj: The object to be interpreted

		Returns: A np.ndarray corresponding to the object with an additional optional dimension for multiple variables
		of interpretation

		"""
		pass


class ColumnInterpreter(Interpreter):
	__metaclass__ = ABCMeta


class ChunkInterpreter(Interpreter):
	__metaclass__ = ABCMeta

	def __init__(self, output_dimension):
		super(ChunkInterpreter, self).__init__()
		self.output_dimension = output_dimension


class ChunkColumnInterpreter(ChunkInterpreter):
	def __init__(self, column_interpreter, aggregator, output_dimension=(1,)):
		"""

		Args:
			column_interpreter (Interpreter): Is applied to every column in the chunk
			aggregator (Callable): Aggregates the output of the column_interpreter calls
			output_dimension (tuple): Size of aggregator output
		"""
		super(ChunkColumnInterpreter, self).__init__(output_dimension)
		self.__column_interpreter = column_interpreter
		self.__aggregator = aggregator
		self.output_dimension = output_dimension

	def interpret(self, obj):
		"""Interpret a Chunk to a single object

		Args:
			obj (AnvilChunk): the chunk to interpret

		Returns: Output of aggregator
		"""
		blocks = obj.Blocks
		column_interpretations = []
		for x in xrange(CHUNK_WIDTH):
			partial_column_interpretations = []
			for z in xrange(CHUNK_HEIGHT):
				column = blocks[x, z, :]
				column_interpretation = self.__column_interpreter.interpret(column)
				partial_column_interpretations.append(column_interpretation)
			column_interpretations.append(partial_column_interpretations)
		return self.__aggregator(column_interpretations)


# TODO: The level probably fits in memory, especially if you take a horizontal slice (lowest surface - 10 : highest tree top)
#  This would mean we don't need the LevelChunkInterpreter complications anymore,
#  although splitting the level into chunks does add some nice hierarchy to the level.
class LevelChunkInterpreter(Interpreter):
	def __init__(
			self,
			chunk_interpreter,  # type: ChunkInterpreter
	):
		self.chunk_interpreter = chunk_interpreter

	def interpret(self, obj):
		level, box = obj
		output_shape = self._calculate_output_shape(box)
		output = np.empty(output_shape + self.chunk_interpreter.output_dimension)
		for chunk_position in box.chunkPositions:
			output_position = self._to_output_position(chunk_position, box)
			chunk = level.getChunk(*chunk_position)
			chunk_interpretation = self.chunk_interpreter.interpret(chunk)
			output[output_position[0], output_position[1], :] = chunk_interpretation

		# TODO: Don't fix failing np dimension stuff here
		#  Removing this bit causes the output to be 3D because output_dimension is (1,)
		#  Setting output_dimension to (,) will have the line above fail because it uses 3 indexes
		output_dimension = self.chunk_interpreter.output_dimension
		if len(output_dimension) == 1 and output_dimension[0] == 1:
			output = np.reshape(output, output_shape)
		return output

	@staticmethod
	def _calculate_output_shape(box):
		return box.maxcx - box.mincx, box.maxcz - box.mincz

	@staticmethod
	def _to_output_position(chunk_position, box):
		return chunk_position[0] - box.mincx, chunk_position[1] - box.mincz


class LevelColumnInterpreter(Interpreter):
	# TODO: support multi-dimensional output
	def __init__(self, column_interpreter, output_length):
		self.column_interpreter = column_interpreter
		self.output_length = output_length

	# TODO: obj should be level, box. Check how to fix with superclass
	def interpret(self, obj):
		level, box = obj
		output = np.empty((box.size.x, box.size.z, self.output_length))
		for cx, cz in box.chunkPositions:
			chunk = level.getChunk(cx, cz)
			blocks = chunk.Blocks
			for dx, dz in itertools.product(xrange(16), xrange(16)):
				x = (cx << 4) + dx
				z = (cz << 4) + dz
				out_x = x - box.minx
				out_z = z - box.minz
				output[out_x, out_z, :] = self.column_interpreter.interpret(blocks[dx, dz, :])
		return output


# TODO: most interpreters work near the surface. To make it easier to handle, let's make a Level where the surface is
#  at the same level everywhere: shift every column with its heightmap offset from the mean. Fill with bedrock/air.
#  Then we can take a e.g. 10-height slice (5 above, 5 below surface) and interpret that. That way, we do not have to
#  pass the heightmap to every column interpreter.
#  We still need to be able to pass full maps to a LevelInterpreter, which are then sliced and the slice handed to
#  the chunk/column interpreter (so it matches with the level slice it is interpreting)

