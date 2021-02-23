"""A convolution is a function applied to a moving window over a tensor (n-dimensional matrix).
A convolution in turn produces another tensor of possible another dimension.
Convolutions can be used to interpret a tensor. Take a 3D minecraft level. A convolution might be used to scan the
level for water-related blocks, indicating with True or False if a 3x3x256 cuboid contains enough water blocks.
Another convolution may then be applied to that 2D output (since there is only 1 possible height to apply it to).
For example a 32x32 square that scans if there is enough water nearby enough to build a village (say 2 True sources)."""
import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from typing import Callable, Tuple, TypeVar, Optional, Any

from village_generation.interpret.interpreter import Interpreter


class Convolution:
	__metaclass__ = ABCMeta

	def __init__(self, convolution_shape):
		self.convolution_shape = convolution_shape

	@abstractmethod
	def __call__(self, arr):
		pass


T = TypeVar("T")
T_ = TypeVar("T_")


class FunctionConvolution(Convolution):
	"""Convolution that simply applies function to the input array"""
	def __init__(self, function, convolution_shape):
		# type: (Callable[[T], T_], Tuple[int, ...]) -> FunctionConvolution
		super(FunctionConvolution, self).__init__(convolution_shape)
		self.function = function

	def __call__(self, arr):
		# type: (T) -> T_
		return self.function(arr)


class ConvolutionInterpreter(Interpreter):
	def __init__(self, convolution, step_size=None):
		# type: (Convolution, Optional[Tuple, ...]) -> ConvolutionInterpreter
		"""

		Args:
			convolution: Convolution (callable) which takes an x0 x x1 x ... xn sized np.ndarray and returns a single value
			step_size (tuple): optional n-length tuple with the step size for each dimension
		"""
		if step_size is None:
			step_size = tuple([1]*len(convolution.convolution_shape))
		else:
			assert len(step_size) == len(convolution.convolution_shape), \
				"Step size and convolution input are not of same dimensionality"
		self.step_size = step_size
		self.convolution = convolution
		self.convolution_shape = convolution.convolution_shape

	def interpret(self, obj):
		# type: (np.ndarray) -> np.ndarray
		"""
		Apply convolution to each subsection of obj

		Args:
			obj (np.ndarray): Object to apply convolution to, shape X0 x X1 x ... x Xn

		Returns:
			Concatenated output of convolution applied to each subsection of obj
			Shape is (obj shape - convolution input shape) / step_size + 1
		"""
		# TODO: readability
		shape = obj.shape
		assert len(shape) == len(self.convolution_shape), "Convolution input and object have different dimensionality"
		output_shape = tuple([(s-c)/ss + 1 for s, c, ss in zip(shape, self.convolution_shape, self.step_size)])
		# output_shape = tuple([d for d in output_shape if d != 1])  # 1-width dimensions is no dimension
		output = np.empty(output_shape)

		for out_coord in itertools.product(*[xrange(n) for n in output_shape]):
			origin = [o*s for o, s in zip(out_coord, self.step_size)]
			slices = [slice(o, o+c) for o, c in zip(origin, self.convolution_shape)]
			output[out_coord] = self.convolution(obj[slices])
		return output
