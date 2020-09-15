import itertools
from abc import ABCMeta, abstractmethod

import numpy as np

from interpreter import Interpreter


class Convolution:
	__metaclass__ = ABCMeta

	def __init__(self, convolution_shape):
		self.convolution_shape = convolution_shape

	@abstractmethod
	def __call__(self, arr):
		pass


class FunctionConvolution(Convolution):
	def __init__(self, function, convolution_shape):
		super(FunctionConvolution, self).__init__(convolution_shape)
		self.function = function

	def __call__(self, arr):
		return self.function(arr)


class ConvolutionInterpreter(Interpreter):
	def __init__(self, convolution, step_size=None):
		"""

		Args:
			convolution: callable which takes an x0 x x1 x ... xn sized np.ndarray and returns a single value
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
		output_shape = tuple([d for d in output_shape if d != 1])  # 1-width dimensions is no dimension
		output = np.empty(output_shape)

		for out_coord in itertools.product(*[xrange(n) for n in output_shape]):
			origin = [o*s for o, s in zip(out_coord, self.step_size)]
			slices = [slice(o, o+c) for o, c in zip(origin, self.convolution_shape)]
			output[out_coord] = self.convolution(obj[slices])
		return output
