import pytest
import numpy as np
from numpy.testing import assert_array_equal

from village_generation.convolution.convolution import Convolution, FunctionConvolution, ConvolutionInterpreter


class DummyConvolution(Convolution):
	def __call__(self, arr):
		return 1


@pytest.fixture
def function_convolution():
	return FunctionConvolution(lambda x: x + 1, (1,))


@pytest.fixture
def function_convolution_2d():
	return FunctionConvolution(lambda x: x[0]+x[1]+x[2], (3,1))


def function_convolution_applies_function(function_convolution):
	assert function_convolution(5) == 6

def test_convolution_interpreter_works_in_1d(function_convolution):
	fc = ConvolutionInterpreter(function_convolution)
	assert_array_equal(fc.interpret(np.array([1,2,3])), np.array([2,3,4]))

def test_convolution_interpreter_works_with_step_size(function_convolution):
	fc = ConvolutionInterpreter(function_convolution, (2,))
	assert_array_equal(fc.interpret(np.array([1,2,3])), np.array([2,4]))

def test_convolution_interpreter_works_in_2d(function_convolution_2d):
	fc = ConvolutionInterpreter(function_convolution_2d)
	assert_array_equal(fc.interpret(np.array([[1,2,3], [2,3,4], [3,4,5]])), np.array([[6, 9, 12]]))

def test_convolution_interpreter_works_with_larger_2d(function_convolution_2d):
	fc = ConvolutionInterpreter(function_convolution_2d)
	assert_array_equal(fc.interpret(np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])), np.array([[6, 9, 12], [9, 12, 15]]))

def test_convolution_interpreter_works_in_2d_with_step_size(function_convolution_2d):
	fc = ConvolutionInterpreter(function_convolution_2d, (1,2))
	assert_array_equal(fc.interpret(np.array([[1,2,3], [2,3,4], [3,4,5]])), np.array([[6, 12]]))

def test_convolution_interpreter_works_with_larger_2d_and_step_size(function_convolution_2d):
	fc = ConvolutionInterpreter(function_convolution_2d, (1,2))
	assert_array_equal(fc.interpret(np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])), np.array([[6, 12], [9, 15]]))
