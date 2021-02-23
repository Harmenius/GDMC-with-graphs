import pytest
import numpy as np
from numpy.testing import assert_array_equal

from village_generation.convolution.functions import spread_scores_over_map, calculate_relief_map


@pytest.fixture
def map_single_1():
	return np.array([[0,0,0,0],[0,2,0,0],[0,0,0,0]])


@pytest.fixture
def spread_map_single_1():
	return np.array([[0,1,0,0],[1,2,1,0],[0,1,0,0]])


@pytest.fixture
def map_stack():
	return np.array([[2,0,2]])

@pytest.fixture
def spread_map_stack():
	return np.array([[2,1,2]])

@pytest.fixture
def map_3():
	return np.array([[2,0,3]])

@pytest.fixture
def spread_map_3():
	return np.array([[2,2,3]])

@pytest.fixture
def flat_map():
	return np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

@pytest.fixture
def small_flat_map():
	return np.array([[0,0],[0,0]])

@pytest.fixture
def relief_map():
	return np.array([[0,1,2,3], [1,2,3,4], [1,1,2,3], [2,2,2,3]])

def test_scores_spread_over_map_spread_distance_1(map_single_1, spread_map_single_1):
	assert_array_equal(spread_scores_over_map(map_single_1), spread_map_single_1)

def test_scores_spread_over_map_dont_stack(map_stack, spread_map_stack):
	assert_array_equal(spread_scores_over_map(map_stack), spread_map_stack)

def test_scores_spread_over_map_take_highest(map_3, spread_map_3):
	assert_array_equal(spread_scores_over_map(map_3), spread_map_3)

def test_calculate_relief_map_works_on_flat_map(flat_map):
	assert_array_equal(calculate_relief_map(flat_map, 1), flat_map)

def test_calculate_relief_map_works_on_flat_map_with_bigger_convolution_size(flat_map, small_flat_map):
	assert_array_equal(calculate_relief_map(flat_map, 2), small_flat_map)

def test_calculate_relief_map_works_on_higher_flat_map(flat_map):
	assert_array_equal(calculate_relief_map(flat_map + 1, 1), flat_map)

def test_calculate_relief_map_works_on_higher_flat_map_with_bigger_convolution_size(flat_map, small_flat_map):
	assert_array_equal(calculate_relief_map(flat_map + 1, 2), small_flat_map)

def test_calculate_relief_map_works_on_map_with_relief(relief_map):
	assert_array_equal(calculate_relief_map(relief_map, 4), np.array([[1.]]))
