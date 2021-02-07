"""Contains Interpreter subclasses that are applied directly to 3D numpy arrays representing minecraft levels"""
import numpy as np
from typing import Iterable, Tuple

from pymclevel import materials
from village_generation.convolution.convolution import Convolution
from village_generation.interpret.interpreter import Interpreter


class HeightInterpreter(Interpreter):
	target_blocks = None

	def interpret(self, level):
		# type: (np.ndarray) -> np.ndarray
		"""Transform an XxZxY tensor into an XxZ tensor with values 0-255 indicating the height of the highest block of
		type target_blocks of that column in the level"""
		surface_height = np.subtract(255, np.isin(level, self.target_blocks)[:, :, ::-1].argmax(2))
		return surface_height


class SurfaceHeightInterpreter(HeightInterpreter):
	target_blocks = [block.ID for block in (  # Convert to ID so we can match blocks independent of state
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


class TopHeightInterpreter(HeightInterpreter):
	def interpret(self, level):
		# type: (np.ndarray) -> np.ndarray
		"""Transform an XxZxY tensor into an XxZ tensor with values 0-255 indicating the height of the highest block
		not of type Air of that column in the level"""
		surface_height = np.subtract(255, (level != materials.alphaMaterials.Air.ID)[:, :, ::-1].argmax(2))
		return surface_height


class MaterialCountConvolution(Convolution):
	def __init__(self, material, bounds, convolution_shape):
		# type: (Iterable[int], slice, Tuple[int, ...]) -> MaterialCountConvolution
		"""

		Args:
			material (Iterable[int]): Materials to count
			bounds (slice): Slice within column to count the materials in
			convolution_shape: Shape of array expected as first parameter to call
		"""
		super(MaterialCountConvolution, self).__init__(convolution_shape)
		self.material = material
		self.bounds = bounds  # TODO: Is 1D now, generify to ND?

	def __call__(self, arr):
		# type: (np.ndarray) -> np.ndarray
		return np.isin(arr[:, :, self.bounds], self.material).sum()