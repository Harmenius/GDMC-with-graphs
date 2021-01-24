import numpy as np


def overlay_on_map(grid):
	# type: (np.ndarray) -> None
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