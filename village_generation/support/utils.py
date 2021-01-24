def traverse_diagonally(rectangle):
	# type: (Tuple[int, int, int, int]) -> Iterator[Tuple[int, int]]
	"""Iterate over coordinates within rectangle diagonally

	Uses the diagonals running from bottom-left to top-right. Starts at the top-left one.
	Runs over each diagonal starting at the bottom-left."""
	x_min, x_max, y_min, y_max = rectangle
	width, height = x_max - x_min + 1, y_max - y_min + 1
	n_diagonals = width + height - 1
	for diagonal in range(n_diagonals):
		for offset in range(min(width, height, diagonal + 1, n_diagonals - diagonal)):
			dx, dy = max(diagonal - (height - 1), 0) + offset, min(diagonal, height - 1) - offset
			yield x_min + dx, y_min + dy