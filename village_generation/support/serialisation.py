import dill as dill


def store_level(level_array, name="level"):
	d = {"level_array": level_array}
	dill.dump(d, open(name + ".pickle", "wb"))


def load_level(name="level"):
	d = dill.load(open(name + ".pickle", "rb"))
	return d["level_array"]
