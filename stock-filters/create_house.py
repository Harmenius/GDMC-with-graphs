import numpy as np
from typing import Tuple

from pymclevel import materials, BoundingBox, MCInfdevOldLevel

template = np.array(
    [
        [
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ] * 2
    +
    [
        [
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ]
)


def perform(level, box, options):
    """Build a house in the box

    Args:
        level (MCInfdevOldLevel): Full level to generate the house
        box (BoundingBox): Box that indicates where the house will be built
        options (dict): Options given to the Filter from MCEdit2
    """
    material = materials.alphaMaterials.WoodPlanks

    for x in range(template.shape[2]):
        xw = x + box.minx
        for z in range(template.shape[1]):
            zw = z + box.minz
            for y in range(template.shape[0]):
                yw = y + box.miny
                level.setBlockAt(xw, yw, zw, material.ID * template[y, z, x])
                level.setBlockDataAt(xw, yw, zw, 0)


def __traverse_diagonally(rectangle):
    x_min, x_max, y_min, y_max = rectangle
    width, height = x_max - x_min + 1, y_max - y_min + 1
    n_diagonals = width + height - 1
    for diagonal in range(n_diagonals):
        for offset in range(min(width, height, diagonal+1, n_diagonals - diagonal)):
            dx, dy = max(diagonal - (height-1), 0) + offset, min(diagonal, height-1) - offset
            yield x_min + dx, y_min + dy


def _determine_height(
        surface_height_map,  # type: np.ndarray
        house_slice  # type: Tuple[Tuple[int, int], Tuple[int, int]]
):
    heights = surface_height_map[house_slice[0][0]:house_slice[0][1],
                                 house_slice[1][0]:house_slice[1][1]]
    return heights.min()


def fill_with_houses(level, box, options, build_map, surface_height_map):
    """Place houses in the level and box within the area indicated by build map at height of surface_height_map

    Args:
        level: Level to build the houses in
        box: 3D box to build the houses in in level
        options: Nothing, passed to perform
        build_map: np.array(bool) same shape as chunks in box (a chunk is 16 x 16 x level_height)
            Indicate whether this is the area where a house can be built
        surface_height_map: np.array(bool) same shape as first 2 dimensions of box
            Indicate the surface level of every column

    Returns: Nothing, but alters level in place

    """
    house_shape = template.shape[0], template.shape[1]

    subrectangle = __grab_subrectangle(build_map)
    available_map = build_map.copy()

    for chunk_coord in __traverse_diagonally(subrectangle):
        if available_map[chunk_coord[0], chunk_coord[1]]:
            coord = chunk_coord[0] << 4, chunk_coord[1] << 4
            house_slice = (coord[0], coord[0] + house_shape[0]), (coord[1], coord[1] + house_shape[1])
            floor = _determine_height(surface_height_map, house_slice)
            house_box = BoundingBox((box.minx + house_slice[0][0], box.miny + floor, box.minz + house_slice[1][0]),
                                    template.shape)
            perform(level, house_box, options)


def __grab_subrectangle(build_map):
    lr = build_map.any(1)
    x_min, x_max = lr.argmax(), len(lr) - 1 - lr[::-1].argmax()
    td = build_map.any(0)
    y_min, y_max = td.argmax(), len(td) - 1 - td[::-1].argmax()
    return x_min, x_max, y_min, y_max
