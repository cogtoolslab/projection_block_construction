import numpy as np

from scoping_simulations.model.heuristics.cost_heuristic import ActionCostHeuristic
from scoping_simulations.model.utils.decomposition_functions import Subgoal


class AreaSizeHeuristic(ActionCostHeuristic):
    """How large is the area of the rectangular subgoals?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the area of the subgoal."""
        bitmap = subgoal.bitmap
        minx = bitmap.min(axis=0)[0]
        maxx = bitmap.max(axis=0)[0]
        miny = bitmap.min(axis=0)[1]
        maxy = bitmap.max(axis=0)[1]
        return (maxx - minx) * (maxy - miny)


class MassHeuristic(ActionCostHeuristic):
    """How many blocks are in the subgoal?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the mass of the subgoal."""
        # remove the previous target from the current one to get delta
        subgoal_target = subgoal.get_current_target()  # returns bool
        return subgoal_target.sum()


class NumberOfHolesHeuristics(ActionCostHeuristic):
    """How many holes (of whatever size) are in the bitmap?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the number of holes in the subgoal."""
        # iterate through all cells and check if they are holes
        bitmap = subgoal.bitmap
        bitmap = subset_bitmap(bitmap)
        holes = 0
        for x in range(bitmap.shape[0]):
            for y in range(bitmap.shape[1]):
                if bitmap[x, y] == 0:
                    # is this a new hole?
                    if (
                        x == 0
                        or bitmap[x - 1, y] == 1
                        or y == 0
                        or bitmap[x, y - 1] == 1
                    ):
                        holes += 1
        return holes


class AspectRatioHeuristic(ActionCostHeuristic):
    """How tall vs wide is the subgoal?

    This returns the aspect ratio of the subgoal.
    """

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the aspect ratio of the subgoal."""
        bitmap = subgoal.bitmap
        minx, maxx, miny, maxy = get_bitmap_bounds(bitmap)
        return (maxy - miny) / (maxx - minx)


class OuterHolesHeuristic(ActionCostHeuristic):
    """How many cells are holes are in the structure that touch the outside?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        bitmap = subgoal.bitmap
        bitmap = subset_bitmap(bitmap)
        holes = 0
        # how many holes on the vertical edges?
        for x in range(bitmap.shape[0]):
            if bitmap[x, 0] == 0:
                holes += 1
            if bitmap[x, -1] == 0:
                holes += 1
        # how many holes on the horizontal edges?
        for y in range(bitmap.shape[1]):
            if bitmap[0, y] == 0:
                holes += 1
            if bitmap[-1, y] == 0:
                holes += 1
        return holes


class NumberOfOuterHolesHeuristic(ActionCostHeuristic):
    """How many unique holes are in the structure that touch the outside?

    Analogous to the number of holes heuristic, but only counts holes that touch the outside.
    """

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        bitmap = subgoal.bitmap
        bitmap = subset_bitmap(bitmap)
        holes = 0
        # how many unique holes on the vertical edges?
        for x in range(bitmap.shape[0]):
            if bitmap[x, 0] == 0:
                if x == 0 or bitmap[x - 1, 0] == 1:
                    holes += 1
            if bitmap[x, -1] == 0:
                if x == 0 or bitmap[x - 1, -1] == 1:
                    holes += 1
        # how many unique holes on the horizontal edges?
        for y in range(bitmap.shape[1]):
            if bitmap[0, y] == 0:
                if y == 0 or bitmap[0, y - 1] == 1:
                    holes += 1
            if bitmap[-1, y] == 0:
                if y == 0 or bitmap[-1, y - 1] == 1:
                    holes += 1
        return holes


def get_bitmap_bounds(bitmap):
    """Get the bounds of the bitmap.

    Ie get the bounds of the parts of the bitmap that are not zero."""
    y_indices, x_indices = np.where(bitmap == 1)

    # If there are no 1s in the array, return None
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return x_min, y_min, x_max, y_max


def subset_bitmap(bitmap):
    """Subset bitmap to only return the subgoal and not the rest of the problem"""
    minx, maxx, miny, maxy = get_bitmap_bounds(bitmap)
    return bitmap[minx:maxx, miny:maxy]


SHAPE_HEURISTICS = [
    AreaSizeHeuristic,
    MassHeuristic,
    NumberOfHolesHeuristics,
    AspectRatioHeuristic,
    OuterHolesHeuristic,
    NumberOfOuterHolesHeuristic,
]
