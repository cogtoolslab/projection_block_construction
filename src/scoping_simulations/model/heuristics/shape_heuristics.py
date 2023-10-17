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
        minx, maxx, miny, maxy = get_bitmap_bounds(bitmap)
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


class NumberOfHolesHeuristic(ActionCostHeuristic):
    """How many holes (of whatever size) are in the bitmap?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the number of holes in the subgoal."""
        # subset and pad with one
        target = pad_by_one(subset_target(subgoal.bitmap, subgoal.target))
        # iterate through all cells and check if they are holes
        holes = 0
        # we are going down and left
        for x in range(1, target.shape[0]):
            for y in range(1, target.shape[1]):
                if (
                    target[x, y] == 0  # it's a hole
                    # continuation from the left
                    and not target[x - 1, y] == 0
                    # continuation from the top
                    and not target[x, y - 1] == 0
                ):
                    holes += 1
        return holes


class HolesHeuristic(ActionCostHeuristic):
    """What is the total area of holes in the bitmap?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the total area of holes in the subgoal."""
        # iterate through all cells and check if they are holes
        target = subset_target(subgoal.bitmap, subgoal.target)
        holes = target == 0
        return holes.sum()


class AspectRatioHeuristic(ActionCostHeuristic):
    """How tall vs wide is the subgoal?

    This returns the aspect ratio of the subgoal.
    Calculated as height / width,
        so a tall subgoal will have a value > 1.
    """

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the aspect ratio of the subgoal."""
        bitmap = subgoal.bitmap
        bounds = get_bitmap_bounds(bitmap)
        if bounds is None:
            return None
        minx, maxx, miny, maxy = bounds
        return (maxy - miny) / (maxx - minx)


class OuterHolesHeuristic(ActionCostHeuristic):
    """How many cells are holes are in the structure that touch the outside?"""

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        target = subset_target(subgoal.bitmap, subgoal.target)
        holes = 0
        # how many holes on the vertical edges?
        for x in range(target.shape[0]):
            if target[x, 0] == 0:
                holes += 1
            if target[x, -1] == 0:
                holes += 1
        # how many holes on the horizontal edges?
        for y in range(target.shape[1]):
            if target[0, y] == 0:
                holes += 1
            if target[-1, y] == 0:
                holes += 1
        # are we double counting the outer corners?
        if target[0, 0] == 0:
            holes -= 1
        if target[-1, 0] == 0:
            holes -= 1
        if target[-1, -1] == 0:
            holes -= 1
        if target[0, -1] == 0:
            holes -= 1
        return holes


class NumberOfOuterHolesHeuristic(ActionCostHeuristic):
    """How many unique holes are in the structure that touch the outside?

    Analogous to the number of holes heuristic, but only counts holes that touch the outside.
    """

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        target = subset_target(subgoal.bitmap, subgoal.target)
        holes = 0
        # how many unique holes on the vertical edges?
        for x in range(target.shape[0]):
            if target[x, 0] == 0:
                if x == 0 or target[x - 1, 0] > 0.5:
                    holes += 1
            if target[x, -1] == 0:
                if x == 0 or target[x - 1, -1] > 0.5:
                    holes += 1
        # how many unique holes on the horizontal edges?
        for y in range(target.shape[1]):
            if target[0, y] == 0:
                if y == 0 or target[0, y - 1] > 0.5:
                    holes += 1
            if target[-1, y] == 0:
                if y == 0 or target[-1, y - 1] > 0.5:
                    holes += 1
                # are we double counting the outer corners?
        if target[0, 0] == 0:
            holes -= 1
        if target[-1, 0] == 0:
            holes -= 1
        if target[-1, -1] == 0:
            holes -= 1
        if target[0, -1] == 0:
            holes -= 1
        return holes


def subset_bitmap(bitmap):
    """Subset bitmap to only return the subgoal and not the rest of the problem"""
    bounds = get_bitmap_bounds(bitmap)
    if bounds is None:
        return None
    minx, maxx, miny, maxy = bounds
    return bitmap[minx:maxx, miny:maxy]


def subset_target(bitmap, target):
    """Subset to only the target contained in the bitmap"""
    bounds = get_bitmap_bounds(bitmap)
    if bounds is None:
        return None
    minx, maxx, miny, maxy = bounds
    return target[minx:maxx, miny:maxy]


def pad_by_one(array):
    """Adds a row and column of zeros around the left and top  of the array"""
    # pad the array by one on the left and top
    array = np.pad(array, ((1, 0), (1, 0)), mode="constant", constant_values=1.0)
    return array


def get_bitmap_bounds(bitmap):
    """Get the bounds of the bitmap.

    Ie get the bounds of the parts of the bitmap that are not zero."""
    x_indices, y_indices = np.where(bitmap != 0)

    # If there are no non-zero values in the array, return None
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return x_min, x_max + 1, y_min, y_max + 1


SHAPE_HEURISTICS = [
    AreaSizeHeuristic,
    MassHeuristic,
    HolesHeuristic,
    NumberOfHolesHeuristic,
    AspectRatioHeuristic,
    OuterHolesHeuristic,
    NumberOfOuterHolesHeuristic,
]
