"""This file contains a number of silhouettes and sets of baseblocks."""

# set up imports
from os.path import isfile, join
from os import listdir
from io import open
import json
import utils.blockworld as blockworld
import numpy as np
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
stimuli_dir = os.path.join(proj_dir, 'stimuli')


# the selection of interesting structures used in McCarthy 2020
SILHOUETTE8 = [4, 5, 6, 8, 9, 11, 12, 16]

# all 16 premade structures
SILHOUETTE16 = list(range(1, 17))

"""A couple of premade silhouettes."""


def load_silhouette_from_Json(path, dimensions=(13, 18)):
    """This is to load the premade structures  by Will in block_construction/stimuli/interesting_structures"""
    with open(path, 'r') as file:
        data = json.loads(file.read())
        blocks = data['blocks']
    silhouette = np.zeros(dimensions)
    for b in blocks:
        silhouette[dimensions[0]-(b['y']+b['height']):dimensions[0]-b['y'], b['x']:b['x']+b['width']] = 1
    return silhouette


def load_interesting_structure(number, dimensions=(8, 8)):
    """Loads a JSON structure from the folder block_construction/stimuli/interesting_structures by number. There are 16."""
    return load_silhouette_from_Json(os.path.join(stimuli_dir, 'interesting_structures/hand_selected_'+str(number).zfill(3)+'.json'), dimensions)


def plot_interesting_figures():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4)
    i = 0
    for _, axis in np.ndenumerate(axes):
        axis.imshow(load_interesting_structure(i))
        axis.set_title(str(i))
        i += 1
    plt.tight_layout()
    plt.show()


stonehenge_18_13 = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])

stonehenge_6_4 = np.array([
    [1., 1., 1., 1., ],
    [1., 1., 1., 1., ],
    [1., 0., 0., 1., ],
    [1., 0., 0., 1., ],
    [1., 0., 0., 1., ],
    [1., 0., 0., 1., ]
])

block = np.array([
    [1., 1., 1., 0., ],
    [1., 1., 1., 0., ],
    [1., 1., 1., 0., ],
    [1., 1., 1., 0., ],
])

T = np.array([
    [1., 1., 1., 1., ],
    [0., 1., 1., 0., ],
    [0., 1., 1., 0., ],
    [0., 1., 1., 0., ],
])

side_by_side = np.array([
    [1., 1., 1., 1., 0., 1., 1., 1., 1.],
    [0., 1., 1., 0., 0., 1., 1., 1., 1.],
    [0., 1., 1., 0., 0., 1., 1., 1., 1.],
    [0., 1., 1., 0., 0., 1., 1., 1., 1.],
])

stonehenge_3_3 = np.array([
    [1., 1., 1., ],
    [1., 0., 1., ],
    [1., 0., 1., ]
])

t_3_3 = np.array([
    [1., 1., 1., ],
    [0., 1., 0., ],
    [0., 1., 0., ]
])


def horizontal_tile(silhouette, reps=2):
    """Places two silhouettes side-by-side with an empty line in the middle."""
    silhouette = np.append(silhouette, np.zeros(
        (silhouette.shape[0], 1)), axis=1)
    return np.tile(silhouette, (1, reps))[:, :]


"""Base block libraries. width first, then height."""
# The defaults are taken from the silhouette2 study. Width x height.
bl_silhouette2_default = [
    blockworld.BaseBlock(1, 2),
    blockworld.BaseBlock(2, 1),
    blockworld.BaseBlock(2, 2),
    blockworld.BaseBlock(2, 4),
    blockworld.BaseBlock(4, 2),
]

# A block library that contains the simplest set of blocks such that larger blocks can't be composed of smaller ones.
bl_nonoverlapping_simple = [
    blockworld.BaseBlock(1, 2),
    blockworld.BaseBlock(2, 1),
    blockworld.BaseBlock(3, 1),
    blockworld.BaseBlock(1, 3)
]

# for convenience named version of the above block library.
bl_nonoverlapping_simple_named = {
    'v2': bl_nonoverlapping_simple[0],
    'h2': bl_nonoverlapping_simple[1],
    'h3': bl_nonoverlapping_simple[2],
    'v3': bl_nonoverlapping_simple[3]
}

# Non-overlapping library of appropiate complexity
bl_nonoverlapping_med = [
    blockworld.BaseBlock(2, 2),
    blockworld.BaseBlock(3, 2),
    blockworld.BaseBlock(1, 3),
    blockworld.BaseBlock(4, 1),
]

# beware, this might never finish, so don't run the agent in an infinte loop
bl_silhouette2_wait = [
    blockworld.BaseBlock(1, 2),
    blockworld.BaseBlock(2, 1),
    blockworld.BaseBlock(2, 2),
    blockworld.BaseBlock(2, 4),
    blockworld.BaseBlock(4, 2),
    blockworld.BaseBlock(0, 0)
]

bl_stonehenge_3_3 = [
    blockworld.BaseBlock(1, 2),
    blockworld.BaseBlock(3, 1),
]

bl_stonehenge_6_4 = [
    blockworld.BaseBlock(1, 2),
    blockworld.BaseBlock(4, 1),
]

bl_side_by_side = [
    blockworld.BaseBlock(2, 1),
    blockworld.BaseBlock(4, 1),
]
