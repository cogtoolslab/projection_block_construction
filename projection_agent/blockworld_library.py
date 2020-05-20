"""This file contains a number of silhuoettes and sets of baseblocks."""
import numpy as np
import blockworld

"""A couple of premade silhuouettes."""
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
    [1., 1., 1., 1.,],
    [1., 1., 1., 1.,],
    [1., 0., 0., 1.,],
    [1., 0., 0., 1.,],
    [1., 0., 0., 1.,],
    [1., 0., 0., 1.,]
])


stonehenge_3_3 = np.array([
    [1.,1., 1.,],
    [1., 0., 1.,],
    [1., 0., 1.,]
])

t_3_3 =  np.array([
    [1.,1., 1.,],
    [0., 1., 0.,],
    [0., 1., 0.,]
])


"""Base block libraries."""
#The defaults are taken from the silhouette2 study.
silhouette2_default_blocklibrary= [
    blockworld.BaseBlock(1,2),
    blockworld.BaseBlock(2,1),
    blockworld.BaseBlock(2,2),
    blockworld.BaseBlock(2,4),
    blockworld.BaseBlock(4,2),
]

stonehenge_3_3_blocklibrary = [
    blockworld.BaseBlock(1,2),
    blockworld.BaseBlock(3,1),
] 

stonehenge_6_4_blocklibrary = [
    blockworld.BaseBlock(1,2),
    blockworld.BaseBlock(4,1),
] 