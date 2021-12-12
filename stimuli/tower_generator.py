import numpy as np
import random

# we need to import from the parent path
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_dir)

from utils.blockworld_library import bl_silhouette2_default,bl_nonoverlapping_simple
from utils.blockworld import *


def _default_size():
    return int(np.ceil(np.random.normal(6, 1)))


class TowerGenerator():
    """
    Generates a tower of blocks.
    """

    def __init__(self, height: int, width: int,
                 block_library=bl_nonoverlapping_simple,  # list of blocks to choose from
                 # function that samples from the distribution of numbers of blocks per tower. Can also pass list or int
                 num_blocks=_default_size,
                 # is called after each block is placed and returns true for placements we want to allow
                 evaluator=lambda x: True,
                 block_selector=random.choice,  # gets the list of blocks and choses one,
                 physics=True,  # do we care about the stability of the tower?
                 max_steps = 1000, # the upper limit of steps to take
                 seed=None,
                 ):
        """
        Initializes the tower generator.
        """
        self.height = height
        self.width = width
        self.evaluator = evaluator
        self.block_selector = block_selector
        self.block_library = block_library
        self.num_blocks = num_blocks
        self.physics = physics
        self.max_steps = max_steps
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self):
        """ Generates a single tower """
        # initialize empty world
        world = Blockworld(dimension=np.array([self.height, self.width]),
                           block_library=self.block_library,
                           legal_action_space=False,
                           physics=self.physics,
                           )
        # sample number of blocks
        if type(self.num_blocks) is int:
            num_blocks = self.num_blocks
        elif callable(self.num_blocks):
            num_blocks = self.num_blocks()
        elif type(self.num_blocks) is list:
            num_blocks = random.choice(self.num_blocks)
        else:
            raise ValueError("num_blocks must be int, function or list, not {}".format(type(self.num_blocks)))
        for i in range(self.max_steps):
            # first, any blocks left to place?
            if num_blocks == 0:
                break
            # get all actions
            # currently defined as a tuple of a baseblock and a x location
            potential_actions = world.current_state.possible_actions()
            if len(potential_actions) == 0: # no more possible actions
                break
            # TODO for now randomyl choosing an action, make it choose location and block according to provided functions
            action = random.choice(potential_actions)
            # take action
            new_state = world.transition(action)
            # evaluate the new state
            if world.stability(new_state):
                # the block placement is stable, so we can place it
                world.apply_action(action)
                num_blocks -= 1
            else:
                # we try again
                continue
        # we should be done
        # get number of blocks
        num_blocks = len(world.current_state.blocks)
        assert num_blocks > 0, "Empty world"
        # generate silhouette object
        silhouette = {}
        silhouette['blocks'] = world.current_state.blocks
        silhouette['blockmap'] = world.current_state.blockmap
        silhouette['block_library'] = self.block_library
        silhouette['dimension'] = (self.height, self.width)
        silhouette['bitmap'] = (world.current_state.blockmap > 0).astype(float)
        return silhouette
