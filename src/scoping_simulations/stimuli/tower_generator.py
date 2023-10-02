# we need to import from the parent path
import os
import random
import sys

import numpy as np

proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_dir)

from scoping_simulations.utils.blockworld import *
from scoping_simulations.utils.blockworld_library import bl_nonoverlapping_simple


def _default_size():
    return int(np.ceil(np.random.normal(6, 1)))


class TowerGenerator:
    """
    Generates a tower of blocks.
    """

    def __init__(
        self,
        height: int,
        width: int,
        block_library=bl_nonoverlapping_simple,  # list of blocks to choose from
        # function that samples from the distribution of numbers of blocks per tower. Can also pass list or int
        num_blocks=_default_size,
        # is called after each block is placed and returns true for placements we want to allow
        evaluator=lambda x: True,
        block_selector=random.choice,  # gets the list of blocks and choses one,
        physics=True,  # do we care about the stability of the tower?
        max_steps=1000,  # the upper limit of steps to take
        seed=None,
        padding=(
            0,
            0,
        ),  # padding applied to the x and y dimension on each side. X padding is applied twice, y padding is applied once (at the top). The original dimension is kept
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
        self.x_padding = padding[0]
        self.y_padding = padding[1]
        assert self.x_padding * 2 < self.width, "Padding doesn't leave width for blocks"
        assert self.y_padding < self.height, "Padding doesn't leave height for blocks"
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self):
        """Generates a single tower"""
        # initialize empty world
        world = Blockworld(
            dimension=np.array(
                [self.height - self.y_padding, self.width - self.x_padding * 2]
            ),
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
            raise ValueError(
                "num_blocks must be int, function or list, not {}".format(
                    type(self.num_blocks)
                )
            )
        for i in range(self.max_steps):
            # first, any blocks left to place?
            if num_blocks == 0:
                break
            # get all actions
            # currently defined as a tuple of a baseblock and a x location
            potential_actions = world.current_state.possible_actions()
            if len(potential_actions) == 0:  # no more possible actions
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
            elif len(potential_actions) == 1:
                # we've only got one option, and it's bad
                # need to start over
                # we do that by calling generate again
                return self.generate()
            else:
                # we try again
                continue
        # we should be done
        # get number of blocks
        num_blocks = len(world.current_state.blocks)
        assert num_blocks > 0, "Empty world"
        if self.physics:
            assert world.current_state.stability(), "Tower is unstable"
        # apply the padding
        # blocks
        blocks = copy.deepcopy(world.current_state.blocks)
        for block in blocks:
            block.x += self.x_padding
            block.y += self.y_padding
            for vert_i in range(block.verts.shape[0]):
                block.verts[vert_i][0] += self.x_padding
                block.verts[vert_i][1] += self.y_padding
        # for
        # blockmap
        blockmap = copy.copy(world.current_state.blockmap)
        blockmap = np.pad(
            blockmap, ((self.y_padding, 0), (self.x_padding, self.x_padding))
        )
        # generate silhouette object
        silhouette = {}
        silhouette["blocks"] = blocks
        silhouette["blockmap"] = blockmap
        silhouette["block_library"] = self.block_library
        silhouette["dimension"] = (self.height, self.width)
        silhouette["bitmap"] = (blockmap > 0).astype(float)
        return silhouette
