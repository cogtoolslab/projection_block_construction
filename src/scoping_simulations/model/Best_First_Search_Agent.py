import os
import random
import sys

import scoping_simulations.utils.blockworld as blockworld
from scoping_simulations.model.Astar_Agent import Astar_Agent
from scoping_simulations.model.utils.Search_Tree import *

proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_dir)


class Best_First_Search_Agent(Astar_Agent):
    """An agent implementing best first search.
    Choose heuristic so that higher value means better state (ie F1 score).
    """

    def __init__(
        self,
        world=None,
        heuristic=blockworld.F1score,
        only_improving_actions=False,
        random_seed=None,
        shuffle=False,
        label="Best First",
    ):
        self.world = world
        self.heuristic = heuristic
        self.only_improving_actions = only_improving_actions
        self.random_seed = random_seed
        self.label = label
        if self.random_seed is None:
            self.random_seed = random.randint(0, 99999)

    def f(self, node):
        """Unlike A*, this is just the heuristic."""
        return self.heuristic(node) * -1
