import os
import sys
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from model.Astar_Agent import Astar_Agent, Stochastic_Priority_Queue
from model.utils.Search_Tree import *
import utils.blockworld as blockworld
import random
from dataclasses import dataclass, field
from typing import Any
from statistics import mean

class Best_First_Search_Agent(Astar_Agent):
    """An agent implementing best first search. 
        Choose heuristic so that higher value means better state (ie F1 score).
        """

    def __init__(self,world = None,heuristic=blockworld.F1score, only_improving_actions = False, random_seed=None,shuffle=False):
        self.world = world
        self.heuristic = heuristic
        self.only_improving_actions = only_improving_actions
        self.random_seed = random_seed
        if self.random_seed is None: self.random_seed = random.randint(0,99999)

    def f(self,node):
        """Unlike A*, this is just the heuristic.
        """
        return self.heuristic(node) * -1