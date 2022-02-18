from model.utils.Search_Tree import *
import utils.blockworld as blockworld
from itertools import repeat
import random
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_dir)


class BFS_Agent:
    """An agent performing exhaustive BFS search. This can take a long time to finish."""

    def __init__(self, world=None, shuffle=False, random_seed=None, label="BFS"):
        self.world = world
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.label = label

    def __str__(self):
        """Yields a string representation of the agent"""
        return self.__class__.__name__+' shuffle:'+str(self.shuffle)+' random seed: '+str(self.random_seed) + ' label: ' + self.label

    def set_world(self, world):
        self.world = world

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            'agent_type': self.__class__.__name__,
            'shuffle': self.shuffle,
            'random_seed': self.random_seed,
            'label': self.label
        }

    def search(self, current_nodes):
        """Performs one expansion of the nodes in current nodes. Returns either list of expanded nodes, found solution node or empty list. To introduce randomness, the current nodes can be shuffled."""
        cost = 0  # track number of states that are evaluated
        if self.shuffle:
            random.seed(self.random_seed)  # fix random seed
            random.shuffle(current_nodes)
        next_nodes = []  # holds the nodes we get from the current expansion step
        for node in current_nodes:  # expand current nodes
            possible_actions = node.state.possible_actions()
            children = []
            for action in possible_actions:
                child = Node(node.state.transition(action),
                             node.actions+[action])  # generate new node
                # check if child node is winning
                cost += 1
                if child.state.is_win():
                    # we've found a winning state
                    return "Winning", child, cost
                next_nodes.append(child)
        return "Ongoing", next_nodes, cost

    def act(self, steps=None, verbose=False):
        """Makes the agent act, including changing the world state."""
        # Ensure that we have a random seed if none is set
        states_evaluated = 0
        if self.random_seed is None:
            self.random_seed = random.randint(0, 99999)
        # check if we even can act
        if self.world.status()[0] != 'Ongoing':
            print("Can't act with world in status", self.world.status())
            return [], {'states_evaluated': states_evaluated}
        if steps is not None:
            print(
                "Limited number of steps selected. This is not lookahead, are you sure?")
        # perform BFS search
        # initialize root node
        current_nodes = [Node(self.world.current_state, [])]
        result = "Ongoing"
        while current_nodes != [] and result == "Ongoing":
            # keep expanding until solution is found or there are no further states to expand
            result, out, cost = self.search(
                current_nodes)  # run expansion step
            states_evaluated += cost
            if result != "Winning":
                current_nodes = out  # we have no solution, just the next states to expand
                if verbose:
                    print("Found", len(current_nodes),
                          "to evaluate at cost", cost)
        # if we've found a solution
        if result == "Winning":
            actions = out.actions
            # extract the steps to take. None gives complete list
            actions = actions[0:steps]
            if verbose:
                print("Found solution with ", len(actions), "actions")
            # apply steps to world
            for action in actions:
                self.world.apply_action(action)
            if verbose:
                print("Done, reached world status: ", self.world.status())
            # only returns however many steps we actually acted, not the entire sequence
        else:
            actions = []
            if verbose:
                print("Found no solution")
        return actions, {'states_evaluated': states_evaluated}
