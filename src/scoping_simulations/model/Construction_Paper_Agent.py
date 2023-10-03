import copy
import random

import numpy as np

from scoping_simulations.model.BFS_Lookahead_Agent import *

# Decomposition functions
# pass these as arguments to the agent
# they take the agent as the first argument
# they should return a dictionary as second argument with information about the decomposition
# h/horizontal refers to horizontal construction paper


def horizontal_construction_paper_holes(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world.  `Relative` calculates the last block as the basis of where the construction paper is placed. This means it can get stuck in an infinite loop when nothing can be legally built in the proposed decomposition.

    The current position of the "construction paper" is inferered by comparing the what is currently built to the silhouette.


    ```
    world = new world
    full_silhouette = world.silhouette
    low_level_agent = new breadth_first_search_agent
    paper_y = 0
    while w.status is ongoing:
        # "slide up the paper" until the last point where the scene would no longer be bisected into two separate structures
        while full_silhouette.up_to(paper_y) is fully connected:
            paper_y += 1
        while full_silhouette.up_to(paper_y) is not fully connected:
            paper_y += 1
        # run lower level agent on world with silhouette up to a certain y
        temp_world = world
        temp_world.silhouette = full_silhouette.up_to(paper_y)
        low_level_agent.act(world)
        # handling a failed decomposition
        if temp_world.status is failure:
            #the attempted decomposition failed, so we try another
            paper_y += 1
        else:
            #make the change official
            world = temp_world
    ```

    In words, it tries to decompose by sliding the construction paper as far up as it can so it still decomposing the resulting silhouette. Intuitively, it places the lower edge of the construction at the upper edge of the first hole from the bottom up, then the second and so on. If it fails to build the resulting partial silhouette, it just tries the next position further up.
    """
    if current_built is None:
        current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0:
        y += 1
    start_y = y
    # slide up paper to find edge
    # get map of rows containing a vertical edge with built on top and hole below
    row_map = find_edges(full_silhouette)
    while y > 0:
        y = y - 1
        if row_map[y]:
            break
    end_y = y
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y, :] = 0
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": abs(start_y - end_y),
        "decomposition_base_function": "horizontal_construction_paper_holes",
    }


def vertical_construction_paper_holes(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world.  `Relative` calculates the last block as the basis of where the construction paper is placed. This means it can get stuck in an infinite loop when nothing can be legally built in the proposed decomposition.
    Same as `horizontal_construction_paper_holes`, but the construction paper is moved from left to right.
    We don't expect this to be successful. This is just to investigate other ways of chunking the structure.
    """
    if current_built is None:
        current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # rotate the matrices
    full_silhouette = np.rot90(full_silhouette, k=1)
    current_built = np.rot90(current_built, k=1)
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0:
        y += 1
    start_y = y
    # slide up paper to find edge
    # get map of rows containing a vertical edge with built on top and hole below
    row_map = find_edges(full_silhouette)
    while y > 0:
        y = y - 1
        if row_map[y]:
            break
    end_y = y
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y, :] = 0
    # rotate back
    new_silhouette = np.rot90(new_silhouette, k=-1)
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": abs(start_y - end_y),
        "decomposition_base_function": "vertical_construction_paper_holes",
    }


def _random_decomposition_h(self, current_built=None, lower=1, upper=5):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a random increment given. Call this from wrapper functions."""
    if current_built is None:
        current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0:
        y += 1
    # increment y
    increment = random.randint(lower, upper)
    y = y - increment
    # limit to height of area
    y = min(y, full_silhouette.shape[0])
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y, :] = 0
    return new_silhouette, increment


def _random_decomposition_v(self, current_built=None, lower=1, upper=5):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a random increment given. Call this from wrapper functions."""
    if current_built is None:
        current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    full_silhouette = np.rot90(full_silhouette, k=1)
    current_built = np.rot90(current_built, k=1)
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0:
        y += 1
    # increment y
    increment = random.randint(lower, upper)
    y = y - increment
    # limit to height of area
    y = min(y, full_silhouette.shape[0])
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y, :] = 0
    new_silhouette = np.rot90(new_silhouette, k=-1)
    return new_silhouette, increment


def random_1_4_h(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a random increment between 1 and 4."""
    new_silhouette, increment = _random_decomposition_h(self, current_built, 1, 4)
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_function": "random_1_4_h",
    }


def random_2_4_h(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a random increment between 1 and 4."""
    new_silhouette, increment = _random_decomposition_h(self, current_built, 2, 4)
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_function": "random_2_4_h",
    }


def random_1_4_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a random increment between 1 and 4."""
    new_silhouette, increment = _random_decomposition_v(self, current_built, 1, 4)
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_function": "random_1_4_v",
    }


def random_2_4_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a random increment between 1 and 4."""
    new_silhouette, increment = _random_decomposition_v(self, current_built, 2, 4)
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_function": "random_2_4_v",
    }


def random_2_4_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a random increment between 1 and 4."""
    new_silhouette, increment = _random_decomposition_v(self, current_built, 2, 4)
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_function": "random_2_4_v",
    }


def _fixed_h(self, increment, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment."""
    if current_built is None:
        current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # get last placement
    try:
        y = self.world._construction_paper_loc
    except AttributeError:  # we haven't placed the construction paper yet
        y = full_silhouette.shape[0]
    # increment y
    y = y - increment
    # limit to height of area
    y = min(y, full_silhouette.shape[0])
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y, :] = 0
    # save last location
    self.world._construction_paper_loc = y
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_base_function": "fixed_horizontal",
    }


def _fixed_v(self, increment, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a fixed increment."""
    if current_built is None:
        current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    full_silhouette = np.rot90(full_silhouette, k=1)
    current_built = np.rot90(current_built, k=1)
    # get last placement
    try:
        y = self.world._construction_paper_loc
    except AttributeError:  # we haven't placed the construction paper yet
        y = full_silhouette.shape[0]
    # increment y
    y = y - increment
    # limit to height of area
    y = min(y, full_silhouette.shape[0])
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y, :] = 0
    new_silhouette = np.rot90(new_silhouette, k=-1)
    # save last location
    self.world._construction_paper_loc = y
    return new_silhouette, {
        "decomposed_silhouette": new_silhouette,
        "decomposition_increment": increment,
        "decomposition_base_function": "fixed_vertical",
    }


def fixed_1_h(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment."""
    return _fixed_h(self, 1, current_built)


def fixed_2_h(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment."""
    return _fixed_h(self, 2, current_built)


def fixed_3_h(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment."""
    return _fixed_h(self, 3, current_built)


def fixed_4_h(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment."""
    return _fixed_h(self, 4, current_built)


def fixed_1_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a fixed increment."""
    return _fixed_v(self, 1, current_built)


def fixed_2_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a fixed increment."""
    return _fixed_v(self, 2, current_built)


def fixed_3_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a fixed increment."""
    return _fixed_v(self, 3, current_built)


def fixed_4_v(self, current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper vertically upwards by a fixed increment."""
    return _fixed_v(self, 4, current_built)


def no_decomposition(self, current_built=None):
    """Returns the full silhouette. Provides the baseline of using no decomposition altogether."""
    return self.world.silhouette, {"decomposition_base_function": "no_decomposition"}


def half_v(self, current_built=None):
    """Vertically decomposes the silhouette into the left and the right half"""
    return _fixed_v(self, int(self.world.silhouette.shape[1] / 2), current_built)


# def crop(arr,(bl_x,bl_y),(tr_x,tr_y)):
#     """Crops the array that is passed to it. The first tuple marks the x and y coordinates of the bottom left corner, the other the top right corner. Note that the top left corner is (0,0)."""
#     assert(
#         arr.shape
#     )


# Agent
class Construction_Paper_Agent(BFS_Lookahead_Agent):
    """Implements the construction paper proposal for a projection based agent.

    TODO
    - [ ] retry on failure
    - [ ] save the decompositions

    Different decomposition functions can be passed to the agent.
    """

    def __init__(
        self,
        world=None,
        lower_agent=BFS_Lookahead_Agent(only_improving_actions=True),
        decomposition_function=horizontal_construction_paper_holes,
    ):
        self.world = world
        self.lower_agent = lower_agent
        self.decompose = decomposition_function

    def __str__(self):
        """Yields a string representation of the agent"""
        return (
            self.__class__.__name__
            + " lower level agent: "
            + self.lower_agent.__str__()
            + " decomposition function"
            + self.decompose.__name__
        )

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            **{
                "agent_type": self.__class__.__name__,
                "decomposition_function": self.decompose.__name__,
            },
            **{
                "lower level: " + key: value
                for key, value in self.lower_agent.get_parameters().items()
            },
        }

    def act(self, steps=1, verbose=False):
        """Makes the agent act. This means acting for a given number of steps, where higher_steps refers to higher level of the agent (x times running the lower level agent) and lower_steps refers to the total number of steps performed by the lower level agent, ie actual steps in the world.

        The actual action happens in act_single_higher_step"""
        if steps != 1:
            print(
                "Using a step that isn't one will compute cost and decomposition info only for the last step for CPA"
            )
        higher_step = 0
        actions = []
        costs = 0
        decompose_step_infos = []
        while (
            higher_step != steps and self.world.status()[0] == "Ongoing"
        ):  # action loop
            action, cost, decompose_step_info = self.act_single_higher_step(verbose)
            actions += action
            costs += cost
            higher_step += 1
            decompose_step_infos.append(decompose_step_info)
        return actions, {"states_evaluated": costs, **decompose_step_info}

    def act_single_higher_step(self, verbose):
        """Takes a single step of the higher level agent. This means finding a new decomposition and then running the lower level agent on it. If it fails, it jumps to the next decomposition.

        The current position of the "construction paper" is inferered by comparing the what is currently built to the silhouette.
        """
        # get decomposition
        new_silhouette, decompose_step_info = self.decompose(
            self
        )  # external function needs to explicitly passed the agent object
        if verbose:
            print("Got decomposition\n ", new_silhouette)
        # create temporary world object containing the modified silhouette
        temp_world = copy.deepcopy(self.world)
        temp_world.silhouette = new_silhouette
        self.lower_agent.set_world(temp_world)
        # let's run the lower level agent
        action_seq = []
        costs = 0
        while temp_world.status()[0] == "Ongoing":
            action, agent_step_info = self.lower_agent.act(verbose=verbose)
            cost = agent_step_info["states_evaluated"]
            action_seq += action
            costs += cost
            if action == []:
                break  # stop the loop if no further action is possible. This means that a perfect reconstruction of the temp world was not achieved
        # apply actions to the world
        if verbose:
            print(
                "Decomposition done, applying action_seq:",
                str([str(a) for a in action_seq]),
                "with world state",
                temp_world.status()[0],
            )
        for action in action_seq:
            self.world.apply_action(
                action, force=True
            )  # we need force here since the baseblock objects are in different memory locations
        # we need to carry over the decomposition status to the actual world
        try:
            self.world.current_state._construction_paper_loc = (
                temp_world.current_state._construction_paper_loc
            )
        except AttributeError:
            pass
        return (
            action_seq,
            costs,
            decompose_step_info,
        )  # only returning cost here, not the other parameters of the lower level agent. That could be changed, but that would require allowing to pass a list to experiment_runner, and that's too complicated


# Helper functions


def find_edges(silhouette):
    """Returns a map for each row in the table that has an edge with a filled out portion on the upper side and empty space on the lower side."""
    row_map = [False] * silhouette.shape[0]
    # go thru every row of the silhouette
    for y in range(silhouette.shape[0]):
        for x in range(silhouette.shape[1]):
            if silhouette[y, x] == 0 and silhouette[y - 1, x] != 0:
                row_map[y] = True
                break
    row_map[0] = True  # always offer the top edge as potential decomposition
    return row_map
