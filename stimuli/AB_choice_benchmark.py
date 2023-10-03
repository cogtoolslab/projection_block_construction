# # Benchmarking the speed of subgoal tree generation

# Purpose of this notebook is:
# * to create a set of towers
# * for each tower, create a tree of branching subgoal choices, which each subgoal on each turn being either the cheapest or the most expensive one meeting a certain condition.
#     * ensuring that each node has a path to the goal (can we do that?)
# * visualize the different choices
# * create a list of **subgoals** with an associated world state, not necessarily the best and worst sequence.
# * save that out to a pickled dataframe for the upload notebook in the `_human_expperiment` repo
#
# Requires:
# *
#
# See also:
# *

# ## Setup

# set up imports
import os

__file__ = os.getcwd()
from scoping_simulations.utils.directories import PROJ_DIR

DF_DIR = os.path.join(PROJ_DIR, "results/dataframes")
import datetime

import p_tqdm
import pandas as pd
import tower_generator
from scipy.stats import sem as sem
from tqdm import tqdm

import scoping_simulations.utils.blockworld_library as bl
from scoping_simulations.model.Best_First_Search_Agent import Best_First_Search_Agent
from scoping_simulations.model.Subgoal_Planning_Agent import Subgoal_Planning_Agent
from scoping_simulations.model.utils.decomposition_functions import *
from scoping_simulations.utils.blockworld import *
from scoping_simulations.utils.blockworld_library import *

# Usually we would fix the random seeds here, but the agents are being run with fixed random seeds, so this is not necessary here.

# show all columns in dataframe
pd.set_option("display.max_columns", None)

# ## Generating towers
#


def get_tower_cost(agent, world):
    cost = 0
    agent.set_world(world)
    world.reset()
    while world.status()[0] == "Ongoing":
        _, step_info = agent.act()
        cost += step_info["states_evaluated"]
    return cost, world.status()


def get_subgoal_tree_from_tower(agent, world):
    world.reset()
    agent.set_world(world)
    return agent.get_subgoal_tree(only_solved_sequences=True)


block_library = bl_nonoverlapping_simple

tower_size = 0

while True:
    tower_size += 1
    start_time = datetime.datetime.now()
    print("---------------")
    print("Generating tower of size:", tower_size)

    generator = tower_generator.TowerGenerator(
        8,
        8,
        block_library=block_library,
        seed=3,
        padding=(1, 0),
        num_blocks=lambda: tower_size,  #  flat random interval of tower sizes (inclusive)
    )

    NUM_TOWERS = 10
    towers = []
    print("Generating towers")
    for i in tqdm(range(NUM_TOWERS)):
        tower = generator.generate()
        towers.append(tower)

    worlds = [
        Blockworld(silhouette=t["bitmap"], block_library=bl.bl_nonoverlapping_simple)
        for t in towers
    ]

    lower_agent = Best_First_Search_Agent(random_seed=42)

    costs = []
    statusses = []
    print("Solving towers once (parallel)")
    # parallelized
    agents = [copy.deepcopy(a) for a in [lower_agent] * len(worlds)]

    # remove process connection before handoff to the threads
    for world in worlds:
        world.physics_provider.kill_server(force=True)

    results = p_tqdm.p_map(get_tower_cost, agents, worlds)
    costs = [c[0] for c in results]
    statusses = [c[1] for c in results]

    # Split the basic costs into three percentiles: easy, medium, hard.

    # ## Find best and worst subgoals
    # We compute the full subgoal tree for each tower and extract the best and worst sequence.
    #
    # Note: for the planned studies, we will use individual states and subgoals, not sequences of subgoals.

    decomposer = Rectangular_Keyholes(
        sequence_length=3,
        necessary_conditions=[
            Area_larger_than(area=1),
            # Area_smaller_than(area=30), # used to be 21
            Mass_smaller_than(area=16),
            No_edge_rows_or_columns(),
        ],
        necessary_sequence_conditions=[
            Complete(),
            No_overlap(),
            Supported(),
        ],
    )

    sg_agent = Subgoal_Planning_Agent(
        lower_agent=lower_agent, random_seed=42, decomposer=decomposer
    )

    # Calculate the subgoal tree for each tower.

    agents = [copy.deepcopy(a) for a in [sg_agent] * len(worlds)]

    # remove process connection before handoff to the threads
    for world in worlds:
        world.physics_provider.kill_server(force=True)

    print("Generating subgoal trees (parallelized)")
    trees = p_tqdm.p_map(get_subgoal_tree_from_tower, agents, worlds)

    # ### Select most divergent subgoals
    # For each tower, select n state/subgoal combinations that are maximally divergent.

    NUM_SGs_PER_TOWER = 6  # how many subgoals do we want to choose?

    best_worst_subgoals = []
    for tree in trees:
        best_worst_subgoals.append(
            tree.get_most_divergent_matching_pairs_of_subgoals(NUM_SGs_PER_TOWER)
        )

    ## results
    total_time = datetime.datetime.now() - start_time
    print(
        "Finished for {} towers of size {} in {}".format(
            NUM_TOWERS, tower_size, total_time
        )
    )
