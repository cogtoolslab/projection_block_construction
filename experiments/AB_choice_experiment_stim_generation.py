# %% [markdown]
# # Generating stimuli for A/B choice experiment

# %% [markdown]
# Purpose of this notebook is:
# * to create a set of towers
# * for each tower, create a tree of branching subgoal choices, which each subgoal on each turn being either the cheapest or the most expensive one meeting a certain condition.
#     * ensuring that each node has a path to the goal (can we do that?)
# * visualize the different choices
# 
# Requires:
# *
# 
# See also:
# * 

# %% [markdown]
# ## Setup

# %%
# set up imports
import os
import sys
__file__ = os.getcwd()
proj_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(proj_dir)
utils_dir = os.path.join(proj_dir, 'utils')
sys.path.append(utils_dir)
analysis_dir = os.path.join(proj_dir, 'analysis')
analysis_utils_dir = os.path.join(analysis_dir, 'utils')
sys.path.append(analysis_utils_dir)
agent_dir = os.path.join(proj_dir, 'model')
sys.path.append(agent_dir)
agent_util_dir = os.path.join(agent_dir, 'utils')
sys.path.append(agent_util_dir)
experiments_dir = os.path.join(proj_dir, 'experiments')
sys.path.append(experiments_dir)
df_dir = os.path.join(proj_dir, 'results/dataframes')
stim_dir = os.path.join(proj_dir, 'stimuli')

# %%
import stimuli.tower_generator as tower_generator

from tqdm import tqdm
import p_tqdm

import pickle

import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import scipy.stats as stats
from scipy.stats import sem as sem

from utils.blockworld_library import *
from utils.blockworld import *

from model.BFS_Lookahead_Agent import BFS_Lookahead_Agent
from model.BFS_Agent import BFS_Agent
from model.Astar_Agent import Astar_Agent
from model.Best_First_Search_Agent import Best_First_Search_Agent
from model.Subgoal_Planning_Agent import Subgoal_Planning_Agent

from model.utils.decomposition_functions import *
import utils.blockworld_library as bl



# %%
# show all columns in dataframe
pd.set_option('display.max_columns', None)

# %% [markdown]
# ## Generating towers
# 

# %%
block_library = bl_nonoverlapping_simple

# %%
generator = tower_generator.TowerGenerator(8, 8,
                                           block_library=block_library,
                                           seed=42,
                                           padding=(2, 0),
                                           num_blocks=lambda: random.randint(4, 10), #  flat random interval of tower sizes (inclusive)
                                           )


# %%
NUM_TOWERS  = 64
towers = []
for i in tqdm(range(NUM_TOWERS)):
    towers.append(generator.generate())

# %%
worlds = [Blockworld(silhouette=t['bitmap'], block_library=bl.bl_nonoverlapping_simple) for t in towers]

# %% [markdown]
# ### Visualize the generated towers

# %%
# look at towers
def visualize_towers(towers, text_parameters=None):
    fig,axes = plt.subplots(math.ceil(len(towers)/5),5,figsize=(20,15*math.ceil(len(towers)/20)))
    for axis, tower in zip(axes.flatten(),towers):
        axis.imshow(tower['bitmap']*1.0)
        if text_parameters is not None:
            if type(text_parameters) is not list:
                text_parameters = [text_parameters]
            for y_offset,text_parameter in enumerate(text_parameters):
                axis.text(0,y_offset*1.,str(text_parameter+": "+str(tower[text_parameter])),color='gray',fontsize=20)
    plt.tight_layout()
    plt.show()

# %%
# visualize_towers(towers)

# %% [markdown]
# ## Score towers for basic difficulty
# For each tower, compute the cost of solving it using a planning agent.

# %% [markdown]
# Here, we use Best First Search without lookahead or subgoals.

# %%
lower_agent = Best_First_Search_Agent(random_seed=42)

# %%
def get_tower_cost(agent,world):
    cost = 0
    agent.set_world(world)
    world.reset()
    while world.status()[0] == 'Ongoing':
        _,step_info = agent.act()
        cost += step_info['states_evaluated']
    return cost,world.status()

# %%
costs = []
statusses = []
for world in tqdm(worlds):
    cost,status = get_tower_cost(lower_agent,world)
    costs.append(cost)
    statusses.append(status)

# %% [markdown]
# Split the basic costs into three percentiles: easy, medium, hard.

# %%
difficulty_percentiles = [np.percentile(costs, i)
               for i in [33, 66, 99]]

percentiles = [None] * len(costs)
for i, cost in enumerate(costs):
    if cost < difficulty_percentiles[0]:
        percentiles[i] = 'easy'
    elif cost < difficulty_percentiles[1]:
        percentiles[i] = 'medium'
    else:
        percentiles[i] = 'hard'

# %% [markdown]
# ## Find best and worst sequence of subgoals for each tower
# We compute the full subgoal tree for each tower and extract the best and worst sequence.

# %%
decomposer = Rectangular_Keyholes(
    sequence_length=4,
    necessary_conditions=[
        Area_larger_than(area=1),
        Area_smaller_than(area=21),
        No_edge_rows_or_columns(),
    ],
    necessary_sequence_conditions=[
        Complete(),
        No_overlap(),
        Supported(),
    ]
)

sg_agent = Subgoal_Planning_Agent(lower_agent=lower_agent,
    random_seed=42,
    decomposer=decomposer)



# %% [markdown]
# Calculate the subgoal tree for each tower.
# 
# Sadly, the sockets seem to make this hard to parallelize.

# %%
# # parallelized—does not presently work (somehow the sockets in p_tqdm just don't work)
# def get_subgoal_tree_from_tower(agent, world):
#     agent.set_world(world)
#     return agent.get_subgoal_tree()

# agents = [copy.deepcopy(a) for a in [sg_agent]*len(worlds)]

# trees = p_tqdm.p_map(get_subgoal_tree_from_tower, agents, worlds)

# %%
# sequential version
trees = []
for world in tqdm(worlds):
    world.reset()
    sg_agent.set_world(world)
    trees.append(sg_agent.get_subgoal_tree())

# %% [markdown]
# Visualize the best and worst sequence of subgoals for each tower.

# %%
for i, tree in enumerate(trees):
    print("Tower {}".format(i))
    best_seq = tree.get_best_sequence()
    try:
        print("Best sequence with cost",best_seq.solution_cost(),"for tower",i)
    except:
        print("No Best sequence for tower",i)
    worst_seq = tree.get_worst_sequence()
    try:
        print("Worst sequence with cost",worst_seq.solution_cost(),"for tower",i)
    except:
        print("No Worst sequence for tower",i)


# %% [markdown]
# Let's save out everything

# %%
results = [{'world':world,'subgoal tree':tree,'cost':cost,'percentile':percentile} for world,tree,cost,percentile in zip(worlds,trees,costs,percentiles)]

# %%
pickle.dump(results, open("AB_choice subgoal results.pkl", "wb"))

# %%



