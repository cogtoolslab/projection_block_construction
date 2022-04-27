# %% [markdown]
# # Generating stimuli for A/B choice experiment, given_subgoal experiment

# %% [markdown]
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
import tower_generator

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



# %% [markdown]
# Usually we would fix the random seeds here, but the agents are being run with fixed random seeds, so this is not necessary here.

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
                                           padding=(1, 0),
                                           num_blocks=lambda: random.randint(8, 16), #  flat random interval of tower sizes (inclusive)
                                           )


# %%
NUM_TOWERS  = 16 #64
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
visualize_towers(towers)

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
# costs = []
# statusses = []
# for world in tqdm(worlds):
#     cost,status = get_tower_cost(lower_agent,world)
#     costs.append(cost)
#     statusses.append(status)

# %% [markdown]
# Split the basic costs into three percentiles: easy, medium, hard.

# %%
# difficulty_percentiles = [np.percentile(costs, i)
#                for i in [33, 66, 99]]

# percentiles = [None] * len(costs)
# for i, cost in enumerate(costs):
#     if cost < difficulty_percentiles[0]:
#         percentiles[i] = 'easy'
#     elif cost < difficulty_percentiles[1]:
#         percentiles[i] = 'medium'
#     else:
#         percentiles[i] = 'hard'

# %% [markdown]
# ## Find best and worst subgoals
# We compute the full subgoal tree for each tower and extract the best and worst sequence.
# 
# Note: for the planned studies, we will use individual states and subgoals, not sequences of subgoals.

# %%
decomposer = Rectangular_Keyholes(
    sequence_length=4,
    necessary_conditions=[
        Area_larger_than(area=1),
        Area_smaller_than(area=18), # used to be 21
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

# %%
# # parallelized
def get_subgoal_tree_from_tower(agent, world):
    world.reset()
    agent.set_world(world)
    return agent.get_subgoal_tree(only_solved_sequences=True)

agents = [copy.deepcopy(a) for a in [sg_agent]*len(worlds)]

# remove process connection before handoff to the threads
for world in worlds:
    world.physics_provider.kill_server(force=True)

trees = p_tqdm.p_map(get_subgoal_tree_from_tower, agents, worlds)

# %% [markdown]
# ### Select most divergent subgoals
# For each tower, select n state/subgoal combinations that are maximally divergent.

# %%
NUM_SGs_PER_TOWER = 6 # how many subgoals do we want to choose?

# %%
best_worst_subgoals = []
for tree in trees:
    best_worst_subgoals.append(tree.get_most_divergent_matching_pairs_of_subgoals(NUM_SGs_PER_TOWER))

# %%
results = [{'world':world,'world name':index, 'subgoal tree':tree,'cost':cost,'percentile':percentile, 'best_worst_subgoals':best_worst_subgoal} for world,index,tree,cost,percentile,best_worst_subgoal in zip(worlds,range(len(towers)), trees,costs,percentiles,best_worst_subgoals)]

# %% [markdown]
# Add into df

# %%
df = pd.DataFrame(columns=['world name','world','best subgoal', 'worst subgoal', 'subgoal cost delta', 'tower percentile', 'state', 'blocks'])
i = 0

for r in results:
    for best_subgoal,worst_subgoal in r['best_worst_subgoals']:
        # fill in subgoal cost delta
        delta = worst_subgoal.subgoal.solution_cost - best_subgoal.subgoal.solution_cost
        # fill in state from the best subgoal
        state = best_subgoal.subgoal.past_world.current_state
        # insert into dataframe
        df.loc[i] = [r['world name'],r['world'],best_subgoal,worst_subgoal,delta,r['percentile'],state,state.blocks]
        i += 1

# %% [markdown]
# What is the distribution over the subgoal deltas?

# %%
df['subgoal cost delta'].plot()

# %% [markdown]
# Let's display some of the subgoals 

# %%
for i,row in df.sort_values('subgoal cost delta',ascending=False).head(5).iterrows():
    print(i,row['subgoal cost delta'])
    # construct dummy sequence
    sequence = Subgoal_sequence([row['best subgoal'], row['worst subgoal']])
    sequence.visual_display()

# %% [markdown]
# Let's save the dataframe to disk. This will serve as the basis for the `given_subgoal` human experiment.

# %%
df.to_pickle("most divergent subgoals.pkl")


