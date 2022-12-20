# %% [markdown]
# # Cost of future subgoals model preference elicitation
# 
# This notebook contains the code used to generate the subgoal pairs and data for analysis for the third human study.
# 
# In this study, we want to see if people are sensitive to the computational costs of future subgoals. 
# 
# For each tower, we
# * generate a tree of subgoal decompositions
# * get the preferences over hte first subgoals across planners directly from the tree
# 
# Tower generation code is taken from `Future_costs_stim_generation.ipynb`

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

import datetime

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
# used for naming the output file
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
                                           seed=3,
                                           padding=(1, 0),
                                           num_blocks=lambda: random.randint(6, 18), #  flat random interval of tower sizes (inclusive)
                                           )

# %%
NUM_TOWERS  = 1
towers = []
for i in tqdm(range(NUM_TOWERS)):
    tower = generator.generate()
    towers.append(tower)

# %%
worlds = [Blockworld(silhouette=t['bitmap'], block_library=bl.bl_nonoverlapping_simple) for t in towers]
worlds_sizes = [len(t['blocks']) for t in towers]

# %% [markdown]
# ## Generate subgoal decompositon tree

# %%
w = worlds[0]

# %%
sga = Subgoal_Planning_Agent(lower_agent=Best_First_Search_Agent())

# %%
sga.set_world(w)

# %%
sg_tree = sga.get_subgoal_tree()


