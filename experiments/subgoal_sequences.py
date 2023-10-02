import os
import sys

proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from scoping_simulations.model.Resource_Rational_Subgoal_Planning_Agent import *
import scoping_simulations.utils.blockworld as bw
import scoping_simulations.utils.blockworld_library as bl

import pandas as pd


a = Resource_Rational_Subgoal_Planning_Agent(
                lower_agent=BFS_Lookahead_Agent(horizon=2,only_improving_actions=True),
                lookahead = 8,
                include_subsequences=True,
                c_weight = 1,
                S_treshold=0.1,
                S_iterations=1)

w = bw.Blockworld(silhouette=bl.load_interesting_structure(13),block_library=bl.bl_silhouette2_default,fast_failure=False,legal_action_space=True) 

a.set_world(w)

names = []
chosen_seqs = []

for c_weight in np.arange(0,1,0.001):
    a.c_weight = c_weight
    chosen_seq = a.plan_subgoals()
    chosen_seq = chosen_seq[0]
    name =  [sg['name'] for sg in chosen_seq]
    print("For",c_weight,"got subgoal\t"+str(name))
    names.append(name)
    chosen_seqs.append(chosen_seq)

df = pd.DataFrame()