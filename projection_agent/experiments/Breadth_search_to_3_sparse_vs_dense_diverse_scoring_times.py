#add the parent path to import the modules
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from agent import Agent
from MCTS_Agent import MCTS_Agent
import blockworld as bw
import random
import blockworld_library as bl
import experiment_runner

import time
start_time = time.time()

#16 for nightingale
fraction_of_cpus = 12/16

agents = [
    Agent(horizon=1,scoring_function=bw.random_scoring,scoring='Final state'),
    Agent(horizon=1,scoring_function=bw.F1score,scoring='Final state'),
    Agent(horizon=2,scoring_function=bw.F1score,scoring='Final state'),
    Agent(horizon=3,scoring_function=bw.F1score,scoring='Final state'),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score,scoring='Final state'),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score,scoring='Final state'),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score,scoring='Final state'),
    Agent(horizon=1,scoring_function=bw.F1score,sparse=True,scoring='Final state'),
    Agent(horizon=2,scoring_function=bw.F1score,sparse=True,scoring='Final state'),
    Agent(horizon=3,scoring_function=bw.F1score,sparse=True,scoring='Final state'),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Final state'),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Final state'),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Final state'),
    Agent(horizon=1,scoring_function=bw.random_scoring,scoring='Average'),
    Agent(horizon=1,scoring_function=bw.F1score,scoring='Average'),
    Agent(horizon=2,scoring_function=bw.F1score,scoring='Average'),
    Agent(horizon=3,scoring_function=bw.F1score,scoring='Average'),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score,scoring='Average'),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score,scoring='Average'),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score,scoring='Average'),
    Agent(horizon=1,scoring_function=bw.F1score,sparse=True,scoring='Average'),
    Agent(horizon=2,scoring_function=bw.F1score,sparse=True,scoring='Average'),
    Agent(horizon=3,scoring_function=bw.F1score,sparse=True,scoring='Average'),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Average'),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Average'),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Average'),
    Agent(horizon=1,scoring_function=bw.random_scoring,scoring='Sum'),
    Agent(horizon=1,scoring_function=bw.F1score,scoring='Sum'),
    Agent(horizon=2,scoring_function=bw.F1score,scoring='Sum'),
    Agent(horizon=3,scoring_function=bw.F1score,scoring='Sum'),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    Agent(horizon=1,scoring_function=bw.F1score,sparse=True,scoring='Sum'),
    Agent(horizon=2,scoring_function=bw.F1score,sparse=True,scoring='Sum'),
    Agent(horizon=3,scoring_function=bw.F1score,sparse=True,scoring='Sum'),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Sum'),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Sum'),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score,sparse=True,scoring='Sum'),
    ]

silhouettes = {i : bl.load_interesting_structure(i) for i in [14,15,5,8,12,1]}
worlds_silhouettes = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for i,s in silhouettes.items()}
worlds_small = {
    'stonehenge_6_4' : bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
    'stonehenge_3_3' : bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
    'block' : bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
    'T' : bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
    'side_by_side' : bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
}
worlds = {**worlds_silhouettes,**worlds_small}
results = experiment_runner.run_experiment(worlds,agents,10,60,verbose=False,parallelized=fraction_of_cpus,save='breadth_to_3')
print(results[['agent','world','outcome']])

print("Done in %s seconds" % (time.time() - start_time))