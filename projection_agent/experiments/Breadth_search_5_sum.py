#add the parent path to import the modules
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from BFS_Agent import BFS_Agent
from MCTS_Agent import MCTS_Agent
import blockworld as bw
import random
import blockworld_library as bl
import experiment_runner

import time
start_time = time.time()

#16 for nightingale, 96 for google cloud
fraction_of_cpus = 6/8 #2/16

agents = [
    # BFS_Agent(horizon=1,scoring_function=bw.random_scoring,scoring='Sum'),
    # BFS_Agent(horizon=1,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    # BFS_Agent(horizon=1,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    # BFS_Agent(horizon=2,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    # BFS_Agent(horizon=3,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    # BFS_Agent(horizon=4,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    BFS_Agent(horizon=5,scoring_function=bw.silhouette_hole_score,scoring='Sum'),
    ]

silhouettes = {i : bl.load_interesting_structure(i) for i in [14,15,5,8,12,1]}
worlds_silhouettes = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default,fast_failure=True) for i,s in silhouettes.items()}
worlds_small = {
    'stonehenge_6_4' : bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4,fast_failure=True),
    'stonehenge_3_3' : bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3,fast_failure=True),
    'block' : bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3,fast_failure=True),
    'T' : bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4,fast_failure=True),
    'side_by_side' : bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_side_by_side),
}
worlds = {**worlds_silhouettes,**worlds_small}

results = experiment_runner.run_experiment(worlds,agents,1,20,verbose=False,parallelized=fraction_of_cpus,save='breadth_to_4_sum')
print(results[['agent','world','outcome']])

print("Done in %s seconds" % (time.time() - start_time))