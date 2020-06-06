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

import random
import time
start_time = time.time()


agents = [MCTS_Agent(horizon=10**i) for i [2,3,4,5,6,7]]

#16 for nightingale
fraction_of_cpus = len(agents)/16

silhouettes = [bl.load_interesting_structure(i) for i in [14,15,5,8,12,1]]
worlds_silhouettes = [bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for s in silhouettes]
worlds_small = [
    bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
    bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
    bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
    bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
    bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
]
worlds = worlds_silhouettes+worlds_small
results = experiment_runner.run_experiment(worlds,agents,10,60,verbose=False,parallelized=fraction_of_cpus,save='MCTS')
print(results[['agent','world','outcome']])

print("Done in %s seconds" % (time.time() - start_time))