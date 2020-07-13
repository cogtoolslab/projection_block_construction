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

import psutil
import time
start_time = time.time()

# a = BFS_Agent(horizon=5,scoring='Sum',
#     sparse=False,
#     scoring_function=bw.silhouette_hole_score
#     )
a = MCTS_Agent(horizon=10**6)
w = bw.Blockworld(silhouette=bl.load_interesting_structure(15),
    block_library=bl.bl_silhouette2_default,
    fast_failure=False
    )
# no fast failure to set upper bound of time

a.set_world(w)
while w.status()[0] == 'Ongoing':
    a.act(1) #only acting one step! Expensive by factor planning depth
    print(psutil.cpu_percent(),'%|',psutil.virtual_memory()) #print current RAM usage
    print(w.status())
# a.act(-1,verbose=True)
print("Done in %s seconds" % (time.time() - start_time))
print(psutil.cpu_percent(),'%|',psutil.virtual_memory())
