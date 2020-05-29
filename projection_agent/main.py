from agent import Agent
from MCTS_Agent import MCTS_Agent
import blockworld as bw
import random
import blockworld_library as bl
import experiment_runner

import time
start_time = time.time()

# a = MCTS_Agent(horizon=1000)
# w = bw.Blockworld(silhouette=bl.load_interesting_structure(15),block_library=bl.bl_silhouette2_default)
# a.set_world(w)
# while w.status() == 'Ongoing':
#     a.act(-1,verbose=True)

agents = [Agent(horizon=1,scoring_function=bw.F1score),Agent(horizon=1,scoring_function=bw.silhouette_hole_score),Agent(horizon=3,scoring_function=bw.silhouette_hole_score),Agent(horizon=5,scoring_function=bw.silhouette_hole_score),MCTS_Agent(horizon=1000)]
silhuouettes = [bl.load_interesting_structure(i) for i in range(16)]
worlds = [bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for s in silhuouettes]
results = experiment_runner.run_experiment(worlds,agents,1,60,verbose=False)
print(results[['world','outcome']])

print("Done in %s seconds" % (time.time() - start_time))