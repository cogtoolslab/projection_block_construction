from agent import Agent
from MCTS_Agent import MCTS_Agent
import blockworld as bw
import random
import blockworld_library as bl
import experiment_runner

import time
start_time = time.time()

a = MCTS_Agent(horizon=2500)
w = bw.Blockworld(silhouette=bl.load_interesting_structure(15),block_library=bl.bl_silhouette2_default)

a.set_world(w)
# while w.status() == 'Ongoing':
    # a.act(-1,verbose=True)
a.act(-1,verbose=True)
print(w.status())
print("Done in %s seconds" % (time.time() - start_time))

# agents = [
#     Agent(horizon=1,scoring_function=bw.random_scoring),
#     Agent(horizon=1,scoring_function=bw.F1score),
#     Agent(horizon=3,scoring_function=bw.F1score),
#     # Agent(horizon=5,scoring_functioffsozn=bw.F1score),
#     Agent(horizon=1,scoring_function=bw.silhouette_hole_score),
#     # Agent(horizon=2,scoring_function=bw.silhouette_hole_score),
#     Agent(horizon=3,scoring_function=bw.silhouette_hole_score),
#     # Agent(horizon=4,scoring_function=bw.sizesizessilhouette_hole_score),
#     Agent(horizon=5,scoring_function=bw.silhouette_hole_score),
#     MCTS_Agent(horizon=1000),
#     # MCTS_Agent(horizon=2500),
#     # MCTS_Agent(horizon=10000),
#     ]
# silhouettes = [bl.load_interesting_structure(i) for i in [14,15,5,8,12,1]]

# worlds_silhouettes = [bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for s in silhouettes]
# worlds_small = [
#     bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
#     bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
#     bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
#     bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
#     bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
# ]
# worlds = worlds_silhouettes+worlds_small
# results = experiment_runner.run_experiment(worlds,agents,10,60,verbose=False)
# print(results[['agent','world','outcome']])
