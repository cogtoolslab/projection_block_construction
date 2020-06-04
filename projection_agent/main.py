from agent import Agent
from MCTS_Agent import MCTS_Agent
import blockworld as bw
import random
import blockworld_library as bl
import experiment_runner

import time
start_time = time.time()

# a = MCTS_Agent()
# a = MCTS_Agent(horizon=2500)
# w = bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4)
# # w = bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4)
# a.set_world(w)
# # while w.status() == 'Ongoing':
#     # a.act(-1,verbose=True)
# a.act(-1,verbose=True)

agents = [
    Agent(horizon=1,scoring_function=bw.random_scoring),
    Agent(horizon=1,scoring_function=bw.F1score),
    Agent(horizon=3,scoring_function=bw.F1score),
    Agent(horizon=5,scoring_function=bw.F1score),
    Agent(horizon=1,scoring_function=bw.silhouette_hole_score),
    Agent(horizon=2,scoring_function=bw.silhouette_hole_score),
    Agent(horizon=3,scoring_function=bw.silhouette_hole_score),
    Agent(horizon=4,scoring_function=bw.silhouette_hole_score),
    Agent(horizon=5,scoring_function=bw.silhouette_hole_score),
    MCTS_Agent(horizon=1000),
    MCTS_Agent(horizon=5000),
    # MCTS_Agent(horizon=10000),
    ]
silhuouettes = [bl.load_interesting_structure(i) for i in range(16)]
worlds_silhuoettes = [bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for s in silhuouettes]
worlds_small = [
    bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
    bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
    bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
    bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
    bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
]
worlds = worlds_silhuoettes+worlds_small
results = experiment_runner.run_experiment(worlds,agents,20,60,verbose=False)
print(results[['agent','world','outcome']])

print("Done in %s seconds" % (time.time() - start_time))