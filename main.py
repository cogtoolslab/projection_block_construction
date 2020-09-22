from BFS_Agent import BFS_Agent
from MCTS_Agent import MCTS_Agent
from Beam_Search_Agent import Beam_Search_Agent
from Astar_Agent import Astar_Agent
from Naive_Q_Agent import Naive_Q_Agent
from Construction_Paper_Agent import Construction_Paper_Agent
import blockworld as bw
import random
import blockworld_library as bl
import experiment_runner

import time
start_time = time.time()
#to properly benchmark python -m cProfile -o main.prof main.py && snakeviz main.prof 

# a = MCTS_Agent(horizon=2500)
# a = BFS_Agent(horizon=1,scoring='Fixed')
# a = Beam_Search_Agent(beam_width=100,heuristic=bw.F1_stability_score)
# a = Astar_Agent()
# a = Naive_Q_Agent(heuristic=lambda x: bw.F1_stability_score(x)+bw.sparse(x)*1000,max_episodes=10000)
# w = bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_side_by_side)
a = Construction_Paper_Agent(lower_agent=BFS_Agent(horizon=2))
# w = bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4)
# w = bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3)
w = bw.Blockworld(silhouette=bl.load_interesting_structure(14),block_library=bl.bl_silhouette2_default,fast_failure=False)
# w = bw.Blockworld(silhouette=bl.load_interesting_structure(15),block_library=bl.bl_silhouette2_default,fast_failure=False) # 🐘

a.set_world(w)
while w.status()[0] == 'Ongoing':
    a.act(verbose=True)
    print(w.status())
    w.current_state.visual_display(True,w.silhouette)
print('Done,',w.status())

print("Done in %s seconds" % (time.time() - start_time))

