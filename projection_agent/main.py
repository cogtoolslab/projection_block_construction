from agent import Agent
import blockworld
import random
import blockworld_library as bl
import experiment_runner


silhouette = bl.load_interesting_structure(11)
# silhouette = bl.stonehenge_6_4

w1 = blockworld.Blockworld(silhouette=bl.stonehenge_6_4,block_library = bl.stonehenge_6_4_blocklibrary)
w2 = blockworld.Blockworld(silhouette=bl.load_interesting_structure(12),block_library = bl.stonehenge_6_4_blocklibrary)
a = Agent()
a.set_world(w1)
#greedy agent
# while w1.status() == 'Ongoing':
    # a.act(2,2,verbose=True)

results = experiment_runner.run_experiment([w1],[a],100,6,verbose=False)

print(results.shape)
# #brute force search agent
# w = blockworld.Blockworld(silhouette=bl.stonehenge_6_4,dimension=(6,4),block_library = bl.stonehenge_6_4_blocklibrary)
# a = Agent(w,sparse=False)
# a.act(6,verbose=True)