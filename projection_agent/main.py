from agent import Agent
import blockworld
import random
import blockworld_library as bl

w = blockworld.Blockworld(silhouette=bl.stonehenge_6_4,dimension=(6,4),block_library = bl.stonehenge_6_4_blocklibrary)
a = Agent(w,sparse=False)
#greedy agent
while w.status() == 'Ongoing':
    a.act(1,verbose=True)

#brute force search agent
w = blockworld.Blockworld(silhouette=bl.stonehenge_6_4,dimension=(6,4),block_library = bl.stonehenge_6_4_blocklibrary)
a = Agent(w,sparse=False)
a.act(6,verbose=True)