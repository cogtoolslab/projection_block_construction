import random
import time

import numpy as np

import scoping_simulations.utils.matter_server as Matter_Server
from scoping_simulations.utils.blockworld import Blockworld

N = 100

print("Benchmarking Box2D")
w = Blockworld(silhouette=np.ones((8, 8)), physics_provider="box2d")
start_time = time.time()
num_actions = 0
times = []
for i in range(N):
    for j in range(5):
        # take random action
        actions = w.current_state.possible_actions()
        try:
            action = random.choice(actions)
            num_actions += 1
        except:
            break
        w.apply_action(action)
        start_time = time.time()
        stable = w.stability()
        end_time = time.time()
        # print("stability {},{} in {} milliseconds: {}".format(
        # i, j, (end_time - start_time)*1000,stable))
        times.append(end_time - start_time)
    w.reset()
# print("Box2D: %s seconds" % (time.time() - start_time)," #actions:",num_actions)
print("mean time for Box2D physics server:", np.mean(times) * 1000, "milliseconds")

# or to create a server
physics_provider = Matter_Server.Physics_Server()

print("Physics provider created")

print("Benchmarking matter")
w = Blockworld(silhouette=np.ones((8, 8)), physics_provider=physics_provider)
start_time = time.time()
num_actions = 0
times = []
for i in range(N):
    # print(i)
    for j in range(5):
        # take random action
        actions = w.current_state.possible_actions()
        try:
            action = random.choice(actions)
            num_actions += 1
        except:
            break
        w.apply_action(action)
        start_time = time.time()
        stable = w.stability()
        end_time = time.time()
        # print("stability {},{} in {} milliseconds: {}".format(
        # i, j, (end_time - start_time)*1000,stable))
        times.append(end_time - start_time)
    w.reset()
# print("matter: %s seconds" % (time.time() - start_time)," #actions:",num_actions)
print("mean time for matter physics server:", np.mean(times) * 1000, "milliseconds")
w.__del__()
print("Done")
