"""
This is to analyze high-level properties of a tower (or certain decisions that have been made re towers)

Meant to be used with the random agent

* Read branching factor from world after each random step
* random rollouts to get to winning ratio
* mean length
* timecourse plot
"""

from tqdm import tqdm
import numpy as np

def analyze_single_tower(world, agent, n = 1000):
    """Takes a single tower and an agent and takes a single steps.
    
    `n` is number of runs. 

    It returns:
    * ratio of positive to negative outcomes (tracks how many paths lead to a goal state)
    * avg branching factor
    * a dictionary with {depth: branching factor}
    * a list with [(outcome, depth)]
    """
    outcomes = []
    branching_factors = {}
    for i in tqdm(range(n)):
        _world = world.copy()
        agent.set_world(_world)
        agent.random_seed = i
        depth = 0
        while _world.status()[0] == 'Ongoing':
            actions, info = agent.act(steps=1)
            branching_factor = len(_world.current_state.legal_actions())
            try:
                branching_factors[depth].append(branching_factor)
            except KeyError:
                branching_factors[depth] = [branching_factor]
            depth += 1
        # we're done
        outcomes.append((_world.status(), depth))
    # done w all
    # get the branching factor
    all_branching_factors = list(
        np.concatenate(list(branching_factors.values())))
    total_branching_factor = np.mean(all_branching_factors)
    # get the ratio of positive to negative outcomes
    outcome_ratio = np.mean([outcome[0] == 'Winning' for outcome in outcomes])
    return outcome_ratio, total_branching_factor, branching_factors, outcomes


