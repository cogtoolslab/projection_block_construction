"""
This is to analyze high-level properties of a tower (or certain decisions that have been made re towers)

Meant to be used with the random agent.

* Read branching factor from world after each random step
* random rollouts to get to winning ratio
* mean length
* timecourse plot
"""

# set up imports
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(proj_dir)
utils_dir = os.path.join(proj_dir, 'utils')
sys.path.append(utils_dir)
analysis_dir = os.path.join(proj_dir, 'analysis')
analysis_utils_dir = os.path.join(analysis_dir, 'utils')
sys.path.append(analysis_utils_dir)
agent_dir = os.path.join(proj_dir, 'model')
sys.path.append(agent_dir)
agent_util_dir = os.path.join(agent_dir, 'utils')
sys.path.append(agent_util_dir)
experiments_dir = os.path.join(proj_dir, 'experiments')
sys.path.append(experiments_dir)
df_dir = os.path.join(proj_dir, 'results/dataframes')
stim_dir = os.path.join(proj_dir, 'stimuli')

from tqdm import tqdm
import p_tqdm
import numpy as np
import pandas as pd
import pickle

print(proj_dir)

# we need those imports for main
import utils.blockworld_library as bl
import utils.blockworld as bw
from utils.blockworld import *
from model.Random_Agent import Random_Agent # might have to add other agent classes here if theyr're to be used in the analysis


def analyze_single_tower(world, agent, n=1000, verbose=True):
    """Takes a single tower and an agent and takes a single steps.

    `n` is number of runs.

    This function fixes the random seed of the agent.

    It returns:
    * ratio of positive to negative outcomes (tracks how many paths lead to a goal state)
    * avg branching factor
    * a dictionary with {depth: branching factor}
    * a list with [(outcome, depth)]
    """
    outcomes = []
    branching_factors = {}
    if verbose:
        iterator = tqdm(range(n))
    else:
        iterator = range(n)
    for i in iterator:
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
    try:
        all_branching_factors = list(
            np.concatenate(list(branching_factors.values())))
    except ValueError:
        # Somethings gone wrong and we haven't found a single step
        all_branching_factors = []
    total_branching_factor = np.mean(all_branching_factors)
    # get the ratio of positive to negative outcomes
    outcome_ratio = np.mean([outcome[0] == 'Winning' for outcome in outcomes])
    return outcome_ratio, total_branching_factor, branching_factors, outcomes


def run_parallelized_analysis_across_towers(worlds, agent, n=1000):
    """Parallelizes the analysis of multiple worlds.

    Worlds is a dictionary with {name: world}"""
    outcomes = p_tqdm.p_map(lambda w: analyze_single_tower(
        worlds[w], agent, n=n, verbose=False), worlds)
    return {world_name: outcome for world_name, outcome in zip(worlds.keys(), outcomes)}


def run_analysis(sets, agent, n=1000):
    """Runs analysis on a set of worlds (dictionary with {name: {world_name: world}})

    Returns pandas.dataframe with results.
    """

    df = pd.DataFrame(columns=['set', 'tower', 'outcome_ratio',
                               'avg_branching_factor', 'branching_factors', 'outcomes'])

    # run the analysis
    for i, set_ in enumerate(sets):
        worlds = sets[set_]
        print("Analyzing set {} ({} of {}) with {} worlds".format(
            set_, i, len(sets), len(worlds)))
        results = run_parallelized_analysis_across_towers(worlds, agent, n=n)
        # time to parse the results
        for tower in results.keys():
            outcome_ratio, avg_branching_factor, branching_factors, outcomes = results[tower]
            df = df.append({'set': set_, 'tower': tower, 'outcome_ratio': outcome_ratio,
                            'avg_branching_factor': avg_branching_factor,
                            'branching_factors': branching_factors, 'outcomes': outcomes}, ignore_index=True)
    return df


def run_and_save_analysis(set, agent, n=100, filename=None):
    """Runs and saves the analysis. This is meant to save jupyter notebooks from timing out."""
    df = run_analysis(set, agent, n=n)
    if filename is None:
        filename = "_".join(set.keys())+"_"+agent.__class__.__name__ + \
                            "_stochastic_tower_analysis.csv"
    df.to_csv(os.path.join(df_dir, filename))
    print("Saved to {}".format(os.path.join(df_dir, filename)))

def prep_for_offline_running(set, agent, n=100, filename = None):
    """Pickles inputs from jupyter and returns filename of pickle so it can be run offline."""
    # pickle set and agent
    if filename is None:
        filename = "_".join(set.keys())+"_"+agent.__class__.__name__ + \
                            "_stochastic_tower_analysis.pkl"
    pickle.dump((set, agent, n), open(os.path.join(df_dir, "input_"+filename), 'wb'))
    print("Saved to {}".format(os.path.join(df_dir, filename)))
    return os.path.join(df_dir, filename)

def run_offline(filename, input_filename=None):
    """Runs analysis on a set of worlds (dictionary with {name: {world_name: world}})

    Pickles output.
    """"Simply pass the filename of the pickle file. When using `prep_for_offline_running`, you can omit the leading `input_`. Do not pass a path, only the filename for the file in the results/dataframes directory."""
    if input_filename is None:
        input_filename = "input_"+filename
    set, agent, n = pickle.load(open(os.path.join(df_dir, input_filename), 'rb'))
    print("Loaded {} and got {} sets".format(os.path.join(df_dir, input_filename), len(set)))
    df = run_and_save_analysis(set, agent, n=n, filename=filename)
    return df

# if we call the file, run offline
if __name__ == '__main__':
    # get argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str,
                        help='filename of pickle file with the input to run_offline') 
    run_offline(parser.parse_args().filename)