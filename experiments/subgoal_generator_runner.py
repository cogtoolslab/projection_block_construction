# set up directories
import tqdm
import multiprocessing
import random
import time
import psutil
import traceback
import datetime
import copy
import numpy as np
import pandas as pd
from analysis.utils.analysis_helper import preprocess_df
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
results_dir = os.path.join(proj_dir, 'results')
df_dir = os.path.join(results_dir, 'dataframes')


"""
Generates all sequences of subgoals and saves them. Requires subgoal planner.
"""

RAM_LIMIT = 100  # percentage of RAM usage over which a process doesn't run as to not run out of memory
SAVE_INTERMEDIATE_RESULTS = True  # save the results of each experiment to a file


def run_experiment(worlds, agents, per_exp=1, steps=1, verbose=False, save=True, parallelized=True, maxtasksperprocess=1):
    """Runs x experiments on the given worlds with the given agents for up to 100 steps while keeping logging values to a dataframe. Pass blockworlds as named dictionary for readability of results. Pass agents as a list: the __str__ function of an agent will take care of it. The world is assigned to the agent later, so it makes sense to pass none. You can pass negative numbers steps to run until the agent is finished. Pass a float to parallelized to set the fraction of CPUs to use. Note that the system that reads in the dataframe needs identical or compatible versions of Python and it's modules for it to be able to read the dataframe back in again. Hint: `pip freeze > requirements.txt`"""
    # we want human readable labels for the dataframe
    if type(worlds) is not dict:
        # if worlds is list create dictionary
        worlds = {w.__str__()+f"_{i}": w for i,w in enumerate(worlds)}
    if type(agents) is dict:
        # if agents is dict flatten it and rely on informative agent __str__
        agents = [a for a in agents.values()]
    # we need to copy the world and agent to reset them
    # create a list of experiments to run
    experiments = [(copy.deepcopy(w), copy.deepcopy(a), steps, verbose, i)
                   for i in range(per_exp) for a in agents for w in worlds.items()]
    # lets run the experiments
    if parallelized is not False:
        # restart process after a single task is performed—slow for short runs, but fixes memory leak (hopefully)
        P = multiprocessing.Pool(int(multiprocessing.cpu_count(
        )*parallelized), maxtasksperchild=maxtasksperprocess)
        results_mapped = tqdm.tqdm(P.imap_unordered(
            _run_single_experiment, experiments), total=len(experiments))
        P.close()
    else:
        results_mapped = list(map(_run_single_experiment, experiments))

    results = pd.concat(results_mapped).reset_index(drop=True)

    # preprocess_df(results) #automatically fill in code relevant to analysis

    if save is not False:
        # check if results directory exists
        if not os.path.isdir(df_dir):
            os.makedirs(df_dir)
        # save the results to a file.
        if type(save) is str:
            results.to_pickle(os.path.join(df_dir, save+".pkl"))
            print("Saved to", os.path.join(df_dir, save+".pkl"))
        else:
            results.to_pickle(os.path.join(
                df_dir, "Experiment "+str(datetime.datetime.today())+".pkl"))
            print("Saved to", df_dir, "Experiment " +
                  str(datetime.datetime.today())+".pkl")

    return results


def _run_single_experiment(experiment):
    """Runs a single experiment. Returns complete dataframe with an entry for each action."""
    world_dict, agent, steps, verbose, run_nr = experiment
    world_label = world_dict[0]
    world = world_dict[1]
    # if the agent has no random seed assigned yet, assign one now only for this run
    try:
        # if agent.random_seed is None:
        agent.random_seed = random.randint(0, 99999)  # overwrite in any case
    except AttributeError:
        pass
    run_ID = world_label+' | '+agent.__str__()+str(run_nr)+' | ' + \
        str(random.randint(0, 9999))  # unique string representing the run
    while psutil.virtual_memory().percent > RAM_LIMIT:
        print("Delaying running", agent.__str__(), '******', world_label,
              "because of RAM usage. Trying again in 1000 seconds. RAM usage is "+str(psutil.virtual_memory().percent)+'%')
        time.sleep(1000)

    # print('Running', agent.__str__(), '******', world.__str__())
    agent_parameters = agent.get_parameters()
    agent_parameters_w_o_random_seed = {
        key: value for key, value in agent_parameters.items() if key != 'random_seed'}
    agent_parameters_w_o_random_seed = agent_para_dict(
        agent_parameters_w_o_random_seed)
    agent.set_world(world)
    # create dataframe
    # columns=DF_COLS+list(agent_parameters.keys()), index=[1])
    r = pd.DataFrame()

    # run subgoal planning
    start_time = time.perf_counter()
    chosen_seq, all_sequences, solved_sequences = agent.plan_subgoals(
        verbose=verbose)

    r['run_ID'] = [run_ID]
    r['agent'] = [agent_parameters['agent_type']]
    r['agent_attributes'] = [str(agent_parameters_w_o_random_seed)]
    r['world'] = [world_label]
    r['_world'] = [world]
    r['_all_sequences'] = [all_sequences]
    r['n_all_sequences'] = [len(all_sequences)]
    r['_solved_sequences'] = [solved_sequences]
    r['n_solved_sequences'] = [len(solved_sequences)]
    r['_chosen_subgoal_sequence'] = [chosen_seq]
    r['all_sequences_planning_cost'] = [
        sum([s.planning_cost() for s in all_sequences])]
    r['_agent'] = [agent]
    r['_world'] = [world]
    for key, value in agent_parameters.items():
        r[key] = [value]
    r['execution_time'] = [time.perf_counter() - start_time]

    # print("Done with", agent.__str__(), '******', world_label, "in", str(round(time.perf_counter() - start_time)),
        #   "seconds. Found", str(len(solved_sequences)), "solutions to", str(len(all_sequences)), "sequences.")

    if SAVE_INTERMEDIATE_RESULTS:
        # get folder for experiment
        exp_dir = os.path.join(df_dir, "Experiment "+str(datetime.datetime.today()))
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        # save the results to a file.
        r.to_pickle(os.path.join(exp_dir, run_ID+".pkl"))
    return r


class agent_para_dict(dict):
    """A class for hashable dicts for agent parameters. Derives from a regular dictionary. 
    ENTRIES MUST NOT BE CHANGED AFTER CREATION"""

    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()
