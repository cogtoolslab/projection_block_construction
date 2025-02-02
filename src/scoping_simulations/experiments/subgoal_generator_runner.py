import copy
import datetime
import os
import random
import time
from functools import partial
from multiprocessing import Pool

import pandas as pd
import psutil
import tqdm

from scoping_simulations.utils.directories import PROJ_DIR

RESULTS_DIR = os.path.join(PROJ_DIR, "results")
DF_DIR = os.path.join(RESULTS_DIR, "dataframes")


"""
Generates all sequences of subgoals and saves them. Requires subgoal planner.
"""

RAM_LIMIT = 90  # percentage of RAM usage over which a process doesn't run as to not run out of memory
SAVE_INTERMEDIATE_RESULTS = True  # save the results of each experiment to a file


def run_experiment(
    worlds,
    agents,
    per_exp=1,
    steps=1,
    verbose=False,
    save=True,
    parallelized=True,
    maxtasksperprocess=1,
    collate_results=False,
):
    """Runs x experiments on the given worlds with the given agents for up to 100 steps while keeping logging values to a dataframe. Pass blockworlds as named dictionary for readability of results. Pass agents as a list: the __str__ function of an agent will take care of it. The world is assigned to the agent later, so it makes sense to pass none. You can pass negative numbers steps to run until the agent is finished. Pass a float to parallelized to set the fraction of CPUs to use. Note that the system that reads in the dataframe needs identical or compatible versions of Python and it's modules for it to be able to read the dataframe back in again.

    For performance reasons, you might not want to store and collate the dataframe of the results. In that case, set collate_results to False.
    """
    # some helpful printouts for where the data will be saved
    if save is not False and type(save) is not str:
        # save under the current date if no filename given
        save = datetime.datetime.now().strftime("%d-%m-%Y")
        print("No name for the output given. Saving to", save)
    if save is not False:
        save = "subgoal_generator_" + save
        if SAVE_INTERMEDIATE_RESULTS or not collate_results:
            # if necessary, make folder
            if not os.path.exists(os.path.join(DF_DIR, save)):
                os.makedirs(os.path.join(DF_DIR, save))
            # does the folder have dataframes in it already?
            if len(os.listdir(os.path.join(DF_DIR, save))) > 0:
                # increment the name until we find a folder or .pkl that doesn't exist
                i = 1
                while os.path.exists(
                    os.path.join(DF_DIR, save + f"_{i}")
                ) or os.path.exists(os.path.join(DF_DIR, save + f"_{i}.pkl")):
                    i += 1
                save = save + f"_{i}"
            print(
                "For each run, a dataframe will be saved to the folder",
                os.path.join(DF_DIR, save),
            )
        if collate_results:
            print(
                "A single complete dataframe will be saved to",
                os.path.join(DF_DIR, save) + ".pkl",
            )

    # we want human readable labels for the dataframe
    if type(worlds) is not dict:
        # if worlds is list create dictionary
        worlds = {w.__str__() + f"_{i}": w for i, w in enumerate(worlds)}
    if type(agents) is dict:
        # if agents is dict flatten it and rely on informative agent __str__
        agents = [a for a in agents.values()]
    # we need to copy the world and agent to reset them
    # create a list of experiments to run
    experiments = [
        (copy.deepcopy(w), copy.deepcopy(a), steps, verbose, i, save, collate_results)
        for i in range(per_exp)
        for a in agents
        for w in worlds.items()
    ]
    # lets run the experiments
    if parallelized:
        # Set up the process pool and limit the number of tasks per process to avoid memory leaks
        num_cpus = os.cpu_count()
        num_processes = int(num_cpus * parallelized)
        with Pool(processes=num_processes, maxtasksperchild=maxtasksperprocess) as pool:
            run_single_experiment_partial = partial(_run_single_experiment)
            results_mapped = list(
                tqdm.tqdm(
                    pool.imap_unordered(run_single_experiment_partial, experiments),
                    total=len(experiments),
                    desc="Running subgoal agents",
                    dynamic_ncols=True,
                )
            )
    else:
        results_mapped = list(map(_run_single_experiment, tqdm.tqdm(experiments)))

    # preprocess_df(results) #automatically fill in code relevant to analysis

    if save is not False:
        results = pd.concat(results_mapped).reset_index(drop=True)
        # check if results directory exists
        if not os.path.isdir(DF_DIR):
            os.makedirs(DF_DIR)
        # save the results to a file.
        results.to_pickle(os.path.join(DF_DIR, save + ".pkl"))
        print("Saved to", os.path.join(DF_DIR, save + ".pkl"))

    if collate_results:
        return results
    else:
        return


def _run_single_experiment(experiment):
    """Runs a single experiment. Returns complete dataframe with an entry for each action."""
    world_dict, agent, steps, verbose, run_nr, save, return_result = experiment
    world_label = world_dict[0]
    world = world_dict[1]
    # if the agent has no random seed assigned yet, assign one now only for this run
    try:
        # if agent.random_seed is None:
        agent.random_seed = random.randint(0, 99999)  # overwrite in any case
    except AttributeError:
        pass
    run_ID = (
        world_label
        + " | "
        + agent.__str__()
        + str(run_nr)
        + " | "
        + str(random.randint(0, 9999))
    )  # unique string representing the run
    sleep = 100
    while psutil.virtual_memory().percent > RAM_LIMIT:
        print(
            "Delaying running",
            agent.__str__(),
            "******",
            world_label,
            f"because of RAM usage. Trying again in {sleep} seconds. RAM usage is "
            + str(psutil.virtual_memory().percent)
            + "%",
        )
        time.sleep(sleep)
        # exponential backoff
        sleep = sleep * 200

    # print('Running', agent.__str__(), '******', world.__str__())
    agent_parameters = agent.get_parameters()
    agent_parameters_w_o_random_seed = {
        key: value for key, value in agent_parameters.items() if key != "random_seed"
    }
    agent_parameters_w_o_random_seed = agent_para_dict(agent_parameters_w_o_random_seed)
    agent.set_world(world)
    # create dataframe
    # columns=DF_COLS+list(agent_parameters.keys()), index=[1])
    r = pd.DataFrame()

    # run subgoal planning
    start_time = time.perf_counter()
    chosen_seq, all_sequences, solved_sequences = agent.plan_subgoals(verbose=verbose)

    r["run_ID"] = [run_ID]
    r["agent"] = [agent_parameters["agent_type"]]
    r["agent_attributes"] = [str(agent_parameters_w_o_random_seed)]
    r["world"] = [world_label]
    r["_world"] = [world]
    r["_all_sequences"] = [all_sequences]
    r["n_all_sequences"] = [len(all_sequences)]
    r["_solved_sequences"] = [solved_sequences]
    r["n_solved_sequences"] = [len(solved_sequences)]
    r["_chosen_subgoal_sequence"] = [chosen_seq]
    r["all_sequences_planning_cost"] = [sum([s.planning_cost() for s in all_sequences])]
    r["_agent"] = [agent]
    r["_world"] = [world]
    for key, value in agent_parameters.items():
        r[key] = [value]
    r["execution_time"] = [time.perf_counter() - start_time]

    # print("Done with", agent.__str__(), '******', world_label, "in", str(round(time.perf_counter() - start_time)),
    #   "seconds. Found", str(len(solved_sequences)), "solutions to", str(len(all_sequences)), "sequences.")

    if save and (SAVE_INTERMEDIATE_RESULTS or not return_result):
        # get folder for experiment
        exp_dir = os.path.join(DF_DIR, save)
        # make sure that the filename is not too long by truncating from the end
        filename = run_ID
        filename = filename[-125:] + ".pkl"
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        # save the results to a file.
        r.to_pickle(os.path.join(exp_dir, filename))
        # garbage collect to avoid memory leaks
        del r
    if return_result:
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
