"""This file is a workaround for the fact that some simulations in Apr 2023 were run with incorrect conditions on the No Subgoals decomposer.

THis file reads in the rows that have missing data from the analysis notebook, and produces a new dataframe with the correct data.

This is very much a hack, and should be removed once the data is re-run.
"""

import os
import sys
import pandas as pd
import tqdm
proj_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(proj_dir)
agent_dir = os.path.join(proj_dir,'model')
sys.path.append(agent_dir)
agent_util_dir = os.path.join(agent_dir,'utils')
sys.path.append(agent_util_dir)

from scoping_simulations.model.utils.decomposition_functions import *
from scoping_simulations.model.Subgoal_Planning_Agent import Subgoal_Planning_Agent
from scoping_simulations.model.Simulated_Subgoal_Agent import Simulated_Subgoal_Agent
from scoping_simulations.model.Best_First_Search_Agent import Best_First_Search_Agent
import scoping_simulations.experiments.subgoal_generator_runner as subgoal_generator_runner
import scoping_simulations.experiments.simulated_subgoal_planner_experiment_runner as simulated_subgoal_planner_experiment_runner

PATH = "/home/AD/fbinder/tools_block_construction/analysis/Simulation_Analysis_No_Subgoals_Failures.pkl"

def run():
        # set up stuff
    no_subgoal_decomposer = No_Subgoals()
    sga = Subgoal_Planning_Agent(lower_agent=Best_First_Search_Agent(),
                                    decomposer=no_subgoal_decomposer)

    simulated_sga = Simulated_Subgoal_Agent(decomposer=no_subgoal_decomposer, label="No Subgoals", step_size=-1)

    df = pd.read_pickle(PATH)

    print("Loaded dataframe with {} rows".format(len(df)))

    sim_dfs = []

    for i,row in tqdm(list(df.iterrows())):
        # get the world
        world = row['_world']
        world_label = row['world']
        run_ID = row['run_ID']
        
        # get the solutions
        experiment = ((world_label, world), sga, -1, False, i, False, True) #     world_dict, agent, steps, verbose, run_nr, save, return_result = experiment
        result_generator = subgoal_generator_runner._run_single_experiment(experiment)

        # now that we have the result, we use the subgoal generator to get the properly formated result
        result_sim = simulated_subgoal_planner_experiment_runner.run_experiment(result_generator, [simulated_sga], per_exp=1, steps=64, parallelized=False, save=False)
        sim_dfs.append(result_sim)
    
    print("Done running simulations")
    sim_df = pd.concat(sim_dfs)
    print("Got {} rows".format(len(sim_df)))
    sim_df.to_pickle(PATH.replace(".pkl","_fixed.pkl"))
    print("Saved to {}".format(PATH.replace(".pkl","_fixed.pkl")))

if __name__ == "__main__":
    run()
