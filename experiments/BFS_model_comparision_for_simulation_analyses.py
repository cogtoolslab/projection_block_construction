import argparse

import pandas as pd

from scoping_simulations.model.BFS_Agent import BFS_Agent
from scoping_simulations.model.Simulated_Subgoal_Agent import Simulated_Subgoal_Agent
from scoping_simulations.model.Subgoal_Planning_Agent import *
from scoping_simulations.model.utils.decomposition_functions import *

FRACTION_OF_CPUS = 0.5


def get_BFS_dfs(worlds, save="simulated_subgoal_agent_BFS"):
    decomposer = No_Subgoals()
    sga = Subgoal_Planning_Agent(lower_agent=BFS_Agent(), decomposer=decomposer)
    ns_agent = Simulated_Subgoal_Agent(
        decomposer=decomposer, label="BFS No Subgoals", step_size=-1
    )

    print("Running the subgoal generator")
    results = experiments.subgoal_generator_runner.run_experiment(
        worlds,
        [
            sga,
            1,
            1,
        ],
        parallelized=FRACTION_OF_CPUS,
        save=False,
        maxtasksperprocess=1,
    )

    print("Done. Generating No Subgoal runs...")
    experiments.simulated_subgoal_planner_experiment_runner.run_experiment(
        results, [ns_agent], 1, 16, parallelized=FRACTION_OF_CPUS, save=save
    )

    print("Done.")


def load_df_from_other_agent(file_path):
    """Load in a file that contains simulated subgoal agents with other base algorithms generated by `simulated_subgoal_planner_experiment_runner`. Provide path to .pkl"""
    print("Loading {}".format(df_path))
    df = pd.read_pickle(file_path)
    worlds = list(df["_world"].unique())
    print(f"Got {len(worlds)} worlds.")
    get_BFS_dfs(worlds, save=file_path.split("/")[-1].split(".pkl")[0] + "_BFS")


if __name__ == "__main__":
    # get path from command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", help="path to dataframe to load")
    args = parser.parse_args()
    df_path = args.df_path
    load_df_from_other_agent(df_path)
