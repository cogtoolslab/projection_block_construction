if __name__ == "__main__":  # required for multiprocessing
    import os
    import sys

    print(os.path.dirname(os.path.realpath(__file__)))
    if (
        os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        == "tools_block_construction"
    ):
        proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    else:
        # calling from base folder
        proj_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir, "utils")
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir, "model")
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir, "utils")
    sys.path.append(agent_util_dir)
    df_dir = os.path.join(proj_dir, "results/dataframes")

    import time

    import pandas as pd

    import scoping_simulations.experiments.simulated_lookahead_subgoal_planner_experiment_runner as experiment_runner
    from scoping_simulations.model.Simulated_Lookahead_Subgoal_Planning_Agent import *
    from scoping_simulations.model.Simulated_No_Subgoal_Planning_Agent import *
    from scoping_simulations.model.Subgoal_Planning_Agent import *
    from scoping_simulations.model.utils.decomposition_functions import *

    start_time = time.time()

    print("Running experiment....")

    fraction_of_cpus = 1

    agents = (
        [
            Simulated_Lookahead_Subgoal_Planning_Agent(
                sequence_length=l,
                include_subsequences=False,
                c_weight=cw,
                step_size=1,
                note="Lookahead " + str(l - 1),
            )
            for l in [1]
            for cw in np.linspace(0.0, 0.008, 100)
        ]
        + [
            Simulated_Lookahead_Subgoal_Planning_Agent(
                sequence_length=l,
                include_subsequences=False,
                c_weight=1,
                step_size=-1,
                note="Full",
            )
            for l in [8]
        ]
        + [Simulated_No_Subgoal_Planning_Agent(note="Action level")]
    )

    df_paths = ["subgoal planning full BFS3_rep.pkl"]
    # load all experiments as one dataframe
    df = pd.concat([pd.read_pickle(os.path.join(df_dir, l)) for l in df_paths])
    print("Dataframes loaded:", df_paths)

    results = experiment_runner.run_experiment(
        df,
        agents,
        32,
        20,
        parallelized=fraction_of_cpus,
        save="simulated lookaheads BFS3/final",
        maxtasksperprocess=256,
        chunk_experiments_size=1024,
    )
    # print(results[['agent','world','world_status']])

    print("Done in %s seconds" % (time.time() - start_time))
