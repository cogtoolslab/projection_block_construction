FRACTION_OF_CPUS = 1

if __name__=="__main__": #required for multiprocessing
    import os
    import sys
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    print(proj_dir)
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir,'utils')
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir,'model')
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir,'utils')
    sys.path.append(agent_util_dir)
    df_dir = os.path.join(proj_dir,'results/dataframes')

    import pandas as pd
    from model.Simulated_Subgoal_Agent import *
    from model.Simulated_No_Subgoal_Planning_Agent import *
    from model.Subgoal_Planning_Agent import *
    from model.utils.decomposition_functions import *
    import utils.blockworld_library as bl
    import experiments.simulated_subgoal_planner_experiment_runner as experiment_runner

    # get path to dataframes as input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', nargs='+', help='path to dataframe to load')
    args = parser.parse_args()
    df_path = args.df_path

    expname = df_path.split('/')[-1].split('.')[0] + "_simulated_subgoals"


    import time
    start_time = time.time()

    print("Running experiment....")


    agents =  [
        Simulated_Subgoal_Agent(
            sequence_length=l,
            include_subsequences=False,
            c_weight=cw,
            step_size = 1,
            note = "Lookahead "+str(l-1)
        )
        for l in [1] for cw in np.linspace(0.,10.,100)
        ]+ [
            Simulated_Subgoal_Agent(
            sequence_length=l,
            include_subsequences=False,
            c_weight=1,
            step_size = -1,
            note = "Full"
        )
        for l in [1]
        ]+[
            Simulated_No_Subgoal_Planning_Agent(
                note = "Action level"
            )
        ]

    df_paths = [df_paths]
    # load all experiments as one dataframe
    df = pd.concat([pd.read_pickle(os.path.join(df_dir,l)) for l in df_paths])
    print("Dataframes loaded:",df_paths)

    results = experiment_runner.run_experiment(df,agents,32,20,parallelized=FRACTION_OF_CPUS,save=expname,maxtasksperprocess = 256,chunk_experiments_size=2048)

    print("Done in %s seconds" % (time.time() - start_time))