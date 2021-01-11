if __name__=="__main__": #required for multiprocessing
    import os
    import sys
    proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir,'utils')
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir,'model')
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir,'utils')
    sys.path.append(agent_util_dir)
    df_dir = os.path.join(proj_dir,'results/dataframes')

    import pandas as pd
    from model.Simulated_Lookahead_Subgoal_Planning_Agent import *
    from model.Subgoal_Planning_Agent import *
    from model.utils.decomposition_functions import *
    from model.BFS_Agent import BFS_Agent
    import utils.blockworld as bw
    import utils.blockworld_library as bl
    import experiments.simulated_lookahead_subgoal_planner_experiment_runner as experiment_runner

    import time
    start_time = time.time()

    print("Running experiment....")

    fraction_of_cpus = 1

    agents = [
        Simulated_Lookahead_Subgoal_Planning_Agent(
            sequence_length=l,
            include_subsequences=False,
            c_weight=cw
        )
        for l in [1,2,3,4,5,6,7,8] for cw in np.arange(0.,1.,0.1)
        ]

    df_paths = ['subgoal planning full BFS0 to 2.pkl']
    # load all experiments as one dataframe
    df = pd.concat([pd.read_pickle(os.path.join(df_dir,l)) for l in df_paths])
    print("Dataframes loaded")

    

    results = experiment_runner.run_experiment(df,agents,1,20,parallelized=fraction_of_cpus,save="simulated lookaheads TESTING",maxtasksperprocess = 1)
    print(results[['agent','world','world_status']])

    print("Done in %s seconds" % (time.time() - start_time))