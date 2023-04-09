FRACTION_OF_CPUS = .5
FILE_BY_FILE = True # if true, only load the df needed to memory and save out incremental results
PER_EXP = 1#16 # number of repetitions of each experiment
STEPS = 6 # maximum number of steps to run the experiment for
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]
CHUNK_SIZE = 256

if __name__=="__main__": #required for multiprocessing
    import os
    import sys
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir,'utils')
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir,'model')
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir,'utils')
    sys.path.append(agent_util_dir)
    df_dir = os.path.join(proj_dir,'results/dataframes')

    import pandas as pd
    import tqdm
    from model.Simulated_Subgoal_Agent import *
    from model.Subgoal_Planning_Agent import *
    from model.utils.decomposition_functions import *
    import utils.blockworld_library as bl
    import experiments.simulated_subgoal_planner_experiment_runner as experiment_runner

    # get path to dataframes as input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', help='path to dataframe to load')
    args = parser.parse_args()
    df_folder_path = args.df_path

    if df_folder_path is None:
        # try to load the latest .pkl file from the results/dataframes directory
        # sort by date modified
        df_folder_path = sorted([os.path.join(df_dir, f) for f in os.listdir(df_dir) if os.path.isdir(os.path.join(df_dir, f))], key=os.path.getmtime)[-1]
        print("No dataframe path provided. Loading latest dataframe from results/dataframes: {}".format(df_folder_path))

    print("Got path to folder of dataframes: {}".format(df_folder_path))
    df_paths = [os.path.join(df_folder_path, f) for f in os.listdir(df_folder_path) if f.endswith('.pkl')]
    print("Found {} dataframes in folder".format(len(df_paths)))
    if len(df_paths) == 0:
        raise Exception("No dataframes found in folder {}".format(df_folder_path))

    expname = os.path.basename(df_folder_path).split('.')[0]
    # clean up common prefixes
    expname = expname.replace('dataframes','')
    expname = expname.replace('subgoal_generator','')
    expname = expname.replace('generator','')
    expname = expname.replace('superset','')
    expname = expname.replace('  ',' ')
    if expname.startswith('_'):
        expname = expname[1:]
    expname = "simulated_subgoal_agents_" + expname
    print("Experiment name: {}".format(expname))

    # we want to save the results of each experiment to a separate file in expame folder. If that exists, append to it
    exp_folder_path = os.path.join(df_dir, expname)
    i = 0
    while not os.path.exists(exp_folder_path):
        try:
            os.makedirs(exp_folder_path)
        except FileExistsError:
            i += 1
            exp_folder_path = exp_folder_path + f"_{i}"

    import time
    start_time = time.time()

    print("Running experiment....")

    MAX_LENGTH = 3 # maximum length of sequences to consider
    superset_decomposer = Rectangular_Keyholes(
    sequence_length=MAX_LENGTH,
        necessary_conditions=[
            Mass_larger_than(area=3), # ignore small subgoals
            # Area_smaller_than(area=30),
            Mass_smaller_than(area=18),
            No_edge_rows_or_columns(), # Trim the subgoals to remove empty space on the sides
        ],
    necessary_sequence_conditions=[
        # Complete(), # only consider sequences that are complete—THIS NEEDS TO BE OFF FOR THE SUBGOAL GENERATOR
        No_overlap(), # do not include overlapping subgoals
        Supported(), # only consider sequences that could be buildable in theory
    ]
    )


    no_subgoals_decomposer = No_Subgoals()

    myopic_decomposer = Rectangular_Keyholes(
        sequence_length=1,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=superset_decomposer.necessary_sequence_conditions
    )

    lookahead_1_decomposer = Rectangular_Keyholes(
        sequence_length=1+1,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=superset_decomposer.necessary_sequence_conditions
    )

    # lookahead_2_decomposer = Rectangular_Keyholes( # sort of pointless
    #     sequence_length=1+2,
    #     necessary_conditions=superset_decomposer.necessary_conditions,
    #     necessary_sequence_conditions=superset_decomposer.necessary_sequence_conditions
    # )

    full_decomp_decomposer = Rectangular_Keyholes(
        sequence_length=MAX_LENGTH,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=[
            Complete(), # only consider sequences that are complete
        ] + superset_decomposer.necessary_sequence_conditions
    )

    # create agents for each lambda
    agents = []
    for l in LAMBDAS:
        l_agents = [
            Simulated_Subgoal_Agent(c_weight = l, decomposer=no_subgoals_decomposer, label="No Subgoals"),
        Simulated_Subgoal_Agent(c_weight = l, decomposer=myopic_decomposer, label="Myopic"),
        Simulated_Subgoal_Agent(c_weight = l, decomposer=lookahead_1_decomposer, label="Lookahead"),
        # Simulated_Subgoal_Agent(c_weight = l, decomposer=lookahead_2_decomposer, label="Lookahead 2")
        ]
        agents += l_agents

    full_agents = [
        Simulated_Subgoal_Agent(decomposer=full_decomp_decomposer, label="Full Decomp", step_size=-1), # step size of -1 means use the full sequence
        ]
    agents += full_agents

    print(f"Created {len(agents)} agents")

    if FILE_BY_FILE:
        # load and run experiments one by one
        for i,df_path in tqdm(enumerate(df_paths), total = len(df_paths)):
            df = pd.read_pickle(df_path)
            print("Dataframe loaded:",df_path)

            print("Results will be saved to:",os.path.join(exp_folder_path,f"{expname}_{i})")+".pkl")

            results = experiment_runner.run_experiment(df,agents,PER_EXP,STEPS,parallelized=FRACTION_OF_CPUS,save=os.path.join(exp_folder_path,f"{expname}_{i})"),maxtasksperprocess = 256,chunk_experiments_size=CHUNK_SIZE)
    else:
        # load all experiments as one dataframe
        df = pd.concat([pd.read_pickle(os.path.join(df_dir,l)) for l in df_paths])
        print("Dataframes loaded:",df_paths)

        print("Results will be saved to:",expname)

        results = experiment_runner.run_experiment(df,agents,PER_EXP,STEPS,parallelized=FRACTION_OF_CPUS,save=expname,maxtasksperprocess = 256,chunk_experiments_size=CHUNK_SIZE)

    print("Done in %s seconds" % (time.time() - start_time))
    print("Results saved to:",expname)