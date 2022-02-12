"""This experiment is meant to annontate the generated towers with the scoping/no scoping cost."""

if __name__ == "__main__":  # required for multiprocessing
    import os
    import sys
    proj_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir, 'utils')
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir, 'model')
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir, 'utils')
    sys.path.append(agent_util_dir)
    stim_dir = os.path.join(proj_dir, 'stimuli')
    sys.path.append(stim_dir)

    from model.Subgoal_Planning_Agent import *
    from model.utils.decomposition_functions import *
    from model.BFS_Agent import BFS_Agent
    from model.Astar_Agent import Astar_Agent
    from model.Best_First_Search_Agent import Best_First_Search_Agent
    import utils.blockworld as bw
    import utils.blockworld_library as bl
    import experiments.experiment_runner as experiment_runner
    # import experiments.subgoal_generator_runner as experiment_runner

    import pickle

    import time
    start_time = time.time()

    print("Running experiment....")

    fraction_of_cpus = 0.8

    PATH_TO_TOWERS = os.path.join(
        stim_dir, 'generated_towers_bl_nonoverlapping_simple.pkl')
    # load towers
    with open(PATH_TO_TOWERS, 'rb') as f:
        towers = pickle.load(f)
    for i in range(len(towers)):
        towers[i]['name'] = str(i)
    # limit to a few words
    towers = towers[:10]
    towers = {t['name']: t['bitmap'] for t in towers}
    worlds = {name: bw.Blockworld(silhouette=silhouette, block_library=bl.bl_nonoverlapping_simple,
                                         legal_action_space=True, physics=False) for name, silhouette in towers.items()}

    lower_agent = Best_First_Search_Agent()
    decomposer = Rectangular_Keyholes(necessary_conditions=[
        Area_larger_than(area=4),
        # Area_smaller_than(area=28),
        No_edge_rows_or_columns(),
    ],
        necessary_sequence_conditions=[
        Complete(),
        No_overlap(),
        Supported(),
    ]
    )

    agents = [
        Subgoal_Planning_Agent(lower_agent=lower_agent,
                            decomposer=decomposer,
                            sequence_length=4,
                            include_subsequences=True,
                            number_of_sequences=64),
        lower_agent,
    ]

    results = experiment_runner.run_experiment(
        worlds, agents, per_exp=1, steps=16, verbose=False, parallelized=fraction_of_cpus, save="scoring towers", maxtasksperprocess=5)

    print("Done in %s seconds" % (time.time() - start_time))
