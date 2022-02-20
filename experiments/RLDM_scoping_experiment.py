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
    import experiments.subgoal_generator_runner as subgoal_generator_runner
    import stimuli.tower_generator
    import tqdm

    import pickle

    import time
    start_time = time.time()

    print("Running experiment....")

    fraction_of_cpus = 1

    # # loading towers from disk
    # PATH_TO_TOWERS = os.path.join(
    #     stim_dir, 'generated_towers_bl_nonoverlapping_simple.pkl')
    # # load towers
    # with open(PATH_TO_TOWERS, 'rb') as f:
    #     towers = pickle.load(f)

    # create towers on the fly
    print("Generating towers...")
    block_library = bl.bl_nonoverlapping_simple
    generator = stimuli.tower_generator.TowerGenerator(8, 8,
                                                       block_library=block_library,
                                                       seed=42,
                                                       padding=(2, 0),
                                                       num_blocks=lambda: random.randint(
                                                           5, 10),
                                                       physics=True,
                                                       )
    NUM_TOWERS = 64
    towers = []
    for i in tqdm.tqdm(range(NUM_TOWERS)):
        towers.append(generator.generate())

    for i in range(len(towers)):
        towers[i]['name'] = str(i)
    towers = {t['name']: t['bitmap'] for t in towers}
    worlds = {name: bw.Blockworld(silhouette=silhouette, block_library=bl.bl_nonoverlapping_simple,
                                  legal_action_space=True, physics=True) for name, silhouette in towers.items()}
    print("Made {} towers".format(len(towers)))

    lower_agent = Best_First_Search_Agent(random_seed=42)

    scoping_decomposer = Rectangular_Keyholes(
        sequence_length=1,
        necessary_conditions=[
            Area_larger_than(area=1),
            # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            Proportion_of_silhouette_less_than(ratio=3/4),
            No_edge_rows_or_columns(),
            Fewer_built_cells(0),
        ],
        necessary_sequence_conditions=[
            No_overlap(),
            Supported(),
        ]
    )

    lookahead2_decomposer = Rectangular_Keyholes(
        sequence_length=2,
        necessary_conditions=[
            Area_larger_than(area=1),
            # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            Proportion_of_silhouette_less_than(ratio=3/4),
            No_edge_rows_or_columns(),
            Fewer_built_cells(0),
        ],
        necessary_sequence_conditions=[
            No_overlap(),
            Supported(),
            Filter_for_length(2),
        ]
    )

    # now we need to generate a number of scoping agents across ranges of c_weight
    
    lambdas = np.arange(0, 0.525, 0.025)

    scoping_agents = [Subgoal_Planning_Agent(lower_agent=lower_agent,
                                             decomposer=scoping_decomposer,
                                             random_seed=42,
                                             c_weight=cw,
                                             step_size=1,
                                             max_number_of_sequences=8192,
                                             label="Incremental Scoping lambda={}".format(cw)) for cw in lambdas]

    lookahead2_agents = [Subgoal_Planning_Agent(lower_agent=lower_agent,
                                                decomposer=lookahead2_decomposer,
                                                random_seed=42,
                                                c_weight=cw,
                                                step_size=1,
                                                max_number_of_sequences=8192,
                                                label="Lookahead Scoping lambda={}".format(cw)) for cw in lambdas]

    print("Running experiment...")
    results_sg = experiment_runner.run_experiment(
        worlds,
        [*scoping_agents],
        per_exp=10,
        steps=16,
        verbose=False,
        parallelized=fraction_of_cpus,
        save="RLDM_scoping_experiment",
        maxtasksperprocess=5)

    print("Done in %s seconds" % (time.time() - start_time))
