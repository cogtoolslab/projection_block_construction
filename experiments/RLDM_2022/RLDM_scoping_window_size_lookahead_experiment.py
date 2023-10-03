"""This experiment is meant to annontate the generated towers with the scoping/no scoping cost."""

if __name__ == "__main__":  # required for multiprocessing
    import time

    import tqdm

    import scoping_simulations.experiments.experiment_runner as experiment_runner
    import scoping_simulations.utils.blockworld as bw
    import scoping_simulations.utils.blockworld_library as bl
    from scoping_simulations.model.Best_First_Search_Agent import (
        Best_First_Search_Agent,
    )
    from scoping_simulations.model.Subgoal_Planning_Agent import *
    from scoping_simulations.model.utils.decomposition_functions import *

    start_time = time.time()

    print("Running experiment....")

    fraction_of_cpus = 1

    # # loading towers from disk
    # PATH_TO_TOWERS = os.path.join(
    #     STIM_DIR, 'generated_towers_bl_nonoverlapping_simple.pkl')
    # # load towers
    # with open(PATH_TO_TOWERS, 'rb') as f:
    #     towers = pickle.load(f)

    # create towers on the fly
    print("Generating towers...")
    block_library = bl.bl_nonoverlapping_simple
    generator = stimuli.tower_generator.TowerGenerator(
        8,
        8,
        block_library=block_library,
        seed=42,
        padding=(2, 0),
        num_blocks=lambda: random.randint(5, 10),
        physics=True,
    )
    NUM_TOWERS = 64
    towers = []
    for i in tqdm.tqdm(range(NUM_TOWERS)):
        towers.append(generator.generate())

    for i in range(len(towers)):
        towers[i]["name"] = str(i)
    towers = {t["name"]: t["bitmap"] for t in towers}
    worlds = {
        name: bw.Blockworld(
            silhouette=silhouette,
            block_library=bl.bl_nonoverlapping_simple,
            legal_action_space=True,
            physics=True,
        )
        for name, silhouette in towers.items()
    }
    print("Made {} towers".format(len(towers)))

    lower_agent = Best_First_Search_Agent(random_seed=42)

    scoping_decomposer = Rectangular_Keyholes(
        sequence_length=1,
        necessary_conditions=[
            Area_larger_than(area=1),
            Area_smaller_than(area=13),
            # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            Proportion_of_silhouette_less_than(ratio=3 / 4),
            No_edge_rows_or_columns(),
            Fewer_built_cells(0),
        ],
        necessary_sequence_conditions=[
            No_overlap(),
            Supported(),
        ],
    )

    lookahead2_decomposer = Rectangular_Keyholes(
        sequence_length=2,
        necessary_conditions=[
            Area_larger_than(area=1),
            # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            Proportion_of_silhouette_less_than(ratio=3 / 4),
            No_edge_rows_or_columns(),
            Fewer_built_cells(0),
        ],
        necessary_sequence_conditions=[
            No_overlap(),
            Supported(),
            Filter_for_length(2),
        ],
    )

    # now we need to generate a number of scoping agents across ranges of window size

    lambdas = [0.0]
    # will have 1 added to them because of exclusive comparision
    window_sizes = [4, 8, 12, 16, 20, 24, 32]

    scoping_agents = [
        Subgoal_Planning_Agent(
            lower_agent=lower_agent,
            decomposer=Rectangular_Keyholes(
                sequence_length=1,
                necessary_conditions=[
                    Area_larger_than(area=1),
                    Area_smaller_than(area=ms + 1),
                    Proportion_of_silhouette_less_than(ratio=3 / 4),
                    No_edge_rows_or_columns(),
                    Fewer_built_cells(0),
                ],
                necessary_sequence_conditions=[
                    No_overlap(),
                    Supported(),
                ],
            ),
            random_seed=42,
            c_weight=cw,
            step_size=1,
            max_number_of_sequences=8192,
            label="Incremental Scoping max size={} lambda={}".format(ms, cw),
        )
        for ms in window_sizes
        for cw in lambdas
    ]

    lookahead2_agents = [
        Subgoal_Planning_Agent(
            lower_agent=lower_agent,
            decomposer=Rectangular_Keyholes(
                sequence_length=2,
                necessary_conditions=[
                    Area_larger_than(area=1),
                    Area_smaller_than(area=ms),
                    No_edge_rows_or_columns(),
                    Fewer_built_cells(0),
                ],
                necessary_sequence_conditions=[
                    No_overlap(),
                    Supported(),
                    Filter_for_length(2),
                ],
            ),
            random_seed=42,
            c_weight=cw,
            step_size=1,
            max_number_of_sequences=8192,
            label="Lookahead Scoping max size={} lambda={}".format(ms, cw),
        )
        for ms in window_sizes
        for cw in lambdas
    ]

    print("Running experiment...")
    results_sg = experiment_runner.run_experiment(
        worlds,
        [*lookahead2_agents],
        per_exp=10,
        steps=16,
        verbose=False,
        parallelized=fraction_of_cpus,
        save="RLDM_scoping_window_size_lookahead_experiment",
        maxtasksperprocess=5,
    )

    print("Done in %s seconds" % (time.time() - start_time))
