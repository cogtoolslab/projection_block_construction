"""This experiment is meant to annontate the generated towers with the scoping/no scoping cost."""

if __name__ == "__main__":  # required for multiprocessing
    import os
    import sys

    proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir, "utils")
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir, "model")
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir, "utils")
    sys.path.append(agent_util_dir)
    stim_dir = os.path.join(proj_dir, "stimuli")
    sys.path.append(stim_dir)

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
    #     stim_dir, 'generated_towers_bl_nonoverlapping_simple.pkl')
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

    full_decomposer3 = Rectangular_Keyholes(
        sequence_length=3,
        necessary_conditions=[
            Area_larger_than(area=1),
            Proportion_of_silhouette_less_than(
                ratio=3 / 4
            ),  # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            No_edge_rows_or_columns(),
        ],
        necessary_sequence_conditions=[
            Complete(),
            No_overlap(),
            Supported(),
        ],
    )

    full_decomposer2 = Rectangular_Keyholes(
        sequence_length=2,
        necessary_conditions=[
            Area_larger_than(area=1),
            Proportion_of_silhouette_less_than(
                ratio=3 / 4
            ),  # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            No_edge_rows_or_columns(),
        ],
        necessary_sequence_conditions=[
            Complete(),
            No_overlap(),
            Supported(),
        ],
    )

    full_decomposer4 = Rectangular_Keyholes(
        sequence_length=4,
        necessary_conditions=[
            Area_larger_than(area=1),
            Proportion_of_silhouette_less_than(
                ratio=3 / 4
            ),  # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            No_edge_rows_or_columns(),
        ],
        necessary_sequence_conditions=[
            Complete(),
            No_overlap(),
            Supported(),
        ],
    )

    scoping_decomposer = Rectangular_Keyholes(
        sequence_length=1,
        necessary_conditions=[
            Area_larger_than(area=1),
            Proportion_of_silhouette_less_than(
                ratio=3 / 4
            ),  # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
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
            Proportion_of_silhouette_less_than(
                ratio=3 / 4
            ),  # maximum subgoal size is 3/4 of the mass of the tower. Prevents degeneratee case of 1 subgoal
            No_edge_rows_or_columns(),
            Fewer_built_cells(0),
        ],
        necessary_sequence_conditions=[
            No_overlap(),
            Supported(),
            Filter_for_length(2),
        ],
    )

    full_subgoal2_agent = Subgoal_Planning_Agent(
        lower_agent=lower_agent,
        decomposer=full_decomposer2,
        random_seed=42,
        c_weight=1.0,
        step_size=0,
        max_number_of_sequences=8192,
        label="Full Subgoal Decomposition 2",
    )

    full_subgoal3_agent = Subgoal_Planning_Agent(
        lower_agent=lower_agent,
        decomposer=full_decomposer3,
        random_seed=42,
        c_weight=1.0,
        step_size=0,
        max_number_of_sequences=8192,
        label="Full Subgoal Decomposition 3",
    )

    full_subgoal4_agent = Subgoal_Planning_Agent(
        lower_agent=lower_agent,
        decomposer=full_decomposer4,
        random_seed=42,
        c_weight=1.0,
        step_size=0,
        max_number_of_sequences=8192,
        label="Full Subgoal Decomposition 4",
    )

    scoping_agent = Subgoal_Planning_Agent(
        lower_agent=lower_agent,
        decomposer=scoping_decomposer,
        random_seed=42,
        c_weight=1.0,
        step_size=1,
        max_number_of_sequences=8192,
        label="Incremental Scoping",
    )

    lookahead2_agent = Subgoal_Planning_Agent(
        lower_agent=lower_agent,
        decomposer=lookahead2_decomposer,
        random_seed=42,
        c_weight=1.0,
        step_size=1,
        max_number_of_sequences=8192,
        label="Lookahead Scoping",
    )

    print("Running experiment...")
    results_sg = experiment_runner.run_experiment(
        worlds,
        [full_subgoal2_agent, full_subgoal3_agent, lower_agent],
        per_exp=10,  # since we might have different solutions depending on best first order
        steps=16,
        verbose=False,
        parallelized=fraction_of_cpus,
        save="RLDM_full_decomp_experiment",
        maxtasksperprocess=5,
    )

    print("Done in %s seconds" % (time.time() - start_time))
