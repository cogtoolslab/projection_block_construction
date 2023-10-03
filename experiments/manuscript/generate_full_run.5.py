import os

import scoping_simulations.experiments.subgoal_generator_runner as experiment_runner
import scoping_simulations.utils.blockworld as bw
import scoping_simulations.utils.blockworld_library as bl
from scoping_simulations.model.Best_First_Search_Agent import Best_First_Search_Agent
from scoping_simulations.model.Subgoal_Planning_Agent import Subgoal_Planning_Agent
from scoping_simulations.model.utils.decomposition_functions import *

EXP_NAME = "best first superset 5"
FRACTION_OF_CPUS = 0.9
MAX_LENGTH = 3  # maximum length of sequences to consider

if __name__ == "__main__":  # required for multiprocessing
    try:
        import scoping_simulations.stimuli.tower_generator as tower_generator
    except:
        import tower_generator

    import time

    start_time = time.time()

    # create towers on the fly
    block_library = bl.bl_nonoverlapping_simple

    generator = tower_generator.TowerGenerator(
        8,
        8,
        block_library=block_library,
        seed=3,
        padding=(1, 0),
        num_blocks=lambda: random.randint(
            5, 8
        ),  #  flat random interval of tower sizes (inclusive)
    )

    print("Generating towers")
    NUM_TOWERS = 128
    towers = []
    for i in tqdm(range(NUM_TOWERS)):
        tower = generator.generate()
        towers.append(tower)

    # sort the towers by size
    towers = sorted(towers, key=lambda t: len(t["blocks"]))

    # save out the tower content to disk to a file
    import pickle

    with open(os.path.join("results/dataframes/", f"towers_{EXP_NAME}.pkl"), "wb") as f:
        pickle.dump(towers, f)
        print("Saved towers to {}".format(f))

    worlds = [
        bw.Blockworld(silhouette=t["bitmap"], block_library=bl.bl_nonoverlapping_simple)
        for t in towers
    ]

    print("Generated {} towers".format(len(worlds)))

    print("Tower sizes are {}".format([len(t["blocks"]) for t in towers]))

    print("Running experiment....")

    superset_decomposer = Rectangular_Keyholes(
        sequence_length=MAX_LENGTH,
        necessary_conditions=[
            Mass_larger_than(area=3),  # ignore small subgoals
            # Area_smaller_than(area=30),
            # Mass_smaller_than(area=25),
            No_edge_rows_or_columns(),  # Trim the subgoals to remove empty space on the sides
        ],
        necessary_sequence_conditions=[
            # Complete(), # only consider sequences that are complete—THIS NEEDS TO BE OFF FOR THE SUBGOAL GENERATOR
            No_overlap(),  # do not include overlapping subgoals
            Supported(),  # only consider sequences that could be buildable in theory
        ],
    )
    no_subgoal_decomposer = No_Subgoals()
    decomposer = Decomposition_Functions_Combined(
        [superset_decomposer, no_subgoal_decomposer]
    )

    sga = Subgoal_Planning_Agent(
        lower_agent=Best_First_Search_Agent(), decomposer=decomposer
    )

    agents = [sga]

    results = experiment_runner.run_experiment(
        worlds,
        agents,
        1,
        1,
        verbose=False,
        parallelized=FRACTION_OF_CPUS,
        save=EXP_NAME,
        maxtasksperprocess=1,
    )

    print("Done in %s seconds" % (time.time() - start_time))
