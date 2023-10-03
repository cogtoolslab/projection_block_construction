# Various imports

import os

# go up one folder until we find the project folder
while os.path.basename(os.getcwd()) != "tools_block_construction":
    os.chdir("..")
# add the project folder to the path
PROJ_DIR = os.getcwd()

# get relevant other directories


import scoping_simulations.experiments.simulated_subgoal_planner_experiment_runner as simulated_subgoal_planner_experiment_runner
import scoping_simulations.experiments.subgoal_generator_runner as subgoal_generator_runner
import scoping_simulations.stimuli.tower_generator as tower_generator
import scoping_simulations.utils.blockworld as bw
import scoping_simulations.utils.blockworld_library as bl
from scoping_simulations.model.Best_First_Search_Agent import Best_First_Search_Agent
from scoping_simulations.model.Simulated_Subgoal_Agent import Simulated_Subgoal_Agent
from scoping_simulations.model.Subgoal_Planning_Agent import Subgoal_Planning_Agent

# import the relevant modules
from scoping_simulations.model.utils.decomposition_functions import *

# set up the experiment
EXP_NAME = "Simulated Subgoals on Best First Search Template"
FRACTION_OF_CPUS = 1  # how many CPUs to use. False will turn off parallelization (for debugging). 1 will use all available CPUs, 0.5 will use half, etc.
NUM_TOWERS = 4
MAX_LENGTH = 3  # maximum length of sequences to consider

# only execute the following code if this file is run directly, not during import or parallelization
if __name__ == "__main__":  # required for multiprocessing
    import time

    start_time = time.time()

    # We need to generate the towers we want to use
    print("Generating towers...")
    generator = tower_generator.TowerGenerator(
        4,
        4,  # 8, 8,
        block_library=bl.bl_nonoverlapping_simple,
        seed=3,
        padding=(0, 0),  # (1, 0),
        num_blocks=lambda: random.randint(
            6, 18
        ),  #  flat random interval of tower sizes (inclusive)
    )

    towers = []
    for i in tqdm(range(NUM_TOWERS)):
        tower = generator.generate()
        towers.append(tower)
    worlds = [
        bw.Blockworld(silhouette=t["bitmap"], block_library=bl.bl_nonoverlapping_simple)
        for t in towers
    ]
    print("Generated {} towers".format(len(worlds)))

    # Now that we have the towers, we generate the superset of subgoal sequences
    print("Generating superset of subgoal sequences")
    superset_decomposer = Rectangular_Keyholes(
        sequence_length=MAX_LENGTH,
        necessary_conditions=[
            Area_larger_than(area=1),  # ignore subgoals smaller than the smallest block
            # Area_smaller_than(area=30),
            # Mass_smaller_than(area=18),
            No_edge_rows_or_columns(),  # Trim the subgoals to remove empty space on the sides
        ],
        necessary_sequence_conditions=[
            # Complete(), # only consider sequences that are complete—THIS NEEDS TO BE OFF FOR THE SUBGOAL GENERATOR
            No_overlap(),  # do not include overlapping subgoals
            Supported(),  # only consider sequences that could be buildable in theory
        ],
    )
    sga = Subgoal_Planning_Agent(
        lower_agent=Best_First_Search_Agent(), decomposer=superset_decomposer
    )
    # actually run it
    superset_results = subgoal_generator_runner.run_experiment(
        worlds,
        [sga],
        1,
        1,
        verbose=False,
        parallelized=FRACTION_OF_CPUS,
        save=EXP_NAME + "_cached_superset",
        maxtasksperprocess=1,
        collate_results=True,
    )
    print("Done in %s seconds" % (time.time() - start_time))
    print(f"Generated {len(superset_results)} lines of data from the parent agent(s))")

    # In practice you might want to split here and load the pickled file from disk
    # Now we have the cached superset of subgoal sequences, we can use them to simulate subgoal agents
    start_time = time.time()
    print("Running simulated subgoal agents...")

    no_subgoals_decomposer = No_Subgoals()

    myopic_decomposer = Rectangular_Keyholes(
        sequence_length=1,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=superset_decomposer.necessary_sequence_conditions,
    )

    lookahead_1_decomposer = Rectangular_Keyholes(
        sequence_length=1 + 1,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=superset_decomposer.necessary_sequence_conditions,
    )

    lookahead_2_decomposer = Rectangular_Keyholes(
        sequence_length=1 + 2,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=superset_decomposer.necessary_sequence_conditions,
    )

    full_decomp_decomposer = Rectangular_Keyholes(
        sequence_length=MAX_LENGTH,
        necessary_conditions=superset_decomposer.necessary_conditions,
        necessary_sequence_conditions=[
            Complete(),  # only consider sequences that are complete
        ]
        + superset_decomposer.necessary_sequence_conditions,
    )

    agents = [
        Simulated_Subgoal_Agent(decomposer=no_subgoals_decomposer, label="No Subgoals"),
        Simulated_Subgoal_Agent(decomposer=myopic_decomposer, label="Myopic"),
        Simulated_Subgoal_Agent(decomposer=lookahead_1_decomposer, label="Lookahead 1"),
        Simulated_Subgoal_Agent(decomposer=lookahead_2_decomposer, label="Lookahead 2"),
        Simulated_Subgoal_Agent(
            decomposer=full_decomp_decomposer, label="Full Decomp", step_size=-1
        ),  # step size of -1 means use the full sequence
    ]

    # actually run it
    results = simulated_subgoal_planner_experiment_runner.run_experiment(
        superset_results,
        agents,
        per_exp=1,
        steps=32,
        parallelized=FRACTION_OF_CPUS,
        save=EXP_NAME + "_simulated_results",
        maxtasksperprocess=1,
    )
    print("Done in %s seconds" % (time.time() - start_time))
    print(f"Generated {len(results)} lines of data from the simulated subgoal agents")
    print(
        "Files saved to {} with postfixes".format(
            os.path.join(PROJ_DIR, "experiments", "results", EXP_NAME)
        )
    )
