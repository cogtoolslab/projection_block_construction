import copy
import random
from random import choice, randint

import scoping_simulations.model.utils.decomposition_functions as decomposition_functions
from scoping_simulations.model.BFS_Lookahead_Agent import *
from scoping_simulations.stimuli.subgoal_tree import *

MAX_STEPS = 20
DISPLAY_N_SEQS = 5  # how many sequences to display at most


class Subgoal_Planning_Agent(BFS_Lookahead_Agent):
    """Implements n subgoal planning. Works by running the lower level agent until it has found a solution or times out.

    Whether or not the sequences considered are complete (`include_subsequences`) is determined by the decomposition_function object, not this agent.

    Three costs:
    * solution cost: how expensive was it to find the path of winning actions in the case that it actually found a solution across the sequence of subgoals
    * planning cost: how expensive was it to plan the sequence of subgoals including the failed attempts
    * all sequences planning cost: how expensive was planning over all sequences that were considered, not just the winning one?
    """

    def __init__(
        self,
        world=None,
        decomposer=None,
        step_size=1,  # how many subgoals to act. Negative or zero to act from end of plan
        # randomly sample n sequences. Use `None` to use all possible sequences
        max_number_of_sequences=None,
        c_weight=1,
        # Will keep retrying to solve subgoals until this cost is reached. Set to 1 for a single try (ie. deterministic algorithms)
        max_cost=1,
        lower_agent=BFS_Lookahead_Agent(only_improving_actions=True),
        random_seed=None,
        label="Subgoal Planner",
    ):
        self.world = world
        # only consider sequences of subgoals exactly `lookahead` long or ending on final decomposition
        self.max_number_of_sequences = max_number_of_sequences
        # this is used to make the agent act the entire sequence (0) or only one step (1). Used as default value for `steps` in `act`
        self.step_size = step_size
        self.c_weight = c_weight
        self.max_cost = max_cost
        self.lower_agent = lower_agent
        self.random_seed = random_seed
        self.label = label
        if self.random_seed is None:
            self.random_seed = self.random_seed = randint(0, 99999)
        if decomposer is None:
            decomposer = decomposition_functions.Rectangular_Keyholes(
                sequence_length=3,
                necessary_conditions=[
                    decomposition_functions.Area_larger_than(area=1),
                    # Area_smaller_than(area=30), # used to be 21
                    decomposition_functions.Mass_smaller_than(area=16),
                    decomposition_functions.No_edge_rows_or_columns(),
                ],
                necessary_sequence_conditions=[
                    decomposition_functions.Complete(),
                    decomposition_functions.No_overlap(),
                    decomposition_functions.Supported(),
                ],
            )
        self.decomposer = decomposer
        self._cached_subgoal_evaluations = {}  # sets up cache for  subgoal evaluations

    def __str__(self):
        """Yields a string representation of the agent"""
        return str(self.get_parameters()) + " label: " + self.label

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            **{
                "agent_type": self.__class__.__name__,
                "label": self.label,
                "decomposition_function": self.decomposer.__class__.__name__,
                "number_of_sequences": self.max_number_of_sequences,
                "c_weight": self.c_weight,
                "max_cost": self.max_cost,
                "step_size": self.step_size,
                "random_seed": self.random_seed,
            },
            **{
                "lower level: " + key: value
                for key, value in self.lower_agent.get_parameters().items()
            },
            **{
                "decomposition function: " + key: value
                for key, value in self.decomposer.get_parameters().items()
            },
        }

    def set_world(self, world):
        super().set_world(world)
        self.decomposer.set_silhouette(world.full_silhouette)
        self._cached_subgoal_evaluations = {}  # clear cache

    def act(self, steps=None, verbose=False):
        """Finds subgoal plan, then builds the first _steps_ subgoals. Pass none to build plan length. Steps here refers to subgoals (ie. 2 steps is acting the first two planned subgoals). Pass -1 to steps to execute the entire subgoal plan.

        NOTE: only the latest decomposed silhouette is saved by experiment runner: the plan needs to be extracted from the saved subgoal sequence.
        """
        if self.random_seed is None:
            self.random_seed = randint(0, 99999)
        # get best sequence of subgoals
        sequence, all_sequences, solved_sequences = self.plan_subgoals(verbose=verbose)
        if steps is None:
            if (
                self.step_size <= 0
            ):  # if we've got a negative value, we act the length of the plan up to n steps before the end
                steps = len(sequence) + self.step_size
            else:  # otherwise, we act `step_size` many subgoals (ie. 1)
                steps = self.step_size
        # finally plan and build all subgoals in order
        cur_i = 0
        actions = []
        solution_cost = 0
        partial_planning_cost = 0
        last_silhouette = None
        for sg in sequence:
            if cur_i == steps:
                break  # stop after the nth subgoal
            if sg.actions is None:  # no actions could be found
                print("No actions could be found for subgoal " + str(sg.name))
                continue
            for action in sg.actions:
                # applying the actions to the world — we need force here because the reference of the baseblock objects aren't the same
                self.world.apply_action(action, force=True)
                actions.append(action)
            solution_cost += sg.solution_cost
            partial_planning_cost += sg.planning_cost
            last_silhouette = sg.target
            cur_i += 1
        all_sequences_cost = sum([s.planning_cost() for s in all_sequences])
        return actions, {
            "partial_solution_cost": solution_cost,  # solutions cost of steps acted
            "solution_cost": sequence.solution_cost(),
            "partial_planning_cost": partial_planning_cost,  # planning cost of steps
            "planning_cost": sequence.planning_cost(),
            "num_subgoals_acted": cur_i,  # how many subgoals did we execute?
            "all_sequences_planning_cost": all_sequences_cost,
            "decomposed_silhouette": last_silhouette,
            "_all_subgoal_sequences": all_sequences,
            "_chosen_subgoal_sequence": sequence,
        }

    def plan_subgoals(self, verbose=False):
        """Plan a sequence of subgoals. First, we need to compute a sequence of subgoals however many steps in advance (since completion depends on subgoals). Then, we compute the cost and value of every subgoal in the sequence. Finally, we choose the sequence of subgoals that maximizes the total value over all subgoals within. Returns chosen sequence and the set of all sequences"""
        self.decomposer.set_silhouette(
            self.world.full_silhouette
        )  # make sure that the decomposer has the right silhouette
        sequences = self.decomposer.get_sequences(
            state=self.world.current_state,
            number_of_sequences=self.max_number_of_sequences,
            verbose=verbose,
        )
        if verbose:
            print("Got", len(sequences), "sequences:")
            if verbose > 1:
                for sequence in sequences:
                    print([g.name for g in sequence])
            # sample a few sequences and show them
            if verbose > 2:
                for i, sequence in enumerate(
                    random.sample(sequences, min(len(sequences), DISPLAY_N_SEQS))
                ):
                    sequence.visual_display(
                        blocking=True,
                        title="Sequence {} of {} (showing {})".format(
                            i + 1, len(sequences), min(len(sequences), DISPLAY_N_SEQS)
                        ),
                    )
        # we need to score each in sequence (as it depends on the state before)
        self.fill_subgoals_in_sequence(sequences, verbose=verbose)
        # now we need to find the sequences that maximizes the total value of the parts according to the formula $V_{Z}^{g}(s)=\max _{z \in Z}\left\{R(s, z)-C_{\mathrm{Alg}}(s, z)+V_{Z}^{g}(z)\right\}$
        # return the sequence of subgoals with the highest score, all sequences
        return (
            self.choose_sequence(sequences, verbose=verbose),
            sequences,
            [s for s in sequences if s.complete()],
        )

    def choose_sequence(self, sequences, verbose=False):
        """Chooses the sequence that maximizes $V_{Z}^{g}(s)=\max _{z \in Z}\left\{R(s, z)-C_{\mathrm{Alg}}(s, z)+V_{Z}^{g}(z)\right\}$ including weighing by lambda"""
        # if we get an empty sequence, there is nothing to choose and we return an empty sequence
        if len(sequences) == 0:
            if verbose:
                print("No sequences to choose from")
            return decomposition_functions.Subgoal_sequence([])
        scores = [None] * len(sequences)
        for i in range(len(sequences)):
            scores[i] = sequences[i].V(self.c_weight)
            if verbose:
                print(
                    "Scoring sequence",
                    i + 1,
                    "of",
                    len(sequences),
                    "->",
                    [g.name for g in sequences[i]],
                    "score:\t\t",
                    scores[i],
                )
        try:
            max_scores = max([score for score in scores if score is not None])
        except ValueError:
            # looks like no sequences worked out (this happens when none of the sequences can be solved)
            if verbose:
                print("No sequences have a found solution, choosing empty sequence")
            return decomposition_functions.Subgoal_sequence([])
        top_indices = [i for i in range(len(scores)) if scores[i] == max_scores]
        top_sequences = [sequences[i] for i in top_indices]
        seed(self.random_seed)  # fix random seed
        chosen_sequence = choice(top_sequences)
        if verbose:
            print(
                "Chose sequence:",
                chosen_sequence.names(),
                "with score",
                chosen_sequence.V(self.c_weight),
            )
            # visually display
            if verbose > 2:
                chosen_sequence.visual_display(blocking=True, title="Chosen sequence")
        return chosen_sequence

    def fill_subgoals_in_sequence(
        self, sequences, cumulative_subgoals=False, verbose=False
    ):
        """Computes the cost and value of every subgoal in the sequence. Also computes the cost of the entire sequence. Returns the sequence with the subgoal costs, values and solutions filled in.
        Cumulative subgoals: set to False if the subgoal decompositions do not overlap (ie. rectangular). Shouldn't do any harm even with overlapping decompositions.
        """
        seq_counter = 0  # for verbose printing
        for sequence in sequences:
            current_world = self.world
            if verbose:
                print(
                    "Solving sequence:",
                    str(sequence.names()),
                    "\t",
                    seq_counter,
                    "/",
                    len(sequences),
                )
            if not cumulative_subgoals:
                # we change the targets in the sequence to be cumulative
                sequence = cumulatize(sequence, world=current_world)
            seq_counter += 1  # for verbose printing
            sg_counter = 0  # for verbose printing
            for subgoal in sequence:
                sg_counter += 1  # for verbose printing
                # get reward and cost and success of that particular subgoal and store the resulting world
                subgoal.prior_world = copy.deepcopy(current_world)
                self.solve_subgoal(subgoal, verbose=verbose)
                if verbose:
                    print(
                        "For sequence",
                        seq_counter,
                        "/",
                        len(sequences),
                        str(sequence.names()),
                        "scored subgoal",
                        sg_counter,
                        "/",
                        len(sequence),
                        "named",
                        subgoal.name,
                        "with C:" + str(subgoal.C),
                        " R:" + str(subgoal.R()),
                        " actions:" + str(subgoal.actions),
                    )
                # if we can't solve it to have a base for the next one, we break
                if subgoal.C is None:
                    break
                # store the resulting world as the input to the next one
                current_world = subgoal.past_world

    def solve_subgoal(self, subgoal, verbose=False):
        """Tries as long as needed to find a single solution to the current subgoal"""
        if subgoal.prior_world is None:
            subgoal.prior_world = self.world
        # generate key for cache
        key = subgoal.key()
        if key in self._cached_subgoal_evaluations:
            # print("Cache hit for",key)
            # NOTE this will add the cost taken the first time during planning—ie we don't count the caching for the total cost calculation
            hit = self._cached_subgoal_evaluations[key]
            subgoal.past_world = hit.past_world
            subgoal.actions = hit.actions
            subgoal.C = hit.C
            subgoal.solution_cost = hit.solution_cost
            subgoal.planning_cost = hit.planning_cost
            subgoal.iterations = hit.iterations
            return subgoal
        total_costs = 0
        i = 0
        while total_costs < self.max_cost:
            temp_world = copy.deepcopy(subgoal.prior_world)
            temp_world.set_silhouette(subgoal.target)
            temp_world.current_state.clear()  # clear caches
            if temp_world.current_state.possible_actions() == []:
                # we can't do anything in this world
                break
            self.lower_agent.world = temp_world
            # fix random seed to ensure that we don't needlessly repeat ourselves
            self.lower_agent.random_seed = self.random_seed + i
            steps = 0
            costs = 0
            actions = []
            while (
                temp_world.status()[0] == "Ongoing"
                and costs < self.max_cost
                and steps < MAX_STEPS
            ):
                chosen_actions, info = self.lower_agent.act()
                actions += chosen_actions
                costs += info["states_evaluated"]
                steps += 1
            total_costs += costs
            i += 1  # counting attempts
            if verbose:
                print(
                    "Attempted subgoal",
                    subgoal.name,
                    "on attempt",
                    str(i),
                    "with result",
                    str(temp_world.status()),
                    "and total cost",
                    str(total_costs),
                )
            if temp_world.status()[0] == "Win":
                # we've found a solution! write it to the subgoal
                subgoal.past_world = copy.deepcopy(temp_world)
                subgoal.actions = actions
                subgoal.C = costs
                subgoal.solution_cost = costs
                subgoal.planning_cost = total_costs
                subgoal.iterations = i
                self._cached_subgoal_evaluations[key] = subgoal
                return subgoal
        # if we've made it here, we've failed to find a solution
        # store cached evaluation
        subgoal.solution_cost = None
        subgoal.C = None
        subgoal.planning_cost = total_costs
        subgoal.iterations = i
        self._cached_subgoal_evaluations[key] = subgoal
        return subgoal

    def get_subgoal_tree(self, only_solved_sequences=False, verbose=False):
        """Plans subgoals according to agent specification and returns a subgoal tree of filled out subgoals. Note that this should be preceeded by a call to world.reset() in most cases, as the resulting tree will depend on the state of the world.
        'only_solved_sequences' will ensure that all subgoals in the tree are buildable. Combined with a decomposer that only considers complete sequences this ensures that all subgoals in the tree is on the path to an achievable win.
        """
        # get all subgoals
        sequence, all_sequences, solved_sequences = self.plan_subgoals(verbose=verbose)
        if only_solved_sequences:
            all_sequences = solved_sequences
        # create root node
        root = SubgoalTreeNode(subgoal=None, parent=None, children=[])
        subgoal_tree = SubgoalTree(root=root, world=self.world)
        for sequence in all_sequences:
            subgoal_tree.insert_sequence(sequence)
        return subgoal_tree


def cumulatize(sequence, world=None):
    """Takes a sequence with non-overlapping subgoals and makes it so that each subgoal target also contains the last few. Pass it a world to include the blockmap—this should enable incremental subgoals."""
    previous_targets = sequence.subgoals[0].target
    if world is not None:
        # we assume that the blocks that have already been placed in the world have been placed legally and we can keep them there
        previous_targets = (previous_targets + world.current_state.blockmap) > 0
    for subgoal in sequence.subgoals:
        subgoal.target = previous_targets + subgoal.target
        previous_targets = subgoal.target > 0
    return sequence
