from scoping_simulations.model.Resource_Rational_Subgoal_Planning_Agent import *


class Lax_Resource_Rational_Subgoal_Planning_Agent(
    Resource_Rational_Subgoal_Planning_Agent
):
    """Plans every subgoal as good as it can and uses that as basis for the next one no matter if it's complete. Reward is how much we were actually able to build"""

    def __init__(
        self,
        world=None,
        decomposer=None,
        lookahead=1,
        include_subsequences=False,
        c_weight=1000,
        S_iterations=1,
        lower_agent=BFS_Lookahead_Agent(only_improving_actions=True),
        random_seed=None,
    ):
        super().__init__(
            world=world,
            decomposer=decomposer,
            lookahead=lookahead,
            include_subsequences=include_subsequences,
            c_weight=c_weight,
            S_treshold=0,
            S_iterations=S_iterations,
            lower_agent=lower_agent,
            random_seed=random_seed,
        )
        if S_iterations != 1:
            print(
                "S_iterations larger than 1 are not implemented yet—only last result will be passed on"
            )

    def score_subgoals_in_sequence(self, sequences, verbose=False):
        """Add C,R,S to the subgoals in the sequences"""
        number_of_states_evaluated = 0
        seq_counter = 0  # for verbose printing
        for sequence in sequences:  # reference or copy?
            seq_counter += 1  # for verbose printing
            sg_counter = 0  # for verbose printing
            prior_world = self.world
            for subgoal in sequence:
                sg_counter += 1  # for verbose printing
                # get reward and cost and success of that particular subgoal and store the resulting world
                S, C, post_world, total_cost, stuck = self.success_and_cost_of_subgoal(
                    subgoal["decomposition"], prior_world, iterations=self.S_iterations
                )
                R = self.reward_of_subgoal(
                    post_world.current_state.blockmap,
                    prior_world.current_state.blockmap,
                )
                number_of_states_evaluated += total_cost
                if verbose:
                    print(
                        "For sequence",
                        seq_counter,
                        "/",
                        len(sequences),
                        "scored subgoal",
                        sg_counter,
                        "/",
                        len(sequence),
                        "named",
                        subgoal["name"],
                        "with C:" + str(C),
                        " R:" + str(R),
                        " S:" + str(S),
                    )
                # store in the subgoal
                subgoal["R"] = R + stuck * BAD_SCORE
                subgoal["C"] = C
                subgoal["S"] = S
                # if we can't solve it to have a base for the next one, we break
                if prior_world is None:
                    break
        return number_of_states_evaluated

    def reward_of_subgoal(self, post_blockmap, prior_blockmap):
        """Gets the unscaled reward of a subgoal: the area of a figure that we can fill out by completing the subgoal in number of cells beyond what is already filled out."""
        return np.sum(
            ((post_blockmap > 0) * self.world.full_silhouette) - (prior_blockmap > 0)
        )

    def success_and_cost_of_subgoal(
        self, decomposition, prior_world=None, iterations=1, max_steps=20
    ):
        """The cost of solving for a certain subgoal given the current block construction"""
        if prior_world is None:
            prior_world = self.world
        # generate key for cache
        key = decomposition * 100 - (
            prior_world.current_state.order_invariant_blockmap() > 0
        )
        key = key.tostring()  # make hashable
        if key in self._cached_subgoal_evaluations:
            # print("Cache hit for",key)
            cached_eval = self._cached_subgoal_evaluations[key]
            return (
                cached_eval["S"],
                cached_eval["C"],
                cached_eval["winning_world"],
                1,
                cached_eval["stuck"],
            )  # returning 1 as lookup cost, not the cost it tool to calculate the subgoal originally
        current_world = copy.deepcopy(prior_world)
        costs = 0
        wins = 0
        post_world = None
        stuck = 0  # have we taken no actions in all iterations?
        for i in range(iterations):
            temp_world = copy.deepcopy(current_world)
            temp_world.set_silhouette(decomposition)
            temp_world.current_state.clear()  # clear caches
            self.lower_agent.world = temp_world
            steps = 0
            while temp_world.status()[0] == "Ongoing" and steps < max_steps:
                _, info = self.lower_agent.act()
                costs += info["states_evaluated"]
                steps += 1
            post_world = copy.deepcopy(temp_world)
            wins += temp_world.status()[0] == "Win"
            if (
                steps == 0
            ):  # we have no steps, which means that the subgoal will lead to infinite costs
                stuck += 1
        # store cached evaluation
        cached_eval = {
            "S": wins / iterations,
            "C": costs / iterations,
            "winning_world": post_world,
            "stuck": stuck == iterations,
        }
        self._cached_subgoal_evaluations[key] = cached_eval
        return (
            wins / iterations,
            costs / iterations,
            post_world,
            costs / iterations,
            stuck == iterations,
        )


class Full_Lax_Sample_Subgoal_Planning_Agent(
    Lax_Resource_Rational_Subgoal_Planning_Agent
):
    """Same as subgoal planning agent, only that we act the entire sequence of subgoals after planning and plan the entire sequence."""

    def __init__(
        self,
        world=None,
        decomposer=None,
        c_weight=1000,
        S_iterations=1,
        lower_agent=BFS_Lookahead_Agent(only_improving_actions=True),
        random_seed=None,
    ):
        super().__init__(
            world=world,
            decomposer=decomposer,
            lookahead=MAX_NUMBER_OF_SUBGOALS,
            include_subsequences=False,
            c_weight=c_weight,
            S_iterations=S_iterations,
            lower_agent=lower_agent,
            random_seed=random_seed,
        )

    def act(self, verbose=False):
        """Plans and acts entire sequence"""
        return super().act(steps=MAX_NUMBER_OF_SUBGOALS, verbose=verbose)
