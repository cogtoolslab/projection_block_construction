"""Cost heuristic take in a world with a subgoal mask
and return a numerical value that can be used as feature
to estimate the planning cost of such a subgoal."""


import p_tqdm

from scoping_simulations.model.utils.decomposition_functions import Subgoal


class ActionCostHeuristic:
    """ActionCostHeuristic is a heuristic that estimates the action level cost of a given subgoal.

    Call the created object directly.

    Args:
        subgoal (Subgoal): The subgoal to evaluate the heuristic on

    Returns:
        float: The estimated cost of the subgoat.
    """

    def __init__(self):
        pass

    def __call__(self, subgoal: Subgoal) -> float:
        """Returns the cost of an action in a given world."""
        raise NotImplementedError

    def repeat(self, subgoal: Subgoal, n: int) -> float:
        """Returns the average cost of n runs of the heuristic."""
        costs = []
        for _ in range(n):
            costs.append(self(subgoal))
        return sum(costs) / len(costs)


def run_multiple_heuristics(subgoal: Subgoal, heuristics: [ActionCostHeuristic]):
    """Run multiple heuristics on a world and return a list of their results as a dictionary."""
    results = {}
    for heuristic in heuristics:
        if issubclass(heuristic, ActionCostHeuristic):
            # we have received the class, not an instance
            heuristic = heuristic()
        if not isinstance(heuristic, ActionCostHeuristic):
            raise TypeError("Heuristic must be of type ActionCostHeuristic.")
        # run heuristic
        cost = heuristic(subgoal)
        # save result
        results[heuristic.__class__.__name__] = cost
    return results


def _run_multiple_heuristics_wrapper(args):
    """Wrapper for run_multiple_heuristics to be used with p_tqdm."""
    subgoal, heuristics = args
    return run_multiple_heuristics(subgoal, heuristics)


def run_multiple_heuristics_on_list_of_subgoals(
    subgoals: [Subgoal], heuristics: [ActionCostHeuristic], cpu_ratio=1.0
):
    """Run multiple heuristics on a list of subgoals and return a list of their results as a dictionary.

    Uses p_tqdm for parallel processing.
    """
    # create a list of arguments for the function
    args = []
    for subgoal in subgoals:
        args.append((subgoal, heuristics))
    # run the function
    results = p_tqdm.p_map(_run_multiple_heuristics_wrapper, args, num_cpus=cpu_ratio)
    # results = map(
        _run_multiple_heuristics_wrapper, args
    )  # debug without parallelization
    # convert the results to a list
    results = list(results)
    return results
