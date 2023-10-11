import pandas as pd

from scoping_simulations.model.heuristics.cost_heuristic import (
    ActionCostHeuristic,
    run_multiple_heuristics_on_list_of_subgoals,
)
from scoping_simulations.stimuli.subgoal_tree import ensure_subgoal


def fill_df_with_heuristics(
    df: pd.DataFrame,
    heuristics: [ActionCostHeuristic],
    subgoal_columns: [str] = ["subgoal"],
) -> pd.DataFrame:
    """Fill a dataframe with the results of multiple heuristics.
    Returns a changed dataframe.
    If you include duplicates of the same subgoals,
        the heuristics are run multiple times.

    Args:
        df (pd.DataFrame): The dataframe to fill
        heuristics ([ActionCostHeuristic]): The heuristics to run
        subgoal_columns ([str]): The columns that contain the subgoals

    Returns:
        pd.DataFrame: The filled dataframe
    """
    # create a new dataframe to hold the results
    results = pd.DataFrame()
    # iterate through all subgoal columns
    for subgoal_column in subgoal_columns:
        subgoals = list(df[subgoal_column])
        # ensure that we have subgoals and not subgoal tree nodes
        subgoals = [ensure_subgoal(s) for s in subgoals]
        results = run_multiple_heuristics_on_list_of_subgoals(subgoals, heuristics)
        # add the column name to key
        for i in range(len(results)):
            results[i] = {
                ": ".join([subgoal_column, key]): value
                for key, value in results[i].items()
            }
        # add the results to the dataframe as new columns
        results = pd.DataFrame(results)
        df = df.join(results)
    return df
