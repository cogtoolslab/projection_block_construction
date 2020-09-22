from analysis_helper import *
from analysis_graphs import *
import time

import pandas as pd

df_paths = ['exp_runner_test.pkl']
# df_paths = ['../dataframes/Astar_search.pkl',
#            '../dataframes/breadth_search_no_subsequences.pkl',
#            '../dataframes/beam_search.pkl',
#            '../dataframes/MCTS.pkl',
#            '../dataframes/Naive_Q_search.pkl']
#load all experiments as one dataframe
df = pd.concat([pd.read_pickle(l) for l in df_paths])

print("Dataframes loaded")

start_time = time.time()

heatmaps_at_peak_per_agent_over_world(df)

# mean_win_per_agent_over_worlds(df)

# mean_peak_F1_per_agent_over_worlds(df)

# mean_failure_reason_per_agent_over_worlds(df)

# mean_win_per_agent(df)

# mean_peak_score_per_agent(df)

# mean_failure_reason_per_agent(df)

# avg_steps_to_end_per_agent(df)

# mean_avg_area_under_curve_to_peakF1_per_agent(df)

# graph_mean_F1_over_time_per_agent(df)

# graph_avg_blocksize_over_time_per_agent(df)

# mean_touching_last_block_per_agent(df)

# mean_pairwise_raw_euclidean_distance_between_runs(df)

# total_avg_states_evaluated_per_agent(df)

print("Done in %s seconds" % (time.time() - start_time))