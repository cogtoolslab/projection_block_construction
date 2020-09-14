"""This file contains helper functions to generate plots and scalar measures. 
Actual code for graphs can be found in graphs.py"""

#setup
import blockworld as bw
import blockworld_library as bl
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from pathos.multiprocessing import ProcessPool

_final_row_dict = {} #type: dict

class State():
    """A dummy state to pass to blockworld scoring functions"""
    def __init__(self,world,blockmap):
        self.blockmap = blockmap
        self.world = world

def smart_short_agent_names(attr_dicts):
    """Takes in a list of agent attributes and returns a list of strings corresponding to agent descriptions showing only properties that differ within an agent. """
    #split names
    #only compare within types
    change_map = {}#[[False for w in n] for n in attr_dicts]
    types = set([n['agent_type'] for n in attr_dicts])
    type_maps = {}
    for type in types:
        type_attr_dicts = [n for n in attr_dicts if n['agent_type'] == type] #get attr_dicts for current type
        change_map[type] = {key:False for key in type_attr_dicts[0].keys() if key != 'agent_type'} #map of changes—assuming that every agent of a certain task has the same attributes. 
        last = type_attr_dicts[0]
        for type_attr_dict in type_attr_dicts: #for each set of attributes go over and compare with the last
            for key in type_attr_dict.keys():
                if last[key] != type_attr_dict[key]: change_map[type][key] = True
            last = type_attr_dict
    #construct output list
    out_names = []
    for attr_dict in attr_dicts:
        type = attr_dict['agent_type']
        #generate descriptors
        descriptor = [key + ':' + str(attr_dict[key]) for key,include in change_map[type].items() if include]
        #produce string
        out_names.append(type + ' ' + ' '.join(descriptor)) #always print type
    return out_names

def get_runs(table):
    """Returns a list of tables corresponding to one run."""
    return [table[table['run_ID'] == id] for id in table['run_ID'].unique()]

def final_rows(df):
    """Returns the dataframe only with the final row of each run."""
    global _final_row_dict
    df_hash = hash(df.values.tobytes())
    try:
        return _final_row_dict[df_hash]
    except KeyError:
        cache_final_rows(df)
        return _final_row_dict[df_hash]

def compute_final_rows(df):
    """Returns the dataframe only with the final row of each run."""
    rows = []
    for run_ID in df['run_ID'].unique():
        rows.append(df[(df['run_ID'] == run_ID)].tail(1))
    if rows != []: return pd.concat(rows) 
    
def cache_final_rows(df):
    """Calculating the final rows is expensive, so let's save the dataframe with the final rows in a dictionary to only compute it once.
    """
    global _final_row_dict
    df_hash = hash(df.values.tobytes())
    _final_row_dict[hash(df.values.tobytes())] = compute_final_rows(df)

def mean_win(table):
    """Proportion of wins, aka perfect & stable reconstructions"""
    table = final_rows(table)
    count = len(table[table['world_status'] == 'Win'])
    total = len(table)
    try:
        return count/total
    except ZeroDivisionError:
        return 0

def mean_failure_reason(table,reason):
    """Proportion of fails for a certain reason ("Full","Unstable", for fast fail:"Holes","Outside") over all runs (incl wins)"""
    table = final_rows(table)
    try:
        count = len(table[table['world_failure_reason'] == reason])
    except KeyError:
        count = 0
    total = len(table)
    try:
        return count/total
    except ZeroDivisionError:
        return 0

def avg_steps_to_end(table):
    """Returns average steps until the end of the run. Only pass wins if we want a measure of success."""
    table = final_rows(table)
    if len(table) == 0: return 0,0 #if we get an empty table
    results = list(table['step'])
    try:
        return statistics.mean(results),statistics.stdev(results)
    except statistics.StatisticsError:
        return 0,0

def mean_score(table,scoring_function):
    """Returns the mean and standard deviation of the chosen score at the end of a run. Pass a scoring function like bw.F1score"""
    table = final_rows(table)
    scores = []
    for i,row in table.iterrows():
        world = row['_world']
        bm = row['blockmap']
        state = State(world,bm) #create the dummy state object
        state.blockmap = bm
        score = scoring_function(state)
        scores.append(score)
    try:
        return statistics.mean(scores),statistics.stdev(scores)
    except statistics.StatisticsError:
        return 0,0


def mean_peak_score(table,scoring_function):
    """Returns the mean and standard deviation of the chosen score at the peak of F1 score for a run. 
    Pass a scoring function like bw.F1score"""
    table_final_rows = final_rows(table)
    scores = []
    for i,row in table_final_rows.iterrows():
        #get index peak F1 score
        F1s = get_scores(table[table['run_ID'] == row['run_ID']],bw.F1score)
        peak_index = F1s.index(max(F1s))
        #get the last blockmap
        bm = row['blockmap'] * (row['blockmap'] <= peak_index + 1) #its cheaper to recreate the blockmap than to find it in the df
        # bm = table[(table['run_ID'] == row['run_ID']) & (table['step'] == peak_index)]['blockmap'].tail(1).item()
        world = row['_world']
        state = State(world,bm) #create the dummy state object
        state.blockmap = bm
        score = scoring_function(state)
        scores.append(score)
    try:
        return statistics.mean(scores),statistics.stdev(scores)
    except statistics.StatisticsError:
        return 0,0    

def mean_avg_area_under_curve(table,scoring_function):
    table_final_rows = final_rows(table)
    scores = []
    for i,row in table_final_rows.iterrows():
        world = row['_world']
        bm = row['blockmap']
        state = State(world,bm) #create the dummy state object
        state.blockmap = bm
        score = avg_area_under_curve_score(table[table['run_ID'] == row['run_ID']],scoring_function)
        scores.append(score)
    try:
        return statistics.mean(scores),statistics.stdev(scores)
    except statistics.StatisticsError:
        return 0,0

def mean_avg_area_under_curve_to_peakF1(table,scoring_function):
    table_final_rows = final_rows(table)
    scores = []
    for i,row in table_final_rows.iterrows():
        world = row['_world']
        bm = row['blockmap']
        state = State(world,bm) #create the dummy state object
        state.blockmap = bm
        run = run_to_peakF1(table[table['run_ID'] == row['run_ID']])
        score = avg_area_under_curve_score(run,scoring_function)
        scores.append(score)
    try:
        return statistics.mean(scores),statistics.stdev(scores)
    except statistics.StatisticsError:
        return 0,0

#helper functions

class RunError(Exception):
    pass

def assert_run(table):
    """Throws an error if the passed table is not a subset corresponding to a run"""
    if len(table['run_ID'].unique()) != 1:
        raise RunError("{} run_IDs present".format(len(table['run_ID'].unique())))

def get_scores(table,scoring_function):
    """Returns the sequence of chosen scores for a part of the table as a list."""
    assert_run(table)
    blockmaps = table['blockmap']
    world = table['_world'].tail(1).item()
    scores = []
    for bm in blockmaps:
        state = State(world,bm)
        state.blockmap = bm
        score = scoring_function(state)
        scores.append(score)
    return scores    

def area_under_curve_score(table,scoring_function):
    """Takes a run and produces the total area under the curve until the end of the run.
    mean_area_under_curve_score is probably more informative."""
    assert_run(table)
    scores = get_scores(table,scoring_function)
    return np.trapz(scores) #integrate using trapezoidal rule

def avg_area_under_curve_score(table,scoring_function):
    """Takes a run and produces the area under the curve per step taken until the end of the run."""
    assert_run(table)
    scores = get_scores(table,scoring_function)
    return np.trapz(scores)/len(scores) #integrate using trapezoidal rule

def run_to_peakF1(table):
    """Returns a run truncated from start to the point with the highest F1 score.
    Corresponds to the point where the agent was doing best."""
    assert_run(table)
    F1s = get_scores(table,bw.F1score)
    peak_index = F1s.index(max(F1s))+1 #+1 to include the row itself
    return table.iloc[0:peak_index]

def number_of_steps(table):
    """Gets the number of steps taken."""
    return max(table['step'])

def get_final_status(table):
    """Takes run as input and returns a tuple of final state and reason for failure.
    None if it hasn't failed."""
    assert_run(table)
    #NaN == NaN returns false
    status = table.tail(1)['world_status']
    reason = table.tail(1)['world_failure_reason']
    return status, reason

def get_final_blocks(table):
    assert_run(table)
    return table.tail(1)['_blocks'].item()

def get_blockmaps(table):
    """Takes a run as input and returns the sequence of blockmaps.
    This produces one blockmap per action taken, even if the act function has only been called once
    (as MCTS does)."""
    assert_run(table)
    return list(table['blockmap'])

def get_final_blockmap(table):
    """Takes a run as input and returns the final blockmap."""
    assert_run(table)
    return table.tail(1)['blockmap']

def touching_last_block_placements(blocks):
    """Takes in a sequence of block objects and returns an array with true if a block was placed touching the last placed block. Nothing is returned for the first block"""
    local_placements = []
    for i in range(1,len(blocks)):
        if blocks[i].touching(blocks[i-1]):
            local_placements.append(True)
        else:
            local_placements.append(False)
    return local_placements

def touching_last_block_score(table):
    """Takes in dataframe and returns mean between 1 and 0 and STD for touching_last_block_placements."""
    scores = []
    table = final_rows(table)
    for run in get_runs(table):
        seq = touching_last_block_placements(get_final_blocks(run))
        mean = statistics.mean(seq)
        scores.append(mean)
    return statistics.mean(scores), statistics.stdev(scores)

def raw_euclidean_distance_between_blocks(blocks1,blocks2=None):
    """Returns the euclidean distance between the two sequence of blocks.

    >For any pair of action sequences, we define the “raw action dissimilarity” as the mean Euclidean distance between corresponding pairs of [x, y, w, h] action vectors (Fig. 4A, light). When two sequences are of different lengths, we evaluate this metric over the first k actions in both, where k represents the length of the shorter sequence.
    """
    if blocks2 is None:
        #special case for parallelization allows passing a tuple
        blocks2 = blocks1[1]
        blocks1 = blocks1[0]
    distances_sum = 0
    for i in range(min(len(blocks1),len(blocks2))):
        b1 = blocks1[i]
        b2 = blocks2[i]
        distance = math.sqrt((b1.x-b2.x)**2+(b1.y-b2.y)**2+(b1.width-b2.width)**2+(b1.height-b2.height)**2)
        distances_sum += distance
    return distances_sum

def pairwise_raw_euclidean_distance_between_blocks_across_all_runs(table):
    """Calculates the average euclidean distance between runs—returns mean and standard deviation. Note that this comparision really only makes sense for a particular world.

    ⚠️ Scales exponentially! ⚠️

    >For any pair of action sequences, we define the “raw action dissimilarity” as the mean Euclidean distance between corresponding pairs of [x, y, w, h] action vectors (Fig. 4A, light). When two sequences are of different lengths, we evaluate this metric over the first k actions in both, where k represents the length of the shorter sequence.
    """
    # if there are no relevant runs
    if len(table) == 0: return []
    table = final_rows(table) 
    #get list of final blocks
    all_blocks = list(table['_blocks'])
    #get all possible combinations
    pairs = itertools.combinations(all_blocks,2)
    #calculate distances
    distances = [raw_euclidean_distance_between_blocks(b1,b2) for b1,b2 in pairs]
    return distances