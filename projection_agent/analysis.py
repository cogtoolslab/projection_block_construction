#this file contains helper functions to generate plots and scalar measures.

#setup
import blockworld as bw
import blockworld_library as bl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics

#########################################
#run experiments to debug functions—this should live in another file

#specific setups
#initializing worlds (used for scoring re a certain silhuoette)
silhouettes = {i : bl.load_interesting_structure(i) for i in [14,15,5,8,12,1]}
worlds_silhouettes = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for i,s in silhouettes.items()}
worlds_small = {
    'stonehenge_6_4' : bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
    'stonehenge_3_3' : bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
    'block' : bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
    'T' : bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
    'side_by_side' : bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
}
bw_worlds = {**worlds_silhouettes,**worlds_small}

dfs = ['beam_search.pkl'] #which to load

#load all experiments as one dataframe
df = pd.concat([pd.read_pickle(l) for l in dfs])
r = df.iloc[1]['run'] #for testing purposes load a single run

print("Done with setup, loaded",len(df),"lines")
# print(df)

#Debug functions

##########################################



class State():
    """A dummy state to pass to blockworld scoring functions"""
    def __init__(self,world,blockmap):
        self.block_map = blockmap
        self.world = world


# Scalar measures

def mean_win(table):
    """Proportion of wins, aka perfect & stable reconstructions"""
    wins = 0
    total = 0
    for row in table['outcome']:
        if row == 'Win':
            wins+=1
        total += 1
    return wins/total

def mean_failure_reason(table,reason):
    """Proportion of fails for a certain reason ("Full","Unstable", for fast fail:"Holes","Outside") over all runs (incl wins)"""
    count = 0
    total = 0
    for run in table['run']:
        run_status, run_reason = get_final_status(run)
        if run_status == 'Fail' and run_reason == reason:
            count+=1
        total += 1
    return count/total

def avg_steps_to_end(table):
    """Returns average steps until the end of the run. Only pass wins if we want a measure of success."""
    results = [number_of_steps(r) for r in table['run']]
    return statistics.mean(results), statistics.stdev(results)

def mean_score(table,scoring_function):
    """Returns the mean and standard deviation of the chosen score at the end of a run. Pass a scoring function like bw.F1score"""
    # get unique worlds
    unique_worlds = table['world'].unique()
    scores = []
    for world in unique_worlds:
        world_obj = bw_worlds[world.split('|')[0]] #get the instantiated world object
        for run in df[df['world'] == world]['run']:
            #get the last blockmap
            blockmap = get_final_blockmap(run)
            #create state
            state = State(world_obj,blockmap)
            #calculate score using scoring function
            score = scoring_function(state)
            scores.append(score)
    return statistics.mean(scores),statistics.stdev(scores)    

def mean_peak_score(table,scoring_function):
    """Returns the mean and standard deviation of the chosen score at the peak of F1 score for a run. 
    Pass a scoring function like bw.F1score
    Useful for """
    # get unique worlds
    unique_worlds = table['world'].unique()
    scores = []
    for world in unique_worlds:
        world_obj = bw_worlds[world.split('|')[0]] #get the instantiated world object
        for run in df[df['world'] == world]['run']:
            #get index peak F1 score
            F1s = get_scores(run,bw.F1score,world_obj)
            peak_index = F1s.index(max(F1s))
            #get the last blockmap
            blockmap = get_blockmaps(run)[peak_index]
            #create state
            state = State(world_obj,blockmap)
            #calculate score using scoring function
            score = scoring_function(state)
            scores.append(score)
    return statistics.mean(scores),statistics.stdev(scores)    

def mean_avg_area_under_curve(table,scoring_function):
    # get unique worlds
    unique_worlds = table['world'].unique()
    scores = []
    for world in unique_worlds:
        world_obj = bw_worlds[world.split('|')[0]] #get the instantiated world object
        for run in df[df['world'] == world]['run']:
            scores.append(avg_area_under_curve_score(run,scoring_function,world_obj))
    return statistics.mean(scores),statistics.stdev(scores)

def mean_avg_area_under_curve_to_peakF1(table,scoring_function):
    # get unique worlds
    unique_worlds = table['world'].unique()
    scores = []
    for world in unique_worlds:
        print(world)
        world_obj = bw_worlds[world.split('|')[0]] #get the instantiated world object
        for run in df[df['world'] == world]['run']:
            #truncate run
            run = run_to_peakF1(run,world_obj)
            scores.append(avg_area_under_curve_score(run,scoring_function,world_obj))
    return statistics.mean(scores),statistics.stdev(scores)

#helper functions

def get_scores(run,scoring_function,world_obj):
    """Returns the sequence of chosen scores for a run as a list. Requires passing an instantiated world object."""
    blockmaps = get_blockmaps(run)
    scores = [scoring_function(State(world_obj,bm)) for bm in blockmaps]
    return scores    

def area_under_curve_score(run,scoring_function,world_obj):
    """Takes a run and produces the total area under the curve until the end of the run.
    mean_area_under_curve_score is probably more informative."""
    scores = get_scores(run,scoring_function,world_obj)
    return np.trapz(scores) #integrate using trapezoidal rule

def avg_area_under_curve_score(run,scoring_function,world_obj):
    """Takes a run and produces the area under the curve per step taken until the end of the run."""
    scores = get_scores(run,scoring_function,world_obj)
    return np.trapz(scores)/len(scores) #integrate using trapezoidal rule

def run_to_peakF1(run,world_obj):
    """Returns a run truncated from start to the point with the highest F1 score.
    Corresponds to the point where the agent was doing best."""
    F1s = get_scores(run,bw.F1score,world_obj)
    peak_index = F1s.index(max(F1s))+1 #+1 to inlcude the row itself
    return run.iloc[0:peak_index]

def number_of_steps(run):
    """Gets the number of steps taken."""
    final_bm = run[run['final result'].notnull()]['blockmap'].iloc[-1] #grab final bm
    final_bm = np.array(final_bm)
    return np.max(final_bm) #return counter of highest block placed in blockmap

def get_final_status(run):
    """Takes run as input and returns a touple of final state and reason for failure.
    None if it hasn't failed."""
    #NaN == NaN returns false
    status = run[run['final result'] == run['final result']].iloc[-1]['final result']
    reason = run[run['final result'] == run['final result']].iloc[-1]['final result reason']
    return status, reason

def get_blockmaps(run):
    """Takes a run as input and returns the sequence of blockmaps.
    This produces one blockmap per action taken, even if the act function has only been called once
    (as MCTS does)."""
    blockmaps = []
    final_bm = run[run['final result'].notnull()]['blockmap'].iloc[-1] #grab final bm
    final_bm = np.array(final_bm)
    #generate sequence of blockmaps
    for i in range(np.max(final_bm)+1): #for every placed block
        bm = final_bm * (final_bm <= i+1)
        blockmaps.append(bm)
        print(bm)
    return blockmaps

def get_final_blockmap(run):
    """Takes a run as input and returns the final blockmap."""
#     final_bm = run[run['final result'].notnull()]['blockmap'].iloc[-1] #grab final bm
#     final_bm = np.array(final_bm)
    return get_blockmaps(run)[-1]
#     return final_bm

