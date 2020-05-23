import pandas as pd
from p_tqdm import p_map # using this for a progress bar with multiprocessing. Can be replaced with map or tqdm.
import copy
import datetime

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def run_experiment(worlds,agents,per_exp=10,steps=100,verbose=False,save=True):
    """Runs x experiments on the given worlds with the given agents for up to 100 steps while keeping logging values to a dataframe. Pass blockworlds & agents as named dictionary for readibility of result. The world is assigned to the agent later, so it makes sense to pass none. You can pass negative numbers steps to run until the agent is finished."""
    #TODO add dictionary for datakeeping
    #we need to copy the world and agent to reset them
    # create a list of experiments to run
    experiments = [(copy.deepcopy(w),copy.deepcopy(a),steps,verbose) for i in range(per_exp) for a in agents for w in worlds] 
    #TODO: automatically create labels if a dictionary isn't passed
    
    # lets run the experiments
    #parallelized:
    results_mapped = p_map(_run_single_experiment,experiments)
    results = pd.DataFrame(columns=['agent','world','outcome','run'],
    index=range(len(experiments)))
    #put the experiments into a dataframe
    for i,rm in enumerate(results_mapped):
        run,final_status = rm

        results.iloc[i]['world'] = experiments[i][0]
        results.iloc[i]['agent'] =experiments[i][1]
        results.iloc[i]['outcome'] = final_status
        results.iloc[i]['run'] = run

    #non-parallelized—for history:
    # results = pd.DataFrame(columns=['agent','world','outcome','run'],
    # index=range(len(experiments)))
    # for i,exp in enumerate(experiments):
    #     print("Running experiment",i+1,"of",len(experiments))
    #     run,final_status = _run_single_experiment(exp)
    #     results.iloc[i]['world'] = exp[0]
    #     results.iloc[i]['agent'] =exp[1]
    #     results.iloc[i]['outcome'] = final_status
    #     results.iloc[i]['run'] = run

    if save:
        #save the results to a file.
        results.to_pickle("Experiment "+str(datetime.datetime.today())+".pkl")
    return results

def _run_single_experiment(experiment):
    world,agent,steps,verbose = experiment
    agent.set_world(world)
    r = pd.DataFrame(columns=['blockmap','blocks','stability','F1 score','chosen action','final result'], index=range(steps+1))
    i = 0
    while i != steps and world.status() == 'Ongoing':
        r.iloc[i]['blockmap'] = [world.current_state.block_map]
        r.iloc[i]['blocks'] = [world.current_state.blocks]
        r.iloc[i]['stability'] = world.stability()
        r.iloc[i]['F1 score'] = world.F1score()
        r.iloc[i][ 'final result'] = world.status()
        chosen_action = agent.act(verbose=verbose)
        r.iloc[i]['chosen action'] = chosen_action
        i = i + 1
    #after we stop acting
    r.iloc[i]['blockmap'] = [world.current_state.block_map]
    r.iloc[i]['blocks'] = [world.current_state.blocks]
    r.iloc[i]['stability'] = world.stability()
    r.iloc[i]['F1 score'] = world.F1score()
    final_status = world.status()
    r.iloc[i][ 'final result'] = final_status
    return r,final_status