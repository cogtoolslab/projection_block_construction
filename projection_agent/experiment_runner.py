import pandas as pd
from p_tqdm import p_map # using this for a progress bar with multiprocessing. Can be replaced with map (plus evaluation) or tqdm.
from tqdm import tqdm
import copy
import datetime

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def run_experiment(worlds,agents,per_exp=10,steps=100,verbose=False,save=True,parallelized=True):
    """Runs x experiments on the given worlds with the given agents for up to 100 steps while keeping logging values to a dataframe. Pass blockworlds & agents as named dictionary for readibility of result. The world is assigned to the agent later, so it makes sense to pass none. You can pass negative numbers steps to run until the agent is finished. Pass a float to parallized to set the fraction of CPUs tp use."""
    #we want human readable labels for the dataframe
    if type(worlds) is dict:
        world_labels = [w.label()+'|'+w.value().__str__() for w in worlds]
        worlds = [w.value() for w in worlds]
    else:
        world_labels = [w.__str__() for w in worlds]
    if type(agents) is dict:
        agent_labels = [a.label()+'|'+a.value().__str__() for a in agents]
        agents = [a.value() for a in agents]
    else:
        agent_labels = [a.__str__() for a in agents]

    #we need to copy the world and agent to reset them
    # create a list of experiments to run
    experiments = [(copy.deepcopy(w),copy.deepcopy(a),steps,verbose) for i in range(per_exp) for a in agents for w in worlds]    
    labels = [(w,a) for i in range(per_exp) for a in agent_labels for w in world_labels]
    # lets run the experiments
    if parallelized:
        results_mapped = p_map(_run_single_experiment,experiments,num_cpus=float(parallelized)) #by default we use all CPUs
    else:
        results_mapped = map(_run_single_experiment,tqdm(experiments)) #non-parallelized
    results = pd.DataFrame(columns=['agent','world','outcome','run'],index=range(len(experiments)))
    #put the experiments into a dataframe
    for i,rm in enumerate(results_mapped):
        run,final_status = rm
        results.iloc[i]['world'] = labels[i][0]
        results.iloc[i]['agent'] =labels[i][1]
        results.iloc[i]['outcome'] = final_status
        results.iloc[i]['run'] = run #we save the entire dataframe into a cell

    if save is not False:
        #save the results to a file.
        if type(save) is str:
            results.to_pickle(save+".pkl")
        else:
            results.to_pickle("Experiment "+str(datetime.datetime.today())+".pkl")

    return results

def _run_single_experiment(experiment):
    world,agent,steps,verbose = experiment
    print('Runnning',agent.__str__(),'******',world.__str__())
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