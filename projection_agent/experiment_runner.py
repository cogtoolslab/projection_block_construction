import pandas as pd
import copy
import datetime
import traceback
import psutil
import time
import random
import multiprocessing
import tqdm

def run_experiment(worlds,agents,per_exp=10,steps=100,verbose=False,save=True,parallelized=True):
    """Runs x experiments on the given worlds with the given agents for up to 100 steps while keeping logging values to a dataframe. Pass blockworlds & agents as named dictionary for readability of result. The world is assigned to the agent later, so it makes sense to pass none. You can pass negative numbers steps to run until the agent is finished. Pass a float to parallelized to set the fraction of CPUs tp use."""
    #we want human readable labels for the dataframe
    if type(worlds) is dict:
        world_labels = [label+'|'+w.__str__() for label,w in worlds.items()]
        worlds = [w for w in worlds.values()]
    else:
        world_labels = [w.__str__() for w in worlds]
    if type(agents) is dict:
        agent_labels = [label+'|'+a.value().__str__() for label,a in items]
        agents = [a for a in agents.values()]
    else:
        agent_labels = [a.__str__() for a in agents]

    #we need to copy the world and agent to reset them
    # create a list of experiments to run
    experiments = [(copy.deepcopy(w),copy.deepcopy(a),steps,verbose) for i in range(per_exp) for a in agents for w in worlds]    
    random.shuffle(experiments)
    labels = [(w,a) for i in range(per_exp) for a in agent_labels for w in world_labels]
    # lets run the experiments
    P = multiprocessing.Pool(int(multiprocessing.cpu_count()*parallelized),maxtasksperchild=1) #restart process after a single task is performed—slow for short runs, but fixes memory leak (hopefully)
    results_mapped = tqdm.tqdm(P.imap_unordered(_run_single_experiment,experiments), total=len(experiments))
    # results_mapped = P.imap_unordered(_run_single_experiment,experiments)
    P.close()
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
    # to prevent memory overflows only run if enough free memory exists.
    start_time = time.time()
    world,agent,steps,verbose = experiment
    while psutil.virtual_memory().percent > 75:
        print("Delaying running",agent.__str__(),'******',world.__str__(),"because of RAM usage. Trying again in 120 seconds.")
        time.sleep(120)
    
    print('Running',agent.__str__(),'******',world.__str__())
    agent.set_world(world)
    r = pd.DataFrame(columns=['blockmap','blocks','stability','F1 score','chosen action','final result','final result reason'], index=range(steps+1))
    i = 0
    while i != steps and world.status()[0] == 'Ongoing':
        r.iloc[i]['blockmap'] = [world.current_state.blockmap]
        r.iloc[i]['blocks'] = [world.current_state.blocks]
        r.iloc[i]['stability'] = world.stability()
        r.iloc[i]['F1 score'] = world.F1score()
        status,reason = world.status()
        r.iloc[i][ 'final result'] = status
        r.iloc[i][ 'final result reason'] = reason
        try:
            chosen_action = agent.act(verbose=verbose)
        except SystemError as e:
            print("Error while acting:",e)
            print(traceback.format_exc())
            
        r.iloc[i]['chosen action'] = chosen_action
        i = i + 1
    #after we stop acting
    print("Done with",agent.__str__(),'******',world.__str__(),"in %s seconds" % round((time.time() - start_time)))
    r.iloc[i]['blockmap'] = [world.current_state.blockmap]
    r.iloc[i]['blocks'] = [world.current_state.blocks]
    r.iloc[i]['stability'] = world.stability()
    r.iloc[i]['F1 score'] = world.F1score()
    status,reason = world.status()
    r.iloc[i][ 'final result'] = status
    r.iloc[i][ 'final result reason'] = reason
    return copy.deepcopy(r),copy.copy(status)