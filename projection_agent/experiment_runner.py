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
    """Runs x experiments on the given worlds with the given agents for up to 100 steps while keeping logging values to a dataframe. Pass blockworlds as named dictionary for readability of results. Pass agents as a list: the __str__ function of an agent will take care of it. The world is assigned to the agent later, so it makes sense to pass none. You can pass negative numbers steps to run until the agent is finished. Pass a float to parallelized to set the fraction of CPUs to use."""
    #we want human readable labels for the dataframe
    if type(worlds) is not dict:
        #if worlds is list create dictionary
        worlds = {w.__str__():w for w in worlds}
    if type(agents) is dict:
        #if agents is dict flatten it and rely on informative agent __str__
        agents = [a for a in agents.values()]

    #we need to copy the world and agent to reset them
    # create a list of experiments to run
    experiments = [(copy.deepcopy(w),copy.deepcopy(a),steps,verbose) for i in range(per_exp) for a in agents for w in worlds.items()]    
    # lets run the experiments
    P = multiprocessing.Pool(int(multiprocessing.cpu_count()*parallelized),maxtasksperchild=1) #restart process after a single task is performed—slow for short runs, but fixes memory leak (hopefully)
    results_mapped = tqdm.tqdm(P.imap(_run_single_experiment,experiments), total=len(experiments))
    P.close()
    results = pd.DataFrame(columns=['agent','world','outcome','run'],index=range(len(experiments)))
    #put the experiments into a dataframe
    for i,rm in enumerate(results_mapped):
        run,final_status,labels = rm
        results.iloc[i]['world'] = labels[0]
        results.iloc[i]['agent'] =labels[1]
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
    world_dict,agent,steps,verbose = experiment
    world_label = world_dict[0]
    world = world_dict[1]
    # while psutil.virtual_memory().percent > 65:
    #     print("Delaying running",agent.__str__(),'******',world.__str__(),"because of RAM usage. Trying again in 1000 seconds.")
    #     time.sleep(1000)
    
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
    return copy.deepcopy(r),copy.copy(status),(world_label,agent.__str__())