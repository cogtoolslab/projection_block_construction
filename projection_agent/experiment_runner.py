import pandas as pd
import numpy as np
import copy
import datetime
import traceback
import psutil
import time
import random
import multiprocessing
import tqdm

"""
# Meaning of columns in dataframe:
A row corresponds to an individual action taken, ie. a single block placed.
## every action
'run_ID': unique ID for that run (very long)
'agent': type of agent as string
'world': name of world as string
'step': index of action taken. Starts at 0
'planning_step': index of planning step. Starts at 0
'states_evaluated': in the process of planning, how often was a state evaluated? A proxy for how expensive planning is
'action': action as human readable string (['(width x height)', 'x location'])
'_action': action as object (for analysis)
'action_x': x coordinate of block placed
'action_block_width': width of block placed
'action_block_height: height of block placed
'blocks': blocks in world as human readable string ("width, height at x,y")
'_blocks': blocks as objects (for analysis)
'blockmap': bitmap of the world at the current state
various agent parameters (depends on agent loaded)
## only at planning step
'_world_cur_state': current state of the world as object (for analysis)
'_blocks': current list of blocks in the world as object (for analysis)
'execution_time': computation time of the planning step in seconds
'world_status': either fail, ongoing, winning
'world_failure_reason': if world_status is fail, then the reason is given here
'agent_attributes': the attributes of the agent as dictionary. Allow for easy grouping of agents across runs. Does not include random seed.
**agent attributes unrolled** as provided by the class of agent. Includes random_seed.
"""

DF_COLS = ['run_ID','agent','world','step','planning_step','states_evaluated','action','_action','action_x','action_block_width','action_block_height','blocks','_blocks','blockmap','_world_cur_state','execution_time','world_status','world_failure_reason','agent_attributes']
RAM_LIMIT = 100 # percentage of RAM usage over which a process doesn't run as to not run out of memory

def run_experiment(worlds,agents,per_exp=100,steps=40,verbose=False,save=True,parallelized=True):
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
    experiments = [(copy.deepcopy(w),copy.deepcopy(a),steps,verbose,i) for i in range(per_exp) for a in agents for w in worlds.items()]    
    # lets run the experiments
    P = multiprocessing.Pool(int(multiprocessing.cpu_count()*parallelized),maxtasksperchild=1) #restart process after a single task is performed—slow for short runs, but fixes memory leak (hopefully)
    results_mapped = tqdm.tqdm(P.imap_unordered(_run_single_experiment,experiments), total=len(experiments))
    # results_mapped = P.map(_run_single_experiment,experiments), total=len(experiments)
    # results_mapped =list(map(_run_single_experiment,experiments))
    P.close()

    results = pd.concat(results_mapped).reset_index(drop = True)

    if save is not False:
        #save the results to a file.
        if type(save) is str:
            results.to_pickle(save+".pkl")
        else:
            results.to_pickle("Experiment "+str(datetime.datetime.today())+".pkl")

    return results

def _run_single_experiment(experiment):
    """Runs a single experiment. Returns complete dataframe with an entry for each action."""
    # to prevent memory overflows only run if enough free memory exists.
    world_dict,agent,steps,verbose,run_nr = experiment
    world_label = world_dict[0]
    world = world_dict[1]
    run_ID = world_label+agent.__str__()+str(run_nr)+'|'+str(random.randint(0,9999)) #unique string representing the run
    while psutil.virtual_memory().percent > RAM_LIMIT:
        print("Delaying running",agent.__str__(),'******',world.__str__(),"because of RAM usage. Trying again in 1000 seconds.")
        time.sleep(1000)
    
    print('Running',agent.__str__(),'******',world.__str__())
    agent_parameters = agent.get_parameters()
    agent_parameters_w_o_random_seed = {key:value for key,value in agent_parameters.items() if key != 'random_seed'}    
    agent.set_world(world)
    #create dataframe
    r = pd.DataFrame(columns=DF_COLS+list(agent_parameters.keys()), index=range(steps+1))
    i = 0 #the ith action taken
    planning_step = 0
    while i != steps and world.status()[0] == 'Ongoing':
        #execute the action
        try:
            start_time = time.perf_counter()
            chosen_actions,number_of_states_evaluated = agent.act(verbose=verbose)
            duration = time.perf_counter() - start_time 
            planning_step += 1
        except SystemError as e:
            print("Error while acting:",e)
            print(traceback.format_exc())
        #unroll the chosen actions to get step-by-step entries in the dataframe
        planning_step_blockmaps = get_blockmaps(world.current_state.blockmap) # the blockmap for every step
        for ai,action in enumerate(chosen_actions):
            if ai > steps: 
                Warning("Number of actions ({}) exceeding steps ({}).".format(ai,steps))
                break
            # adding the agent parameters
            r.iloc[i] = agent_parameters
            r.iloc[i]['run_ID'] = run_ID
            r.iloc[i]['agent'] = agent_parameters['agent_type']
            r.iloc[i]['agent_attributes'] = agent_parameters_w_o_random_seed
            r.iloc[i]['world'] = world_label
            r.iloc[i]['step'] = i
            r.iloc[i]['planning_step'] = planning_step
            r.iloc[i]['states_evaluated'] = number_of_states_evaluated
            r.iloc[i]['action'] = [str(e) for e in action] #human readable action
            r.iloc[i]['_action'] = action #action as object
            r.iloc[i]['action_x'] = action[1]
            r.iloc[i]['action_block_width'] = action[0].width
            r.iloc[i]['action_block_height'] = action[0].height
            r.iloc[i]['blocks'] = [block.__str__() for block in world.current_state.blocks[:i+1]]  #human readable blocks
            r.iloc[i]['_blocks'] = world.current_state.blocks[:i+1]
            r.iloc[i]['blockmap'] = planning_step_blockmaps[i]
            i += 1 
        #the following are only filled for each planning step, not action step
        r.iloc[i-1]['execution_time'] = duration
        r.iloc[i-1]['_world_cur_state'] = world.current_state
        world_status = world.status()
        r.iloc[i-1]['world_status'] = world_status[0] 
        r.iloc[i-1]['world_failure_reason'] = world_status[1]

    #after we stop acting
    # add info one last time
    print("Done with",agent.__str__(),'******',world.__str__(),"in %s seconds with outcome "% round((time.perf_counter() - start_time)),str(world_status))
    #truncate df and return
    return  r[r['run_ID'].notnull()]

def get_blockmaps(blockmap):
    """Takes a blockmao as input and returns the sequence of blockmaps leading up to it.
    This produces one blockmap per action taken, even if the act function has only been called once
    (as MCTS does)."""
    blockmaps = []
    #maybe it's wrapped in a list
    if len(blockmap) == 1: blockmap = blockmap[0]
    blockmap = np.array(blockmap)
    #generate sequence of blockmaps
    for i in range(np.max(blockmap)+1): #for every placed block
        bm = blockmap * (blockmap <= i+1)
        blockmaps.append(bm)
    return blockmaps