# set up directories
from analysis.utils.analysis_helper import preprocess_df
import os
import sys
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
results_dir = os.path.join(proj_dir,'results')
df_dir = os.path.join(results_dir,'dataframes')

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
from experiments.experiment_runner import get_blockmaps,agent_para_dict

"""
Takes in a dataframe from `subgoal_generator_runner` and outputs a dataframe with simulated lookahead agents based on the subgoals found.
"""

DF_COLS = ['run_ID','agent','world','step','planning_step','states_evaluated','action','_action','action_x','action_block_width','action_block_height','blocks','_blocks','blockmap','_world','legal_action_space','fast_failure','execution_time','world_status','world_failure_reason','agent_attributes']
RAM_LIMIT = 100 # percentage of RAM usage over which a process doesn't run as to not run out of memory

def run_experiment(parent_df,agents,per_exp=100,steps=40,save=True,parallelized=True,maxtasksperprocess=1,chunk_experiments_size=2048):
    """Takes in a dataframe from `subgoal_generator_runner` and outputs a dataframe with simulated lookahead agents based on the subgoals found."""
    #we want human readable labels for the dataframe
    if type(agents) is dict:
        #if agents is dict flatten it and rely on informative agent __str__
        agents = [a for a in agents.values()]
    experiments = [(copy.deepcopy(a),row,steps,i) for i in range(per_exp) for a in agents for row in parent_df.iterrows()]    
    experiments = [[*exp,j] for j,exp in enumerate(experiments)] #add unique label per experiment
    print("Created",len(experiments),"experiments")
    # if we have many experiments, then the dataframe can get too large for memory. Let's break it down and save them in smaller increments. 
    chunked_experiments = []
    index = 0
    while index <= len(experiments):
        chunked_experiments.append(experiments[index:index+chunk_experiments_size])
        index = index+chunk_experiments_size
    print("Chunked experiments into",len(chunked_experiments),"chunks")
    all_results = []
    for chunk_i,experiments in enumerate(chunked_experiments):
        chunk_i+=1 #since we don't count from 0
        print("Running experiment block",chunk_i,"of",len(chunked_experiments))
        # lets run the experiments
        if parallelized is not False:
            P = multiprocessing.Pool(int(multiprocessing.cpu_count()*parallelized),maxtasksperchild=maxtasksperprocess) #restart process after a number of task is performed—slow for short runs, but fixes memory leak (hopefully)
            # results_mapped = tqdm.tqdm(P.imap_unordered(_run_single_experiment,experiments), total=len(experiments))
            results_mapped = P.map(_run_single_experiment,experiments)
            P.close()
            P.join() #should run the garbage collector and prevent memory leaks
        else:
            results_mapped =list(map(_run_single_experiment,experiments))

        try:
            results = pd.concat(results_mapped).reset_index(drop = True)
        except ValueError:
            #nothing to concatenate
            print("Got empty results")
        else:
            preprocess_df(results) #automatically fill in code relevant to analysis
        all_results.append(results)


        if save is not False:
            #check if results directory exists
            if not os.path.isdir(df_dir):
                os.makedirs(df_dir)
            #save the results to a file.
            if type(save) is str:
                results.to_pickle(os.path.join(df_dir,save+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".pkl"))
                print("Saved to",os.path.join(df_dir,save+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".pkl"))
            else:
                results.to_pickle(os.path.join(df_dir,"Experiment "+str(datetime.datetime.today())+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".pkl"))
                print("Saved to",df_dir,"Experiment "+str(datetime.datetime.today())+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".pkl")

            #lets also save it as a csv without embedded objects
            columns = [col for col in results.columns if col[0] != '_']
            results_wo_objects = results[columns]
            #save the results to a file.
            if type(save) is str:
                results_wo_objects.to_csv(os.path.join(df_dir,save+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".csv"))
                print("Saved to",os.path.join(df_dir,save+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".csv"))
            else:
                results_wo_objects.to_csv(os.path.join(df_dir,"Experiment "+str(datetime.datetime.today())+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".csv"))
                print("Saved to",df_dir,"Experiment "+str(datetime.datetime.today())+'_'+str(chunk_i)+"_of_"+str(len(chunked_experiments))+".csv")
    return pd.concat(all_results).reset_index(drop = True)
    

def _run_single_experiment(experiment):
    """Runs a single experiment. Returns complete dataframe with an entry for each action."""
    # to prevent memory overflows only run if enough free memory exists.
    agent,row,steps,run_nr,unique_nr = experiment
    if type(row) is tuple:
        #unpack row if needed
        row = row[1]
    world_label = row['world']
    #if the agent has no random seed assigned yet, assign one now only for this run
    try:
        if agent.random_seed is None:
            agent.random_seed = random.randint(0,99999)
    except AttributeError:
        pass
    run_ID = world_label+' | '+agent.__str__()+str(run_nr)+' | '+str(random.randint(0,9999))+'|'+str(unique_nr) #unique string representing the run
    while psutil.virtual_memory().percent > RAM_LIMIT:
        print("Delaying running",agent.__str__(),'******',world_label,"because of RAM usage. Trying again in 1000 seconds. RAM usage is "+str(psutil.virtual_memory().percent)+'%')
        time.sleep(1000)
    
    #fill agent
    agent.set_parent_agent(copy.deepcopy(row['_agent']))
    world = agent.world #comes from parent agent
    #reset world to empty
    world.current_state.blocks = []
    world.current_state.clear()
    world.current_state.blockmap = world.current_state._get_new_map_from_blocks([])
    agent.all_sequences = row['_all_sequences']

    # print('Running',agent.__str__(),'******',world.__str__())
    agent_parameters = agent.get_parameters()
    agent_parameters_w_o_random_seed = {key:value for key,value in agent_parameters.items() if  'random_seed' not in key}    
    agent_parameters_w_o_random_seed = agent_para_dict(agent_parameters_w_o_random_seed)
    
    #create dataframe
    r = pd.DataFrame(columns=DF_COLS+list(agent_parameters.keys()), index=range(steps+1))

    #just for logging
    run_start_time = time.perf_counter()
    world_status = "Untouched"

    i = 0 #the ith action taken
    planning_step = 0
    while i != steps and planning_step != steps and world.status()[0] == 'Ongoing':
        #execute the action
        start_time = time.perf_counter()
        try:
            chosen_actions,agent_step_info = agent.act()
            planning_step += 1
        except SystemError as e:
            print("Error while acting:",e)
            print(traceback.format_exc())
        if chosen_actions == []:#If we cannot find any actions, we still record the facts of the run
            chosen_actions = [None] 
        duration = time.perf_counter() - start_time 
        #unroll the chosen actions to get step-by-step entries in the dataframe
        planning_step_blockmaps = get_blockmaps(world.current_state.blockmap) # the blockmap for every step
        for ai,action in enumerate(chosen_actions):
            if ai > steps: 
                Warning("Number of actions ({}) exceeding steps ({}).".format(ai,steps))
                break
            # adding the agent parameters
            for key,value in agent_parameters.items():
                r.at[i,key] = value
            r.at[i,'run_ID'] = run_ID
            r.at[i,'agent'] = agent_parameters['agent_type']
            r.at[i,'agent_attributes'] = str(agent_parameters_w_o_random_seed)
            r.at[i,'world'] = world_label
            r.at[i,'step'] = i
            r.at[i,'planning_step'] = planning_step
            if action is not None:
                r.at[i,'action'] = str([str(e) for e in action]) #human readable action
                r.at[i,'_action'] = action #action as object
                r.at[i,'action_x'] = action[1]
                r.at[i,'action_block_width'] = action[0].width
                r.at[i,'action_block_height'] = action[0].height
            r.at[i,'blocks'] = [block.__str__() for block in world.current_state.blocks[:i+1]]  #human readable blocks
            r.at[i,'_blocks'] = world.current_state.blocks[:i+1]
            r.at[i,'blockmap'] = planning_step_blockmaps[i]
            r.at[i,'_world'] = world
            r.at[i,'legal_action_space'] = world.legal_action_space
            r.at[i,'fast_failure'] = world.fast_failure
            i += 1 
        #the following are only filled for each planning step, not action step
        r.at[i-1,'execution_time'] = duration
        world_status = world.status()
        r.at[i-1,'world_status'] = world_status[0] 
        r.at[i-1,'world_failure_reason'] = world_status[1]
        #if we have it, unroll the miscellaneous output from agent
        #should include `states_evaluated`
        for key,value in agent_step_info.items():
            try:
                r[key]
            except KeyError:
                #we need to create column
                r[key] = np.NaN
            try:
                r.at[i-1,key] = value
            except ValueError: #happens when the datatype of the columns is inferred as numeric
                r[key] = r[key].astype(object)
                r.at[i-1,key] = [value]
        # if we've observed no action being taken, we stop execution. We're not changing the world, so we might as well save the CPU cycles. 
        # Take this out if we have a non-deterministic agent that sometimes chooses no actions.
        if chosen_actions == [] or chosen_actions == [None]: break

    #after we stop acting
    # print("Done with",agent.__str__(),'******',world_label,"in %s seconds with outcome "% round((time.perf_counter() - run_start_time)),str(world_status))
    #truncate df and return
    return  r[r['run_ID'].notnull()]
