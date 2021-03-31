if __name__=="__main__": #required for multiprocessing
    import os
    import sys
    proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(proj_dir)
    utils_dir = os.path.join(proj_dir,'utils')
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir,'model')
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir,'utils')
    sys.path.append(agent_util_dir)

    from model.BFS_Agent import BFS_Agent
    import utils.blockworld as bw
    import utils.blockworld_library as bl
    import experiments.experiment_runner as experiment_runner

    import time
    start_time = time.time()

    #16 for nightingale, 96 for google cloud
    fraction_of_cpus = .5

    agents = [
       BFS_Agent()
        ]

    silhouettes = {i : bl.load_interesting_structure(i) for i in bl.SILHOUETTE8}
    worlds = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default,legal_action_space=True) for i,s in silhouettes.items()}

    results = experiment_runner.run_experiment(worlds,agents,1,1,verbose=False,parallelized=fraction_of_cpus,save='BFS_search_exhaustive',maxtasksperprocess = 1)
    print(results[['agent','world','world_status']])

    print("Done in %s seconds" % (time.time() - start_time))