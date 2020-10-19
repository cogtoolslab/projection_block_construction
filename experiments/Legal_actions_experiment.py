if __name__=="__main__": #required for multiprocessing
    import os
    import sys
    proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    utils_dir = os.path.join(proj_dir,'utils')
    sys.path.append(utils_dir)
    agent_dir = os.path.join(proj_dir,'model')
    sys.path.append(agent_dir)
    agent_util_dir = os.path.join(agent_dir,'utils')
    sys.path.append(agent_util_dir)

    from BFS_Agent import BFS_Agent
    import Construction_Paper_Agent as CPA
    import blockworld as bw
    import random
    import blockworld_library as bl
    import experiment_runner

    import time
    start_time = time.time()

    #16 for nightingale, 96 for google cloud
    fraction_of_cpus = 1

    agents = [
        # BFS_Agent(horizon=1,scoring_function=bw.random_scoring,scoring='Average'),
        # BFS_Agent(horizon=1,scoring_function=bw.F1score,scoring='Average'),
        BFS_Agent(horizon=2,scoring_function=bw.F1score,scoring='Average'),
        # BFS_Agent(horizon=3,scoring_function=bw.F1score,scoring='Average'),
        # BFS_Agent(horizon=4,scoring_function=bw.F1score,scoring='Average'),
        # BFS_Agent(horizon=5,scoring_function=bw.F1score,scoring='Average'),
        CPA.Construction_Paper_Agent(),
        CPA.Construction_Paper_Agent(decomposition_function=CPA.fixed_2)
        CPA.Construction_Paper_Agent(decomposition_function=CPA.random_1_4)
        ]

    silhouette8 = [14,11,3,13,12,1,15,5]
    silhouettes = {i : bl.load_interesting_structure(i) for i in silhouette8}
    worlds_silhouettes_legal = {'int_struct_legal_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default,legal_action_space=True) for i,s in silhouettes.items()}
    worlds_silhouettes_all = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default,legal_action_space=False) for i,s in silhouettes.items()}
    worlds_small = {
        'stonehenge_6_4' : bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
        'stonehenge_3_3' : bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
        # 'block' : bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
        # 'T' : bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
        # 'side_by_side' : bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
    }
    worlds = {**worlds_silhouettes_legal,**worlds_silhouettes_all,**worlds_small}

    results = experiment_runner.run_experiment(worlds,agents,100,40,verbose=False,parallelized=fraction_of_cpus,save='legal_actions_experiment')
    print(results[['agent','world','world_status']])

    print("Done in %s seconds" % (time.time() - start_time))