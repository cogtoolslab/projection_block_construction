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

    import Construction_Paper_Agent as CPA
    from BFS_Agent import BFS_Agent
    import blockworld as bw
    import random
    import blockworld_library as bl
    import experiments.experiment_runner as experiment_runner

    import time
    start_time = time.time()

    #16 for nightingale, 96 for google cloud
    fraction_of_cpus = 1

    agents = [
        CPA.Construction_Paper_Agent(lower_agent=BFS_Agent(horizon=1, scoring_function=bw.F1score,scoring='Average')),     
        CPA.Construction_Paper_Agent(lower_agent=BFS_Agent(horizon=3,scoring_function=bw.F1score,scoring='Average'))     
        ]

    silhouettes = {i : bl.load_interesting_structure(i) for i in bl.SILHOUETTE8}
    worlds_silhouettes = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default) for i,s in silhouettes.items()}
    worlds_small = {
        'stonehenge_6_4' : bw.Blockworld(silhouette=bl.stonehenge_6_4,block_library=bl.bl_stonehenge_6_4),
        'stonehenge_3_3' : bw.Blockworld(silhouette=bl.stonehenge_3_3,block_library=bl.bl_stonehenge_3_3),
        # 'block' : bw.Blockworld(silhouette=bl.block,block_library=bl.bl_stonehenge_3_3),
        # 'T' : bw.Blockworld(silhouette=bl.T,block_library=bl.bl_stonehenge_6_4),
        # 'side_by_side' : bw.Blockworld(silhouette=bl.side_by_side,block_library=bl.bl_stonehenge_6_4),
    }
    worlds = {**worlds_silhouettes,**worlds_small}

    results = experiment_runner.run_experiment(worlds,agents,100,40,verbose=False,parallelized=fraction_of_cpus,save='construction_paper_agent')
    print(results[['agent','world','world_status']])

    print("Done in %s seconds" % (time.time() - start_time))