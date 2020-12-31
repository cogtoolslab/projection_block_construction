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

    from model.Subgoal_Planning_Agent import *
    from model.utils.decomposition_functions import *
    from model.BFS_Agent import BFS_Agent
    import utils.blockworld as bw
    import utils.blockworld_library as bl
    import experiments.experiment_runner as experiment_runner

    import time
    start_time = time.time()

    fraction_of_cpus = .8

    agents = [
        Subgoal_Planning_Agent(
                lower_agent=BFS_Agent(horizon=1,only_improving_actions=True),
                lookahead = l,
                include_subsequences=False,
                c_weight = 1/w,
                S_treshold=1,
                S_iterations=1)
                for l in [1,2,3,8] for w in [1,10,100,1000]
        ] + [
        BFS_Agent(horizon=1,only_improving_actions=True)
        ] + [
        Full_Subgoal_Planning_Agent(
            lower_agent=BFS_Agent(horizon=1,only_improving_actions=True),
            c_weight = 1/w,
            S_treshold=1,
            S_iterations=2)
            for w in [1,10,100,1000]
        ]

    silhouettes = {i : bl.load_interesting_structure(i) for i in bl.SILHOUETTE16}
    worlds = {'int_struct_'+str(i) : bw.Blockworld(silhouette=s,block_library=bl.bl_silhouette2_default,legal_action_space=True) for i,s in silhouettes.items()}
    

    results = experiment_runner.run_experiment(worlds,agents,10,20,verbose=False,parallelized=fraction_of_cpus,save=os.path.basename(__file__),maxtasksperprocess = 1)
    print(results[['agent','world','world_status']])

    print("Done in %s seconds" % (time.time() - start_time))