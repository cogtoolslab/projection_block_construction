"""This file contains code for graphs. These expect to be passed a dataframe output of experiment_runner (not the run dataframe, but the dataframe containing rows with agents and so) with a preselection already made."""
from analysis.utils.analysis_helper import *
import textwrap
import analysis.utils.trajectory as trajectory

PADDING = 20 #How long should runs be padded to ensure no missing value for early termination?

#per agent
def mean_win_per_agent(df):
    df = final_rows(df)
    agents = df['agent_attributes'].unique()
    scores = [mean_win(df[df['agent_attributes']==a]) for a in agents]    
    plt.bar(np.arange(len(scores)),scores,align='center',label=smart_short_agent_names(agents))
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylim(0,1)
    plt.ylabel("Proportion of runs with perfect reconstruction")
    plt.title("Perfect reconstruction")
    plt.show()

def mean_failure_reason_per_agent(df,fast_fail=False):
    agents = df['agent_attributes'].unique()
    df = final_rows(df)
    #Full
    scores = [mean_failure_reason(df[(df['agent_attributes']==a) & (df['world_status'].isin(['Fail','Ongoing']))],"Full") for a in agents]    
    plt.bar(np.arange(len(scores))+0,scores,align='center',label="Full",width=0.15)
    #Unstable
    scores = [mean_failure_reason(df[(df['agent_attributes']==a) & (df['world_status'].isin(['Fail','Ongoing']))],"Unstable") for a in agents]    
    plt.bar(np.arange(len(scores))+.15,scores,align='center',label="Unstable",color='green',width=0.15)
    #Ongoing
    scores = [mean_failure_reason(df[(df['agent_attributes']==a) & (df['world_status'].isin(['Fail','Ongoing']))],"None") for a in agents]    
    plt.bar(np.arange(len(scores))+.3,scores,align='center',label="Did not finish",color='yellow',width=0.3)
    if fast_fail:
        #Outside
        scores = [mean_failure_reason(df[df['agent_attributes']==a],"Outside") for a in agents]    
        plt.bar(np.arange(len(scores))+.45,scores,align='center',label="Outside",color='orange',width=0.15)
        #Holes
        scores = [mean_failure_reason(df[df['agent_attributes']==a],"Holes") for a in agents]    
        plt.bar(np.arange(len(scores))+.6,scores,align='center',label="Holes",color='red',width=0.15)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Proportion of failed runs with failure mode")
    plt.ylim(0)
    plt.title("Reasons for failure in runs without perfect reconstruction")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def avg_steps_to_end_per_agent(df):
    agents = df['agent_attributes'].unique()
    #all
    results = [avg_steps_to_end(df[df['agent_attributes']==a]) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    results = [avg_steps_to_end(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)]) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    results = [avg_steps_to_end(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)]) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Average number of steps")
    plt.ylim(0)
    plt.title("Average number of steps to end of run")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_score_per_agent(df,scoring_function=bw.F1score):
    agents = df['agent_attributes'].unique()
    #all
    results = [mean_score(df[df['agent_attributes']==a],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    results = [mean_peak_score(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    results = [mean_peak_score(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Mean "+scoring_function.__name__+" at end")
    plt.ylim(0)
    plt.title("Mean end score: "+scoring_function.__name__)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_peak_score_per_agent(df,scoring_function=bw.F1score):
    agents = df['agent_attributes'].unique()
    #all
    results = [mean_peak_score(df[df['agent_attributes']==a],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    results = [mean_peak_score(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    results = [mean_peak_score(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Mean "+scoring_function.__name__+" at peak")
    plt.ylim(0)
    plt.title("Mean peak score: "+scoring_function.__name__)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_avg_area_under_curve_to_peakF1_per_agent(df):
    agents = df['agent_attributes'].unique()
    scoring_function = bw.F1score
    #all
    results = [mean_avg_area_under_curve_to_peakF1(df[df['agent_attributes']==a],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    results = [mean_avg_area_under_curve_to_peakF1(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    results = [mean_avg_area_under_curve_to_peakF1(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Mean average F1 score")
    plt.ylim(0)
    plt.title("Mean average F1 score during run")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def graph_mean_F1_over_time_per_agent(df):
    #plot mean F1,std over time for chosen world and over agents in one plot (with continuation)
    agents = df['agent_attributes'].unique() 
    agent_names = smart_short_agent_names(agents)
    for a,agent in enumerate(agents): #plot per agent
        a_runs = get_runs(df[df['agent_attributes'] == agent])
        run_scores = []
        for i,row in enumerate(a_runs): #for each run of the agent
            blockmaps = row['blockmap'] #get sequence of blockmaps
            #calculate the score for each blockmap
            scores = []
            for bm in blockmaps:
                #make a State to score
                state = State(row['_world'].tail(1).item(),bm)
                score = bw.F1score(state)
                scores.append(score)
            #append (pad) score with last value to xlim as a way of handling the early termination of trials
            scores = [scores[i] if i < len(scores) else scores[-1] for i in range(PADDING+1)]
            run_scores.append(scores)
        #avg,std
        avgs = np.mean(run_scores,axis=0)
        stds = np.std(run_scores,axis=0)
        #plot
    #     plt.plot(range(len(avgs)),avgs)
        plt.errorbar(range(len(avgs)),avgs,stds,label=agent_names[a])
        plt.xlim(0,PADDING)
        plt.ylim(0,1)
    plt.title('Mean F1 score over steps')
    plt.ylabel("Mean F1 score")
    plt.xlabel("Step")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def graph_avg_blocksize_over_time_per_agent(df):
    agents = df['agent_attributes'].unique() 
    agent_names = smart_short_agent_names(agents)
    for a,agent in enumerate(agents): #plot per agent
        a_runs = get_runs(df[df['agent_attributes'] == agent])
        run_scores = []
        for run in a_runs: #for each run of the agent
            scores = list(run['action_block_width']*run['action_block_height'])
            #append (pad) score with last value to xlim as a way of handling the early termination of trials
            scores = [scores[i] if i < len(scores) else np.nan for i in range(PADDING+1)]
            run_scores.append(scores)
        #avg,std
        run_scores = np.array(run_scores)
        avgs = np.nanmean(run_scores,axis=0)
        stds = np.nanstd(run_scores,axis=0)
        #plot
    #     plt.plot(range(len(avgs)),avgs)
        plt.errorbar(range(len(avgs)),avgs,stds,label=agent_names[a])
        plt.xlim(0,PADDING)
        # plt.ylim(0,1)
    plt.title('Average block size over steps')
    plt.ylabel("Block size")
    plt.xlabel("Step")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_touching_last_block_per_agent(df):
    agents = df['agent_attributes'].unique()
    #all
    results = [touching_last_block_score(df[df['agent_attributes']==a]) for a in agents]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    results = [touching_last_block_score(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)]) for a in agents]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    results = [touching_last_block_score(df[(df['run_ID'].isin(run_IDs)) & (df['agent_attributes']==a)]) for a in agents]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Proportion of block placements touching last placed block")
    plt.ylim(0,1)
    plt.title("Proportion of local placements")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_pairwise_raw_euclidean_distance_between_runs(df):
    agents = df['agent_attributes'].unique()
    worlds =df['world'].unique()
    #all
    results = [[pairwise_raw_euclidean_distance_between_blocks_across_all_runs(df[(df['agent_attributes']==a) & (df['world'] == w)]) for w in worlds] for a in agents]
    scores = [statistics.mean([l for aw in a for l in aw]) for a in results]
    stds = [statistics.stdev([l for aw in a for l in aw]) for a in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    results = [[pairwise_raw_euclidean_distance_between_blocks_across_all_runs(df[(df['run_ID'].isin(run_IDs)) & (df['world'] == w) & (df['agent_attributes'] == a)]) for w in worlds] for a in agents] #if statement is to prevent empty lists
    scores = [statistics.mean([l for aw in a for l in aw]) if sum([len(r) for r in a]) else 0 for a in results]
    stds = [statistics.stdev([l for aw in a for l in aw]) if sum([len(r) for r in a]) > 1 else 0 for a in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    results = [[pairwise_raw_euclidean_distance_between_blocks_across_all_runs(df[(df['run_ID'].isin(run_IDs)) & (df['world'] == w) & (df['agent_attributes'] == a)]) for w in worlds] for a in agents]
    scores = [statistics.mean([l for aw in a for l in aw]) for a in results]
    stds = [statistics.stdev([l for aw in a for l in aw]) for a in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Mean Euclidean distance")
    plt.title("Average pairwise Euclidean distance between runs on same silhouette per agent")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def total_avg_states_evaluated_per_agent(df):
    agents = df['agent_attributes'].unique()
    #all
    scores = [[sum(run['states_evaluated']) for run in get_runs(df[ df['agent_attributes'] == a ])] for a in agents]
    means = [statistics.mean(r) for r in scores]
    stds = [statistics.stdev(r) for r in scores]
    plt.bar(np.arange(len(scores))+0,means,align='center',yerr=stds,label="All",width=0.2)
    #win
    run_IDs = df[df['world_status'] == 'Win']['run_ID'].unique()
    scores = [[sum(run['states_evaluated']) for run in get_runs(df[ (df['agent_attributes'] == a ) & (df['run_ID'].isin(run_IDs))])] for a in agents]
    means = [statistics.mean(r) if r != [] else 0 for r in scores]
    stds = [statistics.stdev(r) if r != [] else 0 for r in scores]
    plt.bar(np.arange(len(scores))+.2,means,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    run_IDs = df[df['world_status'] == 'Fail']['run_ID'].unique()
    scores = [[sum(run['states_evaluated']) for run in get_runs(df[ (df['agent_attributes'] == a ) & (df['run_ID'].isin(run_IDs))])] for a in agents]
    means = [statistics.mean(r) if r != [] else 0 for r in scores]
    stds = [statistics.stdev(r) if r != [] else 0 for r in scores]
    plt.bar(np.arange(len(scores))+.4,means,align='center',yerr=stds,label="Fail",color='orange',width=0.2)
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylim(bottom=0)
    plt.ylabel("Average planning cost ")
    plt.title("Average planning cost (number of states evaluated) per run")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

#on worlds
def illustrate_worlds(df):
    unique_world_names = df['world'].unique()
    unique_world_obj = {w:df[df['world'] == w].head(1)['_world'].item() for w in unique_world_names}
    unique_world_obj = {key: value for key, value in sorted(unique_world_obj.items(), key=lambda item: item[0])}
    plt.figure(figsize=(20,20))  
    for i,(name,world_obj) in enumerate(list(unique_world_obj.items())):
        plt.subplot(math.sqrt(len(unique_world_obj)),math.sqrt(len(unique_world_obj))+1,i+1)
        plt.imshow(world_obj.silhouette)
        plt.title(name)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def mean_win_per_agent_over_worlds(df):
    df = final_rows(df)
    agents = df['agent_attributes'].unique()
    unique_world_names = df['world'].unique()
    unique_world_obj = {w:df[df['world'] == w].head(1)['_world'].item() for w in unique_world_names}
    unique_world_obj = {key: value for key, value in sorted(unique_world_obj.items(), key=lambda item: item[0])}
    #create plot
    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(len(unique_world_names),2,figsize=(10,20))
    fig.suptitle("Perfect reconstruction per agent over silhouettes")
    for i, (world_name,world_obj) in enumerate(list(unique_world_obj.items())):
        _df = df[df['world'] == world_name]
        # illustrate world
        axes[i,0].imshow(world_obj.silhouette)
        # axes[i,0].set_title(world_name)
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        #from mean_win_per_agent: plot
        scores = [mean_win(_df[_df['agent_attributes']==a]) for a in agents]    
        axes[i,1].bar(np.arange(len(scores)),scores,align='center',label=smart_short_agent_names(agents))
        # axes[i,1].set_xticks(np.arange(len(scores)),smart_short_agent_names(agents))
        axes[i,1].set_xticks([])
        axes[i,1].set_ylim(0,1)
        # axes[i,1].set_ylabel("Proportion of runs with perfect reconstruction")
        # axes[i,1].set_title("Perfect reconstruction on "+world_name)
    #only show agent labels at the bottom
    axes[len(unique_world_obj)-1,1].set_xticks(np.arange(len(scores)),smart_short_agent_names(agents))
    axes[len(unique_world_obj)-1,1].set_xticklabels(smart_short_agent_names(agents))
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.show()

def mean_peak_F1_per_agent_over_worlds(df):
    scoring_function = bw.F1score
    agents = df['agent_attributes'].unique()
    unique_world_names = df['world'].unique()
    unique_world_obj = {w:df[df['world'] == w].head(1)['_world'].item() for w in unique_world_names}
    unique_world_obj = {key: value for key, value in sorted(unique_world_obj.items(), key=lambda item: item[0])}
    #create plot
    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(len(unique_world_names),2,figsize=(10,20))
    fig.suptitle("Perfect reconstruction per agent over silhouettes")
    for i, (world_name,world_obj) in enumerate(list(unique_world_obj.items())):
        _df = df[df['world'] == world_name]
        # illustrate world
        axes[i,0].imshow(world_obj.silhouette)
        # axes[i,0].set_title(world_name)
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        #from mean_peak_score_per_agent: plot
        #all
        results = [mean_peak_score(_df[_df['agent_attributes']==a],scoring_function) for a in agents]    
        scores = [score for score,std in results]
        stds = [std for score,std in results]
        axes[i,1].bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
        #win
        run_IDs = _df[_df['world_status'] == 'Win']['run_ID'].unique()
        results = [mean_peak_score(_df[(_df['run_ID'].isin(run_IDs)) & (_df['agent_attributes']==a)],scoring_function) for a in agents]    
        scores = [score for score,std in results]
        stds = [std for score,std in results]
        axes[i,1].bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
        #fail
        run_IDs = _df[_df['world_status'] == 'Fail']['run_ID'].unique()
        results = [mean_peak_score(_df[(_df['run_ID'].isin(run_IDs)) & (_df['agent_attributes']==a)],scoring_function) for a in agents]    
        scores = [score for score,std in results]
        stds = [std for score,std in results]
        axes[i,1].bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)
        axes[i,1].set_xticks([])
        axes[i,1].set_ylim(0,1)
    #only show agent labels at the bottom
    axes[len(unique_world_obj)-1,1].set_xticks(np.arange(len(scores)),smart_short_agent_names(agents))
    axes[len(unique_world_obj)-1,1].set_xticklabels(smart_short_agent_names(agents))
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_failure_reason_per_agent_over_worlds(df,fast_fail=False):
    df = final_rows(df)
    agents = df['agent_attributes'].unique()
    unique_world_names = df['world'].unique()
    unique_world_obj = {w:df[df['world'] == w].head(1)['_world'].item() for w in unique_world_names}
    unique_world_obj = {key: value for key, value in sorted(unique_world_obj.items(), key=lambda item: item[0])}
    #create plot
    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(len(unique_world_names),2,figsize=(10,20))
    fig.suptitle("Perfect reconstruction per agent over silhouettes")
    for i, (world_name,world_obj) in enumerate(list(unique_world_obj.items())):
        _df = df[df['world'] == world_name]
        # illustrate world
        axes[i,0].imshow(world_obj.silhouette)
        # axes[i,0].set_title(world_name)
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        #from mean_peak_score_per_agent: plot
        #full
        scores = [mean_failure_reason(_df[(_df['agent_attributes']==a) & (_df['world_status'].isin(['Fail','Ongoing']))],"Full") for a in agents]    
        axes[i,1].bar(np.arange(len(scores))+0,scores,align='center',label="Full",width=0.15)
        #Unstable
        scores = [mean_failure_reason(_df[(_df['agent_attributes']==a) & (_df['world_status'].isin(['Fail','Ongoing']))],"Unstable") for a in agents]    
        axes[i,1].bar(np.arange(len(scores))+.15,scores,align='center',label="Unstable",color='green',width=0.15)
        #Ongoing
        scores = [mean_failure_reason(_df[(_df['agent_attributes']==a) & (_df['world_status'].isin(['Fail','Ongoing']))],"Ongoing") for a in agents]    
        axes[i,1].bar(np.arange(len(scores))+.3,scores,align='center',label="Did not finish",color='yellow',width=0.3)
        if fast_fail:
            #Outside
            scores = [mean_failure_reason(_df[_df['agent_attributes']==a],"Outside") for a in agents]    
            axes[i,1].bar(np.arange(len(scores))+.45,scores,align='center',label="Outside",color='orange',width=0.15)
            #Holes
            scores = [mean_failure_reason(_df[_df['agent_attributes']==a],"Holes") for a in agents]    
            axes[i,1].bar(np.arange(len(scores))+.6,scores,align='center',label="Holes",color='red',width=0.15)
        axes[i,1].set_ylim(0)
        axes[i,1].set_xticks([])
    #only show agent labels at the bottom
    axes[len(unique_world_obj)-1,1].set_xticks(np.arange(len(scores)),smart_short_agent_names(agents))
    axes[len(unique_world_obj)-1,1].set_xticklabels(smart_short_agent_names(agents))
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def heatmaps_at_peak_per_agent_over_world(df):
    df = peak_F1_rows(df)
    agents = df['agent_attributes'].unique()
    unique_world_names = df['world'].unique()
    unique_world_obj = {w:df[df['world'] == w].head(1)['_world'].item() for w in unique_world_names}
    unique_world_obj = {key: value for key, value in sorted(unique_world_obj.items(), key=lambda item: item[0])}
    #create plot
    fig, axes = plt.subplots(len(unique_world_names),len(agents)+1,figsize=(2+2*len(agents),4+2*len(unique_world_names)))
    fig.suptitle("Heatmap at peak F1 per agent over silhouettes")
    for i, (world_name,world_obj) in enumerate(list(unique_world_obj.items())):
        _df = df[df['world'] == world_name]
        # illustrate world
        axes[i,0].imshow(world_obj.silhouette)
        # axes[i,0].set_title(world_name)
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        axes[i,0].set_title(textwrap.fill(world_name,width=20), fontsize=10,wrap=True)
        #generate heatmaps
        for j,agent in enumerate(agents):
            bms = df[(df['agent_attributes'] == agent) & (df['world'] == world_name)]['blockmap'] #get the correct bms
            shape = bms.head(1).item().shape
            bms = bms.apply(lambda x: (x > np.zeros(shape))*1.) #make bitmap
            heatmap = np.sum(bms)
            axes[i,j+1].imshow(heatmap,cmap='viridis')    
            axes[i,j+1].set_yticks([])
            axes[i,j+1].set_xticks([])
            tit = smart_short_agent_names(agents)[j]
            axes[i,j+1].set_title(textwrap.fill(tit,width=20), fontsize=10,wrap=True)
    plt.show()

def trajectory_per_agent_over_world(df):
    agents = df['agent_attributes'].unique()
    unique_world_names = df['world'].unique()
    unique_world_obj = {w:df[df['world'] == w].head(1)['_world'].item() for w in unique_world_names}
    unique_world_obj = {key: value for key, value in sorted(unique_world_obj.items(), key=lambda item: item[0])}
    unique_world_names = df['world'].unique()
    #create plot
    fig, axes = plt.subplots(len(unique_world_names),len(agents)+1,figsize=(2+4*len(agents),2+3*len(unique_world_names)))
    fig.suptitle("Trajectory graph per agent over silhouettes")
    for i, (world_name,world_obj) in enumerate(list(unique_world_obj.items())):
        _df = df[df['world'] == world_name]
        dfic = trajectory.agentdf_to_dfic(_df)
        # illustrate world
        axes[i,0].imshow(world_obj.silhouette)
        # axes[i,0].set_title(world_name)
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        axes[i,0].set_title(textwrap.fill(world_name,width=20), fontsize=10,wrap=True)
        #generate heatmaps
        for j,agent in enumerate(dfic['agent'].unique()):
            img = trajectory.plot_trajectory_graph(data=dfic,
                    target_name = world_name, 
                    agent = agent,
                    show = False,
                    save = False,
                    x_upper_bound = world_obj.silhouette.shape[0])
            axes[i,j+1].imshow(img)    
            axes[i,j+1].set_yticks([])
            axes[i,j+1].set_xticks([])
            tit = smart_short_agent_names(agents)[j]
            axes[i,j+1].set_title(textwrap.fill(tit,width=20), fontsize=10,wrap=True)
    plt.show()

# scatter plots
