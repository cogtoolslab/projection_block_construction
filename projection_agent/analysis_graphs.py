"""This file contains code for graphs. These expect to be passed a dataframe output of experiment_runner (not the run dataframe, but the dataframe containing rows with agents and so) with a preselection already made."""
from analysis_helper import *

PADDING = 20 #How long should runs be padded to ensure no missing value for early termination?

#per agent
def mean_win_per_agent(df):
    agents = df['agent'].unique()
    scores = [mean_win(df[df['agent']==a]) for a in agents]    
    plt.bar(np.arange(len(scores)),scores,align='center',label=smart_short_agent_names(agents))
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylim(0,1)
    plt.ylabel("Proportion of runs with perfect reconstruction")
    plt.title("Perfect reconstruction")
    plt.show()

def mean_failure_reason_per_agent(df,fast_fail=False):
    agents = df['agent'].unique()
    #Full
    scores = [mean_failure_reason(df[df['agent']==a],"Full") for a in agents]    
    plt.bar(np.arange(len(scores))+0,scores,align='center',label="Full",width=0.15)
    #Unstable
    scores = [mean_failure_reason(df[df['agent']==a],"Unstable") for a in agents]    
    plt.bar(np.arange(len(scores))+.15,scores,align='center',label="Unstable",color='green',width=0.15)
    #Ongoing
    scores = [mean_failure_reason(df[df['agent']==a],"Ongoing") for a in agents]    
    plt.bar(np.arange(len(scores))+.3,scores,align='center',label="Did not finish",color='yellow',width=0.3)
    if fast_fail:
        #Outside
        scores = [mean_failure_reason(df[df['agent']==a],"Outside") for a in agents]    
        plt.bar(np.arange(len(scores))+.45,scores,align='center',label="Outside",color='orange',width=0.15)
        #Holes
        scores = [mean_failure_reason(df[df['agent']==a],"Holes") for a in agents]    
        plt.bar(np.arange(len(scores))+.6,scores,align='center',label="Holes",color='red',width=0.15)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Proportion of failed runs with failure mode")
    plt.ylim(0)
    plt.title("Reasons for failure in runs without perfect reconstruction")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def avg_steps_to_end_per_agent(df):
    agents = df['agent'].unique()
    #all
    results = [avg_steps_to_end(df[df['agent']==a]) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    results = [avg_steps_to_end(df[(df['agent']==a) & (df['outcome'] == 'Win')]) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    results = [avg_steps_to_end(df[(df['agent']==a) & (df['outcome'] == 'Fail')]) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Average number of steps")
    plt.ylim(0)
    plt.title("Average number of steps to end of run")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_score_per_agent(df):
    agents = df['agent'].unique()
    scoring_function = bw.F1score
    #all
    results = [mean_score(df[df['agent']==a],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    results = [mean_peak_score(df[(df['agent']==a) & (df['outcome'] == 'Win')],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    results = [mean_peak_score(df[(df['agent']==a) & (df['outcome'] == 'Fail')],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Mean F1 score at end")
    plt.ylim(0)
    plt.title("Mean end score: "+scoring_function.__name__)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_peak_score_per_agent(df):
    agents = df['agent'].unique()
    scoring_function = bw.F1score
    #all
    results = [mean_peak_score(df[df['agent']==a],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    results = [mean_peak_score(df[(df['agent']==a) & (df['outcome'] == 'Win')],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    results = [mean_peak_score(df[(df['agent']==a) & (df['outcome'] == 'Fail')],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)

    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Mean peak F1 score")
    plt.ylim(0)
    plt.title("Mean peak score: "+scoring_function.__name__)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_avg_area_under_curve_to_peakF1_per_agent(df):
    agents = df['agent'].unique()
    scoring_function = bw.F1score
    #all
    results = [mean_avg_area_under_curve_to_peakF1(df[df['agent']==a],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    results = [mean_avg_area_under_curve_to_peakF1(df[(df['agent']==a) & (df['outcome'] == 'Win')],scoring_function) for a in agents]    
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    results = [mean_avg_area_under_curve_to_peakF1(df[(df['agent']==a) & (df['outcome'] == 'Fail')],scoring_function) for a in agents]    
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
    agents = df['agent'].unique() 
    agent_names = smart_short_agent_names(agents)
    for a,agent in enumerate(agents): #plot per agent
        a_runs = df[df['agent'] == agent]
        run_scores = []
        for i,row in a_runs.iterrows(): #for each run of the agent
            blockmaps = get_blockmaps(row['run']) #get sequence of blockmaps
            #calculate the score for each blockmap
            scores = []
            for bm in blockmaps:
                #make a State to score
                state = State(bw_worlds[short_world_name(row['world'])],bm)
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
    agents = df['agent'].unique() 
    agent_names = smart_short_agent_names(agents)
    for a,agent in enumerate(agents): #plot per agent
        a_runs = df[df['agent'] == agent]
        run_scores = []
        for i,row in a_runs.iterrows(): #for each run of the agent
            #get the last block list
            blocks = row['run'][row['run']['blocks'].notnull()].iloc[-1]['blocks'][0]
            scores = [b.x*b.y for b in blocks]
            
        #     #append (pad) score with last value to xlim as a way of handling the early termination of trials
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
    agents = df['agent'].unique()
    #all
    results = [touching_last_block_score(df[df['agent']==a]) for a in agents]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    results = [touching_last_block_score(df[(df['agent']==a) & (df['outcome'] == 'Win')]) for a in agents]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    results = [touching_last_block_score(df[(df['agent']==a) & (df['outcome'] == 'Fail')]) for a in agents]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Proportion of block placements touching last placed block")
    plt.ylim(0,1)
    plt.title("Proportion of local placements")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def mean_pairwise_euclidean_distance_between_runs(df):
    agents = df['agent'].unique()
    worlds =df['world'].unique()
    #all
    results = [euclidean_distance_between_blocks_across_all_runs(df[(df['agent']==a) & df['world'] == w]) for a in agents for w in worlds]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+0,scores,align='center',yerr=stds,label="All",width=0.2)
    #win
    results = [euclidean_distance_between_blocks_across_all_runs(df[(df['agent']==a) & (df['outcome'] == 'Win') & df['world'] == w]) for a in agents for w in worlds]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.2,scores,align='center',yerr=stds,label="Win",color='green',width=0.2)
    #fail
    results = [euclidean_distance_between_blocks_across_all_runs(df[(df['agent']==a) & (df['outcome'] == 'Fail') & df['world'] == w]) for a in agents for w in worlds]
    scores = [score for score,std in results]
    stds = [std for score,std in results]
    plt.bar(np.arange(len(scores))+.4,scores,align='center',yerr=stds,label="Fail",color='orange',width=0.2)
    plt.xticks(np.arange(len(scores)),smart_short_agent_names(agents),rotation=45,ha='right')
    plt.ylabel("Average pairwise Euclidean distance between runs on same silhouette per agent")
    plt.ylim(0,1)
    plt.title("Mean Euclidean distance")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()