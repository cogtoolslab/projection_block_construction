"""This file contains code for graphs. These expect to be passed a dataframe output of experiment_runner (not the run dataframe, but the dataframe containing rows with agents and so) with a preselection already made."""
from analysis_helper import *

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

    plt.xticks(np.arange(len(scores)),agents,rotation=45,ha='right')
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

    plt.xticks(np.arange(len(scores)),agents,rotation=45,ha='right')
    plt.ylabel("Average number of steps")
    plt.ylim(0)
    plt.title("Average number of steps to end of run")
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

    plt.xticks(np.arange(len(scores)),agents,rotation=45,ha='right')
    plt.ylabel("Mean peak F1 score")
    plt.ylim(0)
    plt.title("Mean peak score: "+scoring_function.__name__)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()