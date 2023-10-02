"""Contains code for generating figures and more complicated visualizations"""

from scoping_simulations.utils import world
from analysis.utils.analysis_graphs import *
import imageio

proj_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
plots_dir = os.path.join(proj_dir,'results/plots')

def build_animation(slice,title=None):
    """Expects a slice of a dataframe corresponding to one run and outputs a series of images as animation"""
    if title is None: 
        title = 'anim'
    # get current subgoals
    try:
        decomposed_silhouettes = list(slice['decomposed_silhouette'])
        decomposed_silhouettes = [s for s in decomposed_silhouettes if type(s) is not float] #remove nans (which are floats)
    except KeyError:
        decomposed_silhouettes = []
    if decomposed_silhouettes == []: #if we don't have any decomposed silhouette
        decomposed_silhouettes = [slice['_world'].unique()[0].silhouette]
    ds_counter = 0 #how many decompositions have we seen?
    # get sequence of plans for Subgoal agent assuming construction paper
    try:
        plans = list(slice['_subgoal_sequence'])
        plans = [[g['name'] for g in s] if type(s[0]) is dict else [g['name'] for g in s[0]] for s in plans if type(s) is not float] #get list of names for the plans that aren't empty
        # TODO why are some subgoal sequences wrapped in an additional list?
    except KeyError:
        plans = []
    if plans == []: plans = [[8]] #if we dont have a plan the plan is full decomp
    plan_counter = 0
    frame = 0 
    for i,row in slice.iterrows():
        # get world
        world = row['_world'].silhouette
        try:
            read_silhouette = row['decomposed_silhouette']
        except KeyError:
            read_silhouette = np.nan #use fallback below
        if type(read_silhouette) is float: #if we have a nan, get the right silhouette
            silhouette = decomposed_silhouettes[ds_counter]
        else:
            ds_counter += 1
            ds_counter = min(ds_counter,len(decomposed_silhouettes)-1)
            silhouette = decomposed_silhouettes[ds_counter]
        try:
            read_plan = row['_subgoal_sequence']
        except KeyError:
            read_plan = np.nan #fallback for below
        if type(read_plan) is float: #if we have a nan, get the right silhouette
            plan = plans[plan_counter]
        else:
            plan_counter += 1
            plan_counter = min(plan_counter,len(plans)-1)
            plan = plans[plan_counter]
        plt.figure(figsize=(4,4))
        plt.xticks([])
        plt.yticks([])
        if title != 'anim':
            plt.title(title)
        plt.pcolor(row['blockmap'][::-1], cmap='hot_r',vmin=0,vmax=20,linewidth=0,edgecolor='none')
        #we print the target silhouette as very transparent overlay 🌍
        plt.pcolor(world[::-1],cmap='binary',alpha=.4,linewidth=2,facecolor='none',edgecolor='black',capstyle='round',joinstyle='round',linestyle=':')
        if silhouette is not None:
            if len(silhouette) == 1:
                silhouette = silhouette[0]
            #we print the subgoal as transparent overlay. No subgoal will just overlay the complete silhouette twice
            plt.pcolor(silhouette[::-1],cmap='binary',alpha=.8,linewidth=2,facecolor='none',edgecolor='black',capstyle='round',joinstyle='round',linestyle=':')
        if plan is not None:
            plt.hlines(plan,0,8,linestyles='--',colors='blue',linewidth=4)
        if type(row['world_failure_reason']) is not float and row['world_failure_reason'] != "None":
            plt.text(4,4,s=row['world_failure_reason'],fontdict=    {
                    'weight' : 'bold',
                    'size'   : 32
                    },
                    horizontalalignment='center',     verticalalignment='center')
        frame += 1
        plt.savefig("_"+"title"+str(frame)+".png")
    #load the files we created
    files = ["_"+"title"+str(frame+1)+".png" for  frame in range(frame)]
    images = [imageio.imread(file) for file in files]
    imageio.mimwrite(os.path.join(plots_dir,title+'.gif'), images, fps=1)
    #delete temp images
    for file in files:
        os.remove(file)
    #close figures
    plt.close('all')
    print("created "+title+".gif")

