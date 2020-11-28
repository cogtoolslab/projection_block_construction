"""Contains code for generating figures and more complicated visualizations"""

from analysis.utils.analysis_graphs import *
import imageio

proj_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
plots_dir = os.path.join(proj_dir,'results/plots')

def build_animation(slice,title=None):
    """Expects a slice of a dataframe corresponding to one run and outputs a series of images as animation"""
    if title is None: title = 'anim'
    decomposed_silhouettes = list(slice['decomposed_silhouette'])
    decomposed_silhouettes = [s for s in decomposed_silhouettes if type(s) is not float] #remove nans (which are floats)
    if decomposed_silhouettes == []: #if we don't have any decomposed silhouette
        decomposed_silhouettes = [slice['_world'].unique()[0].silhouette]
    ds_counter = 0 #how many decompositions have we seen?
    frame = 0 
    #remove nans
    for i,row in slice.iterrows():
        read_silhouette = row['decomposed_silhouette']
        if type(read_silhouette) is float: #if we have a nan, get the right silhouette
            silhouette = decomposed_silhouettes[ds_counter]
        else:
            ds_counter += 1
            ds_counter = min(ds_counter,len(decomposed_silhouettes)-1)
            silhouette = decomposed_silhouettes[ds_counter]
        plt.figure(figsize=(4,4))
        plt.xticks([])
        plt.yticks([])
        plt.pcolor(row['blockmap'][::-1], cmap='hot_r',vmin=0,vmax=20,linewidth=0,edgecolor='none')
        if silhouette is not None:
            if len(silhouette) == 1:
                silhouette = silhouette[0]
            #we print the target silhouette as transparent overlay
            plt.pcolor(silhouette[::-1],cmap='binary',alpha=.8,linewidth=2,facecolor='none',edgecolor='black',capstyle='round',joinstyle='round',linestyle=':')
        frame += 1
        plt.savefig("_"+"title"+str(frame)+".png")
    #load the files we created
    files = ["_"+"title"+str(frame+1)+".png" for  frame in range(frame)]
    images = [imageio.imread(file) for file in files]
    imageio.mimwrite(os.path.join(plots_dir,title+'.gif'), images, fps=1)
    #delete temp images
    for file in files:
        os.remove(file)
    print("created "+title+".gif")

