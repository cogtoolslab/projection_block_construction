# set up imports
import os
import sys
import random
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(proj_dir)
utils_dir = os.path.join(proj_dir,'utils')
sys.path.append(utils_dir)
analysis_dir = os.path.join(proj_dir,'analysis')
analysis_utils_dir = os.path.join(analysis_dir,'utils')
sys.path.append(analysis_utils_dir)
experiments_dir = os.path.join(proj_dir,'experiments')
sys.path.append(experiments_dir)


from analysis.utils.analysis_figures import *
from analysis.utils.analysis_helper import *
from analysis.utils.analysis_graphs import *
from model.Simulated_Lookahead_Subgoal_Planning_Agent import *
import utils.blockworld as bw
import time


df_dir = os.path.join(proj_dir,'results/dataframes')

df_paths = ['subgoal planning full Astar.pkl']
# load all experiments as one dataframe
df = pd.concat([pd.read_pickle(os.path.join(df_dir,l)) for l in df_paths])
print("Dataframes loaded")
# plt.rcParams["figure.figsize"] = (16,9)
# plt.rcParams.update({'font.size': 22})


start_time = time.time()

row = df.iloc[1]

a = Simulated_Lookahead_Subgoal_Planning_Agent(
    all_sequences=row['_all_sequences'], 
    parent_agent=row['_agent'],
    sequence_length=8,
    step_size=-1,
    include_subsequences=False,
    c_weight=.05)

la_seq = a.generate_sequences()

while a.world.status()[0] == "Ongoing":
    actions,info = a.act()
    print(info['_chosen_subgoal_sequence'].names())
    if actions == []: break

print(a.world.status())
print("Done in %s seconds" % (time.time() - start_time))