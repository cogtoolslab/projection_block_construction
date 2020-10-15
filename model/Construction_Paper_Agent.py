import os
import sys
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from model.BFS_Agent import *
import numpy as np
import copy
import random

# Decomposition functions
# pass these as arguments to the agent
# they take the agent as the first argument

def horizontal_construction_paper_holes(self,current_built = None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. 

    The current position of the "construction paper" is inferered by comparing the what is currently built to the silhouette.
    
    
    ```
    world = new world
    full_silhouette = world.silhouette
    low_level_agent = new breadth_first_search_agent
    paper_y = 0
    while w.status is ongoing:
        # "slide up the paper" until the last point where the scene would no longer be bisected into two separate structures
        while full_silhouette.up_to(paper_y) is fully connected:
            paper_y += 1
        while full_silhouette.up_to(paper_y) is not fully connected:
            paper_y += 1
        # run lower level agent on world with silhouette up to a certain y
        temp_world = world
        temp_world.silhouette = full_silhouette.up_to(paper_y)
        low_level_agent.act(world)
        # handling a failed decomposition
        if temp_world.status is failure:
            #the attempted decomposition failed, so we try another
            paper_y += 1
        else:
            #make the change official
            world = temp_world
    ```

    In words, it tries to decompose by sliding the construction paper as far up as it can so it still decomposing the resulting silhouette. Intuitively, it places the lower edge of the construction at the upper edge of the first hole from the bottom up, then the second and so on. If it fails to build the resulting partial silhouette, it just tries the next position further up.
    """
    if current_built is None: current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0: y += 1
    #slide up paper to find edge
    #get map of rows containing a vertical edge with built on top and hole below
    row_map = find_edges(full_silhouette)
    while y > 0:
        y = y - 1
        if row_map[y]: break
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y,:] = 0
    return new_silhouette

def vertical_construction_paper_holes(self,current_built = None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. 
    Same as `horizontal_construction_paper_holes`, but the construction paper is moved from left to right. 
    We don't expect this to be successful. This is just to investigate other ways of chunking the structure.
    """
    if current_built is None: current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    #rotate the matrices
    full_silhouette = np.rot90(full_silhouette,k=1)
    current_built = np.rot90(current_built,k=1)
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0: y += 1
    #slide up paper to find edge
    #get map of rows containing a vertical edge with built on top and hole below
    row_map = find_edges(full_silhouette)
    while y > 0:
        y = y - 1
        if row_map[y]: break
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y,:] = 0
    #rotate back
    new_silhouette = np.rot90(new_silhouette,k=-1)
    return new_silhouette

def random_1_4(self,current_built = None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a random increment between 1 and 4.
    """
    if current_built is None: current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0: y += 1
    #increment y
    y = y - random.randint(1,4)
    #limit to height of area
    y = min(y,full_silhouette.shape[0])
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y,:] = 0
    return new_silhouette    

def fixed(self,increment,current_built = None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment. 
    """
    if current_built is None: current_built = self.world.current_state.blockmap
    full_silhouette = self.world.silhouette
    current_built = self.world.current_state.blockmap
    # get the index of the first empty row—no need to touch the area where something has been built already
    y = 0
    while y < current_built.shape[0] and current_built[y].sum() == 0: y += 1
    #increment y
    y = y - increment
    #limit to height of area
    y = min(y,full_silhouette.shape[0])
    new_silhouette = copy.deepcopy(full_silhouette)
    new_silhouette[0:y,:] = 0
    return new_silhouette    

def fixed_1(self,current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment. """
    return fixed(self,1,current_built)

def fixed_2(self,current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment. """
    return fixed(self,2,current_built)

def fixed_3(self,current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment. """
    return fixed(self,3,current_built)

def fixed_4(self,current_built=None):
    """Returns a new target silhouette, which is a subset of the full silhouette of the world. Moves the construction paper horizontally upwards by a fixed increment. """
    return fixed(self,4,current_built)

def no_decomposition(self,current_built=None):
    """Returns the full silhouette. Provides the baseline of using no decomposition altogether."""
    return self.world.silhouette

def crop(arr,(bl_x,bl_y),(tr_x,tr_y)):
    """Crops the array that is passed to it. The first touple marks the x and y coordinates of the bottom left corner, the other the top right corner. Note that the top left corner is (0,0)."""
    assert(
        arr.shape
    )

# Agent
class Construction_Paper_Agent(BFS_Agent):
    """Implements the construction paper proposal for a projection based agent.

    TODO 
    - [ ] retry on failure
    - [ ] save the decompositions
    
    Different decomposition functions can be passed to the agent.
    """

    def __init__(self, world=None, lower_agent = BFS_Agent(), decomposition_function = horizontal_construction_paper_holes):
            self.world = world
            self.lower_agent = lower_agent
            self.decompose = decomposition_function

    def __str__(self):
            """Yields a string representation of the agent"""
            return self.__class__.__name__+' lower level agent: '+self.lower_agent.__str__() + ' decomposition function' + self.decompose.__name__

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {**{
            'agent_type':self.__class__.__name__,
            'decomposition_function':self.decompose.__name__
            }, **{"lower level: "+key:value for key,value in self.lower_agent.get_parameters().items()}}

    def act(self,higher_steps=-1,lower_steps=None,verbose=False):
        """Makes the agent act. This means acting for a given number of steps, where higher_steps refers to higher level of the agent (x times running the lower level agent) and lower_steps refers to the total number of steps performed by the lower level agent, ie actual steps in the world.
        
        The actual action happens in act_single_higher_step"""
        higher_step = 0
        actions = []
        costs = 0
        while higher_step != higher_steps and self.world.status()[0] == 'Ongoing': #action loop
            action,cost = self.act_single_higher_step(verbose)
            actions += action
            costs += cost
        return actions,{'states_evaluated':costs}

    def act_single_higher_step(self,verbose):
        """Takes a single step of the higher level agent. This means finding a new decomposition and then running the lower level agent on it. If it fails, it jumps to the next decomposition.
        
        The current position of the "construction paper" is inferered by comparing the what is currently built to the silhouette."""
        #get decomposition
        full_silhouette = self.world.silhouette
        new_silhouette = self.decompose(self) #external function needs to explicitly passed the agent object
        if verbose: print("Got decomposition ",new_silhouette)
        #create temporary world object containing the modified silhouette
        temp_world = copy.deepcopy(self.world)
        temp_world.silhouette = new_silhouette
        self.lower_agent.set_world(temp_world)
        #let's run the lower level agent
        action_seq = []
        costs = 0
        while temp_world.status()[0] == 'Ongoing':
            action,cost = self.lower_agent.act(verbose=verbose)
            action_seq += action
            costs += cost
        #apply actions to the world
        if verbose: print("Decomposition done, applying action_seq:",str([str(a) for a in action_seq]))
        for action in action_seq:
            self.world.apply_action(action,force=True) # we need force here since the baseblock objects are in different memory locations 
        return action_seq,costs 

# Helper functions

def find_edges(silhouette):
    """Returns a map for each row in the table that has an edge with a filled out portion on the upper side and empty space on the lower side."""
    row_map = [False] * silhouette.shape[0]
    #go thru every row of the silhouette
    for y in range(silhouette.shape[0]):
        for x in range(silhouette.shape[1]):
            if silhouette[y,x] == 0 and silhouette[y-1,x] != 0: 
                row_map[y] = True
                break
    row_map[0] = True #always offer the top edge as potential decomposition
    return row_map

