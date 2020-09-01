from BFS_Agent import *
import numpy as np
import copy

class Construction_Paper_Agent(BFS_Agent):
    """Implements the construction paper proposal for a projection based agent.

    TODO 
    - [ ] retry on failure
    - [ ] save the decompositions

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

    In words, it tries to decompose by sliding the construction paper as far up as it can so it still decomposing the resulting silhouette. Intuitively, it places the lower edge of the construction at the upper edge of the first hole from the bottom up, then the second and so on. If it fails to build the resulting partial silhouette, it just tries the next position further up."""

    def __init__(self, world=None, lower_agent = BFS_Agent()):
            self.world = world
            self.lower_agent = lower_agent

    def __str__(self):
            """Yields a string representation of the agent"""
            return self.__class__.__name__+' lower level agent: '+self.lower_agent.__str__()

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {**{
            'agent_type':self.__class__.__name__
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
        return actions,costs

    def act_single_higher_step(self,verbose):
        """Takes a single step of the higher level agent. This means finding a new decomposition and then running the lower level agent on it. If it fails, it jumps to the next decomposition.
        
        The current position of the "construction paper" is inferered by comparing the what is currently built to the silhouette."""
        #get decomposition
        full_silhouette = self.world.silhouette
        new_silhouette = self.decompose()
        if verbose: print("Got decomposition ",new_silhouette)
        #create temporary world object
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
        if verbose: print("Decomposition success, applying action_seq:",str([str(a) for a in action_seq]))
        for action in action_seq:
            self.world.apply_action(action,force=True) # we need force here since the baseblock objects are in different memory locations 
        return action_seq,costs


    def decompose(self,current_built = None):
        """Returns a new target silhouette, which is a subset of the full silhouette of the world. 

        The current position of the "construction paper" is inferered by comparing the what is currently built to the silhouette."""
        if current_built is None: current_built = self.world.current_state.blockmap
        full_silhouette = self.world.silhouette
        current_built = self.world.current_state.blockmap
        # get the index of the first empty row—no need to touch the area where something has been built already
        y = 0
        while y < current_built.shape[0] and current_built[y].sum() == 0: y += 1
        #slide up paper to find edge
        #get map of rows containing a vertical edge with built on top and hole below
        row_map = self.find_edges(full_silhouette)
        while y > 0:
            y = y - 1
            if row_map[y]: break
        new_silhouette = copy.deepcopy(full_silhouette)
        new_silhouette[0:y,:] = 0
        return new_silhouette 

    def find_edges(self,silhouette):
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





            
            
                    