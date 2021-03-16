import os
import sys
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from model.BFS_Agent import BFS_Agent
from model.utils.Search_Tree import *
import utils.blockworld as blockworld
import random
import heapq
from dataclasses import dataclass, field
from typing import Any
from statistics import mean

#class for the priority queue
@dataclass(order=True)
class FringeNode:
    cost: int
    node: Any=field(compare=False)

class Astar_Agent(BFS_Agent):
    """An agent implementing the A* algorithm. The algorithm uses a fixed cost (so it tries to find the shortest path to the goal) and a given scoring function as heuristic to distance to goal. The heuristic should include stability.
    An upper limit can be set to prevent endless in difficult problems. -1 for potentially endless search.

    The heuristic should be admissible: it should be an upper bound to the actual cost of reaching the goal. 
    The h function estimates distance to goal by taking a heuristic, which should return degree of completion between 0 and 1, calculated the number of cells left to fill out and takes the average size of blocks in the library to provice an estimation of steps left to goal. Penalties get represented as really large distances. 
    """

    def __init__(self,world = None,heuristic=blockworld.recall, only_improving_actions = False, random_seed=None):
        self.world = world
        self.heuristic = heuristic
        self.only_improving_actions = only_improving_actions
        self.random_seed = random_seed
        if self.random_seed is None: self.random_seed = random.randint(0,99999)

    def __str__(self):
        """Yields a string representation of the agent"""
        return self.__class__.__name__+' heuristic: '+self.heuristic.__name__+' random seed: '+str(self.random_seed)
    
    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            'agent_type':self.__class__.__name__,
            'heuristic':self.heuristic.__name__,
            'random_seed':self.random_seed,
            }    

    def act(self,steps = None, verbose = False):
        """By default performs a full iteration of A*, then acts all the steps."""
        #check if we even can act
        if self.world.status()[0] != 'Ongoing':
            print("Can't act with world in status",self.world.status())
            return
        #preload values needed for heuristic
        self._silhouette_size = self.world.silhouette.sum() #the number of cells in the silhouette
        self._max_block_size = max([block.width * block.height for block in self.world.block_library])
        if steps is not None: print("Limited number of steps selected. This is not lookahead, are you sure?")
        #run A* search
        actions, states_evaluated = self.search(self.world.current_state, verbose)
        actions = actions[0:steps] #extract the steps to take. None gives complete list
        if verbose: 
            if actions == []: print("Found no solution")
            else: print("Found solution with ",len(actions),"actions")
        #apply steps to world
        for action in actions: self.world.apply_action(action)
        if verbose: print("Done, reached world status: ",self.world.status())
        #only returns however many steps we actually acted, not the entire sequence
        return actions,{'states_evaluated':states_evaluated} 

    def search(self,root,verbose=False):
        """Performs A* search"""
        i = 0 #iterations
        states_evaluated = 0 #track number of states evaluated
        #initialize open set
        open_set = open_set = Stochastic_Priority_Queue(random_seed=self.random_seed)
        #put in root node
        open_set.put(FringeNode(self.f(root),Node(root,[])))
        while not open_set.empty():
            i+=1
            #perform A* expansions
            node = open_set.get() #get node with lowest projected cost. This removes it from the open set
            if node is None:
                #empty means that the open set is empty
                break
            #check if that node is winning
            states_evaluated += 1
            # if verbose: print(node.state.blockmap) #DEBUG
            if node.state.is_win():
                #found winning node
                if verbose: print("Found winning state after",states_evaluated)
                return node.actions,states_evaluated
            #if node is not winning, add it's children to open set
            actions = node.state.possible_actions() #get possible actions
            for action in node.state.possible_actions():
                child = node.state.transition(action)
                open_set.put(FringeNode(self.f(child),Node(child,node.actions+[action])))
            # if verbose: print("added",len(actions),"new states at",i) #DEBUG
            if verbose and i%1000 == 0: print(i,"iterations, got open set of size",open_set.qsize(),"with cost",states_evaluated)
        if verbose: print("Evaluated all states and found nothing after",states_evaluated)
        return [],states_evaluated

    def f(self,node):
        """The combined cost function that takes into account the cost to get to the node (g) as well as the projected cost of reaching the goal (h).
        g is the number of blocks that have already been placed.
        h should be an estimation of the number of blocks that need to be placed before the silhouette is filled out.
        """
        return self.g(node) + self.h(node)

    def g(self,state):
        """Cost to get to state in steps"""
        return len(state.blocks)

    def h(self,state):
        """Estimated cost to get from current state to goal state.
        The heuristic should be admissible: it should be a lower bound to the actual cost of reaching the goal. The h function estimates distance to goal by taking a heuristic, which should return degree of completion between 0 and 1, calculated the number of cells left to fill out and takes the average size of blocks in the library to provice an estimation of steps left to goal. 
        """
        heur = self.heuristic(state)
        out = -(heur-1) * self._silhouette_size / self._max_block_size
        return max(0,out) #return 0 if the cost to goal is less than 0 (which doesn't make sense)

class Stochastic_Priority_Queue:
    """Implements a priority queue that randomly returns one of the elements which has the highest value as opposed to the one that was entered first (which Pythons priority queue does). Implemented using heapq. Follows Pythons priority_queue interface. Expect FringeNodes to be passed."""

    def __init__(self,random_seed=None,highest_first=False):
        self.random_seed = random_seed
        if self.random_seed is None: self.random_seed = random.randint(0,99999)
        self.heap = []
        self.head = None #first element of the list
        self.size = 0
        self.highest_first = highest_first

    def get(self):
        "Return the content of one of the elements that has lowest priority randomly"
        #find all elements with the lowest cost
        try:
            elems = [heapq.heappop(self.heap)]
        except IndexError:
            #empty heap
            return None
        min_priority = elems[0].cost
        while True:
            #keep popping items off the heap until we get the first item that has a higher priority 
            try:
                elem = heapq.heappop(self.heap)
                if elem.cost == min_priority: #save elements of equal priority
                    elems.append(elem)
                else: 
                    #we've found the first element that's less important
                    last_elem = elem
                    break
            except IndexError:
                last_elem = None
                break
        random.seed(self.random_seed) #fix random seed
        ret_index = random.randint(0,len(elems)-1) #index of element to return
        ret_item = elems[ret_index]
        #push the non returned items back onto the heap
        for elem in elems[0:ret_index] + elems[ret_index+1:] + [last_elem]:
            if elem is not None: heapq.heappush(self.heap,elem)
        return ret_item.node

    def put(self,elem: FringeNode):
        #if we actually want the highest value first, invert the cost
        if self.highest_first: elem.cost = elem.cost * -1
        #put onto heap
        heapq.heappush(self.heap,elem)

    def  empty(self):
        return len(self.heap) == 0

    def qsize(self):
        return len(self.heap)

