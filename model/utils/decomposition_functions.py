"""This file contains decomposition functions as objects. The decompositions can additionally include a state object that the agent keeps track of and returns for the next decomposition (ie location of the construction paper for increments)."""

import copy
import numpy as np


class Subgoal:
    """Stores a subgoal"""
    def __init__(self,prior_world,target,name=None,actions=None,C=None,past_world=None,solution_cost=None,planning_cost=None):
        self.prior_world = copy.deepcopy(prior_world)
        self.target = copy.deepcopy(target)
        self.name = name
        self.actions = actions
        self.C = C
        self.past_world = copy.deepcopy(past_world)
        self.solution_cost = solution_cost
        self.planning_cost = planning_cost
    try:
        self.R = self.R()
    except:
        pass

    def key(self):
        key = self.target * 100 - (self.prior_world.current_state.order_invariant_blockmap() > 0)
        return key.tostring() #make hashable

    def R(self):
        try:
            return np.sum((self.past_world.current_state.blockmap > 0)*1. - (self.prior_world.current_state.blockmap > 0)*1.)
        except AttributeError:
            return 0 #if we can't solve it (or haven't yet), we return a reward of 0

class Subgoal_sequence:
    """Stores a sequence"""
    def __init__(self,sequence,prior_world):
        """Generate sequence from dict input from decomposition function"""
        self.subgoals = []
        self.score = None
        for d in sequence:
            subgoal = Subgoal(prior_world=prior_world,target=d['decomposition'],name = d['name'])
            self.subgoals.append(subgoal)

    def names(self):
        return [sg.name for sg in self.subgoals]
    
    def actions(self):
        actions = []
        for sg in self.subgoals:
            try:
                actions += sg.actions
            except: #we don't have any actions (if its unsolvable,...)
                pass
        return actions

    def V(self,c_weight=1):
        """Adds up the cost and rewards of the subgoals"""
        score = sum([sg.R() - sg.C * c_weight 
            if sg.C is not None else
                sg.R() - 0 * c_weight #if we havent scores the cost yet, it's 0, but there should be an unreachable penalty somewhere leading up
            for sg in self.subgoals])
        return score

    def planning_cost(self):
        """Planning cost of the entire sequence"""
        return sum([sg.planning_cost if sg.planning_cost is not None else 0 for sg in self.subgoals]) #could be max cost as well
    
    def solution_cost(self):
        """Planning cost of the entire sequence"""
        return sum([sg.solution_cost if sg.solution_cost is not None else 0 for sg in self.subgoals]) #could be max cost as well
    
    def complete(self):
        """Do we have a solution for all goals in the sequence?"""
        return np.all([s.actions is not None and s.actions != []  for s in self.subgoals])

    def __len__(self):
        return len(self.subgoals)
    
    def __iter__(self):
        self.a = 0
        return self
    
    def __next__(self):
        try:
            sg = self.subgoals[self.a]
            self.a += 1
            return sg
        except IndexError:
            raise StopIteration

class Decomposition_Function:
    def __init__(self, silhouette=None):
        self.silhouette = silhouette

    def set_silhouette(self,silhouette):
        self.silhouette = silhouette

    def get_decompositions(self, state = None):
        """Returns all possible decompositions as a dictionary with format {'decomposition': bitmap, 'name': name}"""
        pass

    def get_sequences(self,state=None,length=1,filter_for_length=True):
        """Generate a list of all legal (ie. only strictly increasing) sequences of subgoals up to *length* deep"""
        subgoals = self.get_decompositions(state=state)
        sequences = [[s] for s in subgoals]
        next_sequences = sequences.copy() #stores the sequences of the current length
        for l in range(length-1):
            current_sequences = next_sequences
            next_sequences = []
            for current_sequence in current_sequences:
                for subgoal in subgoals:
                    if self.legal_next_subgoal(current_sequence[-1],subgoal):
                        #if we can do the subgoal after the current sequence, include it
                        new_sequence = current_sequence+[subgoal]
                        sequences.append(new_sequence)
                        next_sequences.append(new_sequence)
        if filter_for_length: 
            sequences = self.filter_for_length(sequences,length)
        #turn into objects
        sequences = [Subgoal_sequence(s,state.world) for s in sequences]
        return sequences
    
    def filter_for_length(self,sequences,length):
        """Filter out sequences that don't have the required length *unless* they end with the full decomposition (since if the sequence ends, we can only reach the full decomp in lucky cases)"""
        return [s for s in sequences if len(s) == length or (len(s) <= length and np.all(s[-1]['decomposition'] == self.silhouette))]

    def get_name(self):
        return type(self).__name__

    def get_params(self):
        """Returns dict of parameters"""
        return {}
  
    def legal_next_subgoal(self,before, after):
        """Check if the after subgoal can follow the before, ie is a real improvement and subsumes the before"""
        return np.all(after['decomposition']-before['decomposition'] >= 0) and np.sum(after['decomposition']-before['decomposition']) > 0


class Horizontal_Construction_Paper(Decomposition_Function):
    """Horizontal construction paper. Returns all positions irregardless of state."""    
    def __init__(self, silhouette):
        super().__init__(silhouette)

    def get_decompositions(self, state = None):
        decompositions = []
        for y in range(self.silhouette.shape[0]):
            decomposition = np.copy(self.silhouette)
            decomposition[0:y,:] = 0
            decompositions += [{'decomposition':decomposition,'name':abs(self.silhouette.shape[0] - y)}]
        decompositions.reverse()
        return decompositions

