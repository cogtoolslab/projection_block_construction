"""This file contains decomposition functions as objects. The decompositions can additionally include a state object that the agent keeps track of and returns for the next decomposition (ie location of the construction paper for increments)."""

from utils.blockworld import legal
import numpy as np

class Decomposition_Function:
    def __init__(self, silhouette=None):
        self.silhouette = silhouette

    def set_silhouette(self,silhouette):
        self.silhouette = silhouette

    def get_decompositions(self, state = None):
        """Returns all possible decompositions as a dictionary with format {'decomposition': bitmap, 'name': name}"""
        pass

    def get_sequences(self,state=None,length=1,filter_for_length=False):
        """Generate a list of all legal (ie. only strictly increasing) sequences of subgoals up to *length* deep"""
        subgoals = self.get_decompositions(state=state)
        sequences = [[s] for s in subgoals]
        next_sequences = sequences #stores the sequences of the current length
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