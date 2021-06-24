"""This file holds helper functions designed for the human experiments."""

# set up imports
import os
import sys
import random
from typing import List, Set, Tuple
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(proj_dir)
utils_dir = os.path.join(proj_dir,'utils')
sys.path.append(utils_dir)
analysis_dir = os.path.join(proj_dir,'analysis')
analysis_utils_dir = os.path.join(analysis_dir,'utils')
sys.path.append(analysis_utils_dir)
experiments_dir = os.path.join(proj_dir,'experiments')
sys.path.append(experiments_dir)

from model.utils.decomposition_functions import *

from itertools import permutations as permutate
from collections import OrderedDict

MAX_SUBGOALS = 4 #maximum number of subgoals per sequence
FINAL_HEIGHT = 8 #height of the building area

def get_scored_sequences(list_of_list_of_sequences: List) -> OrderedDict:
    """Extract the scores for all sequences from the list of list of sequences (where the outer list is over different runs and the inner list is the list of ALL subgoal sequences for that run).
    
    Pass the list of all possible sequences, not just the solved ones.
    
    Will return a list of total search cost for all sequences that a solution was found for every time."""
    #filter long sequences
    list_of_list_of_sequences = filter_sequences(list_of_list_of_sequences)
    if len(list_of_list_of_sequences) == 0 or sum([len(s) for s in list_of_list_of_sequences]) == 0:
        raise Warning("No solved sequences")
        return None
    #initialize scores
    sequence_scores = {name:0 for name in [s.names() for s in list_of_list_of_sequences[0]]}
    sequence_complete = {name:True for name in [s.names() for s in list_of_list_of_sequences[0]]}
    #compute scores
    for list_of_sequences in list_of_list_of_sequences:
        for sequence in list_of_sequences:
            sequence_scores[sequence.names()] += sequence.planning_cost()
            sequence_complete[sequence.names()] = False if not sequence.complete() else sequence_complete[sequence.names()]
    sequence_filtered = {name:score for name,score in sequence_scores.items() if sequence_complete[name]}
    #sort the dictionary
    sequence_sorted = OrderedDict(sorted(sequence_filtered.items(), key = lambda pair: pair[1]))
    return sequence_sorted

def filter_sequences(list_of_list_of_sequences: List[list]) -> List[list]:
    """Filters out sequences that are too long and that don't end on the full reconstruction"""
    return [[seq for seq in list_of_sequences if len(seq.subgoals) <= MAX_SUBGOALS and seq.names()[-1] == FINAL_HEIGHT] for list_of_sequences in list_of_list_of_sequences]

def permutations(names: Tuple) -> Set:
    """Returns a list of all permutations of a subgoal sequence as names, excluding the main sequence"""
    _names = [0] + list(names)
    increments = [abs(_names[i] - _names[i+1]) for i in range(len(_names)-1)]
    permutated_increments = list(permutate(increments))
    permutated_names = [tuple([abs(sum(pi[0:i]) + pi[i]) for i in range(len(pi))]) for pi in permutated_increments]
    permutated_names = set([pn for pn in permutated_names if pn != names]) #remove duplicates and original sequence
    return permutated_names
    
def extract_best_and_worst_sequence(sequence_scores: OrderedDict):
    """Find the best sequence and the worst sequence that is a permutation of the best sequence. If no permutation of the best sequence is possible, the next best sequence that has a solvable permutation is chosen.
    
    Assumes sorted dictionary"""
    for best_name,best_score in sequence_scores.items():
        best_permutations = permutations(best_name)
        sequence_permutations = {p_name:sequence_scores[p_name] for p_name in best_permutations if p_name in sequence_scores.keys()} 
        #sort permutations
        sequence_permutations = OrderedDict(sorted(sequence_permutations.items(), key=lambda pair: pair[1]))
        if len(sequence_permutations) == 0:
            # no permutations could be solved
            continue
        #we have found a sequence and permutated worst sequence
        worst_name, worst_score = list(sequence_permutations.items())[-1]
        return best_name, worst_name, worst_score - best_score, best_score, worst_score

def extract_most_divergent_pair_sequence(sequence_scores: OrderedDict):
    """Find the sequence such that the distance between best and worst sequence is maximized"""
    pair_scores = {}
    for first_name,first_score in sequence_scores.items():
        second_names = permutations(first_name)
        sequence_permutations = {p_name:sequence_scores[p_name] for p_name in second_names if p_name in sequence_scores.keys()} 
        #sort permutations
        sequence_permutations = OrderedDict(sorted(sequence_permutations.items(), key=lambda pair: pair[1]))
        if len(sequence_permutations) == 0:
            # no permutations could be solved
            continue
        #we have found a sequence and permutated worst sequence
        worst_name, worst_score = list(sequence_permutations.items())[-1]
        pair_scores[tuple([first_name,worst_name])] = (first_score, worst_score)
    #sort pair_scores
    # print(pair_scores)
    pair_scores = OrderedDict(sorted(pair_scores.items(), key = lambda pair: pair[1][1] - pair[1][0], reverse=True))
    #extract most divergent pair
    try:
        best_name, worst_name = list(pair_scores.items())[0][0]
        delta = list(pair_scores.items())[0][1][1] - list(pair_scores.items())[0][1][0]
        best_score = list(pair_scores.items())[0][1][0]
        worst_score = list(pair_scores.items())[0][1][1]
    except IndexError:
        # didn't find any pairs
        return None        
    return best_name, worst_name, delta, best_score, worst_score


def get_best_and_worst_sequence(list_of_list_of_sequences: List[list]):
    """Find the best sequence and the worst sequence that is a permutation of the best sequence. If no permutation of the best sequence is possible, the next best sequence that has a solvable permutation is chosen. Pass a list of lists.
    
    Call this function from notebook.
    """
    scored_sequences = get_scored_sequences(list_of_list_of_sequences)
    return extract_best_and_worst_sequence(scored_sequences)

def get_most_divergent_pair_sequence(list_of_list_of_sequences: List[list]):
    """Finds the sequence where the best and worst sequence diverge the most.

    Call this function from notebook."""
    scored_sequences = get_scored_sequences(list_of_list_of_sequences)
    return extract_most_divergent_pair_sequence(scored_sequences)    