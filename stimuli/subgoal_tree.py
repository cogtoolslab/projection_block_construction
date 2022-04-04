"""This class contains a class for a tree of subgoals that culminates in a full decomposition. This is for the human experiments involving A/B choice between subgoals"""

from nis import match
from typing import Dict
import model.utils.decomposition_functions as dcf
import random
import numpy as np

from utils.blockworld import Blockworld


class SubgoalTree:
    def __init__(self, root, world: Blockworld) -> None:
        self.root = root
        self.world = world

    def insert_sequence(self, sequence):
        """Takes a sequence and inserts it into the tree, creating nodes as necessary"""
        current_node = self.root
        for subgoal in sequence:
            # is the subgoal in the current children?
            matches = [c for c in current_node.children if c.__eq__(subgoal)]
            assert len(
                matches) < 2, "Subgoal entered as children should be unique"
            if len(matches) == 1:
                # great, nothing to do here
                current_node = matches[0]
                continue
            # otherwise insert it
            node = SubgoalTreeNode(subgoal=subgoal, parent=current_node)
            current_node.children.append(node)
            current_node = node

    def get_all_sequences(self):
        """Returns all sequences of subgoals ending in a leaf node"""
        sequences = []
        self.get_all_sequences_helper(self.root, [], sequences)
        return [dcf.Subgoal_sequence(s, prior_world=self.world) for s in sequences]

    def get_all_sequences_helper(self, node, sequence, sequences):
        """Recursive helper function for get_all_sequences"""
        if node.is_leaf():
            sequences.append(sequence)
        else:
            for child in node.children:
                self.get_all_sequences_helper(
                    child, sequence + [child], sequences)

    def get_full_sequences(self):
        """Returns all sequences of subgoals that lead to the full decomposition"""
        all_sequences = self.get_all_sequences()
        full_sequences = []
        for sequence in all_sequences:
            if sequence.fully_covering():
                full_sequences.append(sequence)
        return full_sequences

    def get_complete_sequences(self):
        """Returns all sequences that lead to the full decomposition and that we have actions for."""
        full_sequences = self.get_full_sequences()
        complete_sequences = []
        for sequence in full_sequences:
            if sequence.complete():
                complete_sequences.append(sequence)
        return complete_sequences

    def get_best_sequence(self):
        """Returns the cheapest sequence of subgoals that leads to the full decomposition"""
        sequences = self.get_complete_sequences()
        best_sequence = None
        best_cost = None
        for sequence in sequences:
            cost = 0
            for subgoal in sequence:
                try:
                    cost += subgoal.C
                except TypeError:
                    # no cost for a subgoal means that the subgoal cannot be solved—skip this sequence
                    continue
            if best_cost is None or cost < best_cost:
                best_sequence = sequence
                best_cost = cost
        return best_sequence

    def get_worst_sequence(self):
        """Returns the most expensive sequence of subgoals that leads to the full decomposition"""
        sequences = self.get_complete_sequences()
        worst_sequence = None
        worst_cost = None
        for sequence in sequences:
            cost = 0
            for subgoal in sequence:
                try:
                    cost += subgoal.C
                except TypeError:
                    # no cost for a subgoal means that the subgoal cannot be solved—skip this sequence
                    continue
            if worst_cost is None or cost > worst_cost:
                worst_sequence = sequence
                worst_cost = cost
        return worst_sequence


class SubgoalTreeNode:
    """Holds a particular configuration of the building environment. A node in the tree."""

    def __init__(self, subgoal, parent, children=None) -> None:
        self.subgoal = subgoal
        if subgoal is None:  # there is no subgoal associated with the root node
            self.target = None
            self.cost = 0
        else:
            self.cost = self.subgoal.C
            self.target = self.subgoal.target
        self.parent = parent
        if children is None:
            # we can't assign the empty list in the constructur signature because then all instance refer to the same shared list.
            self.children = []
        else:
            self.children = children

    def __eq__(self, other):
        """This should perform a comparision of the targets of the subgoals"""
        return np.all(np.equal(self.target, other.target))

    def is_leaf(self):
        """Returns true if this node is a full decomposition"""
        return len(self.children) == 0

    def best_subgoal(self):
        assert len(self.subgoals >= 2), "Not enough potential subgoals"
        min_cost = min([sg.cost for sg in self.subgoals])
        # we do random choice between equally good subgoals here
        return random.choice([sg for sg in self.subgoals if sg.cost == min_cost])

    def worst_subgoal(self):
        assert len(self.subgoals >= 2), "Not enough potential subgoals"
        max_cost = max([sg.cost for sg in self.subgoals])
        # we do random choice between equally good subgoals here
        return random.choice([sg for sg in self.subgoals if sg.cost == max_cost])
