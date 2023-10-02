"""This class contains a class for a tree of subgoals that culminates in a full decomposition. This is for the human experiments involving A/B choice between subgoals"""

from nis import match
from typing import Dict
import scoping_simulations.model.utils.decomposition_functions as dcf
import random
import matplotlib.pyplot as plt
import numpy as np

from scoping_simulations.utils.blockworld import Blockworld


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

    def get_most_divergent_pairs_of_subgoals(self, n = None):
        """Returns the n most divergent pairs of subgoals in the tower as tuples. Each pair is based on the same node. Returns a sorted list of tuples (best, worst) of nodes."""
        open_set = [self.root]
        best_worst_pairs = []
        while len(open_set) > 0:
            current_node = open_set.pop()
            best_child = current_node.best_child()
            worst_child = current_node.worst_child()
            if best_child is not None and worst_child is not None and best_child.cost != worst_child.cost: # only add if we truly have different subgoals
                best_worst_pairs.append((best_child, worst_child))
            for child in current_node.children:
                open_set.append(child)
        if n is None:
            n = len(best_worst_pairs)
        return sorted(best_worst_pairs, key=lambda x: x[1].cost - x[0].cost)[:min(n, len(best_worst_pairs))]
    
    def get_most_divergent_matching_pairs_of_subgoals(self, n = None):
        """Returns the n most divergent pairs of subgoals that have the same mass of the tower in the tower as tuples. Each pair is based on the same node. Returns a sorted list of tuples (best, worst) of nodes."""
        open_set = [self.root]
        best_worst_pairs = []
        while len(open_set) > 0:
            current_node = open_set.pop()
            # make dict of children with mass
            children_with_mass = {}
            for child in current_node.children:
                try:
                    children_with_mass[child.subgoal.R()] += [child]
                except:
                    children_with_mass[child.subgoal.R()] = [child]
        for mass, children in children_with_mass.items():
            children = [c for c in children if c.subgoal.solution_cost is not None]
            if len(children) > 1:
                # find the most divergent pair of children for that mass
                children = sorted(children, key=lambda x: x.subgoal.solution_cost)
                if children[0].subgoal.solution_cost != children[-1].subgoal.solution_cost:
                    best_worst_pairs.append((children[0], children[-1]))
        return best_worst_pairs

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

    def best_child(self):
        """Returns the best (cheapest) child of the node."""
        if self.is_leaf(): # no children for leaf nodes
            return None
        try:
            min_cost = min([child.cost for child in self.children if child.cost is not None])
        except ValueError:
            # no children with a cost found
            return None
        # we do random choice between equally good subgoals here
        return random.choice([child for child in self.children if child.cost == min_cost])

    def worst_child(self):
        """Returns the worst (most expensive) child of the node."""
        if self.is_leaf(): # no children for leaf nodes
            return None
        try:
            max_cost = max([child.cost for child in self.children if child.cost is not None])
        except ValueError:
            # no children with a cost found
            return None
        # we do random choice between equally good subgoals here
        return random.choice([child for child in self.children if child.cost == max_cost])

    def visualize(self, block_color = None):
        """Plots current subgoal and the children"""
        fig = plt.figure()
        if self.subgoal is not None:
            self.subgoal.visualize(title="Node subgoal", fig=fig, block_color = block_color)
        for i,child in enumerate(self.children):
            child.visualize(title="Child {}".format(i), fig=fig, block_color = block_color)
        plt.show()

            
