"""Holds helper functions for simple search tree over states of the blockworld as nodes."""
import copy

class Node:
    """A node holds a state (a blockworld state object) and a list of actions that lead up to the state"""
    def __init__(self,state,actions):
        self.state = copy.copy(state)
        self.actions = actions