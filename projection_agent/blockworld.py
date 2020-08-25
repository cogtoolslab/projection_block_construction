from world import World

import numpy as np
from PIL import Image

from matplotlib import pylab, mlab, pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
import matplotlib.patches as patches
import copy
import json
import datetime
import random
from random import randint
import string
import os

import blockworld_helpers
import display_world

import sys

class Blockworld(World):
    """This class implements the blockworld as defined in https://github.com/cogtoolslab/block_construction. It manages state, allows for transition and scoring function of states both by F1 score and stability. Stability is calculated using the box2d as opposed to matter in the browser version. The stable/unstable distinction in the cases here is simple enough that differences between physics engines should not matter. 

    Fast failure means ending returning failure for states in which the agent has built outside the silhouette or left holes that can't be filled. Enable this to spare the agent the misery of having to fill the map with blocks if it can't win anymore. 
    
    All the important functions are implemented by the State class, which represents a particular configuration of the world. The world merely manages the current state and wraps around the functions of the class State to keep compatibility with previous uses of the class.
    
    Default values are chosen to correspond to the silhouette 2 study, see block_construction/experiments/silhouette_2.
    
    Dimensions are in y,x. The origin is top left (in accordance with numpy arrays."""

    def __init__(self,dimension = None,silhouette=None,block_library = None,fast_failure=False):
        self.dimension = dimension
        #Defines dimensions of possible blocks. 
        if block_library is None: #the default block library is the one from the silhouette 2 study
            block_library =  [
            BaseBlock(1,2),
            BaseBlock(2,1),
            BaseBlock(2,2),
            BaseBlock(2,4),
            BaseBlock(4,2),
            ]       
        self.block_library = block_library 
        #load the target silhouette as numpy array
        self.silhouette = self.load_silhouette(silhouette) 
        self.current_state = Blockworld.State(self,[]) #generate a new state with no blocks in it
        self.fast_failure = fast_failure #set world state to fail if the agent has built outside the silhouette or left a hole

    def __str__(self):
        """String representation of the world"""
        return self.__class__.__name__

    def  transition(self,action,state=None):
        """Takes an action and a state and returns the resulting state without applying it to the current state of the world."""
        if state is None:
            state = self.current_state
        # if action not in self.possible_actions(state): #this might be slow and could be taken out for performance sake
            # print("Action is not in possible actions")
        #determine where the new block would land
        baseblock, x = action #unpacking the action
        if baseblock is None:
            return state
        #check for legality of transition
        if action not in self.current_state.possible_actions(): raise Exception("Action not possible") 
        #determine y coordinates of block
        y = 0
        while y < self.dimension[0] and state.blockmap[y,range(x, x + baseblock.width)].sum() ==  0: #find the lowest free row in the tower 
            y+= 1
        y = y-1 #because y marks the first full row
        #create new block
        new_block = Block(baseblock,x,y)
        #create new state
        new_state = Blockworld.State(self,state.blocks + [new_block])
        return new_state

    def status(self):
        """Expanded status function also returns a reason for failure"""
        if self.is_win():
            return "Win","None"
        if self.is_fail():
            if self.stability() == False:
                return "Fail","Unstable"
            if self.current_state.possible_actions() == []:
                return "Fail","Full"
            #the two conditions below are for fast failure only
            if filled_outside(self.current_state) > 0:
                return "Fail","Outside"
            if holes(self.current_state) > 0:
                return "Fail","Holes"
        return "Ongoing","None"

    def load_silhouette(self,silhouette):
        if type(silhouette) is np.ndarray:
            if silhouette.shape == self.dimension:
                return silhouette
            else:
                # print("Silhouette dimensions", silhouette.shape, "don't match world dimensions. Setting world dimensions to match.")
                self.dimension = silhouette.shape
                return silhouette
        elif silhouette is None: #No silhouette returns an empty field.
            return np.zeros(self.dimension)
        else:
            #TODO implement importing from file
            raise Warning("Use other function in file to import from file")


    """Simple functions inherited from the class World"""
    def apply_action(self,action):
        if action not in self.current_state.possible_actions(): raise Exception("Action not possible")
        self.current_state = self.transition(action,self.current_state)

    def is_fail(self,state = None):
        if state is None:
            state = self.current_state
        if self.fast_failure: #should we fail the trial if it can't be completed to save time? Suggested by David. 
            if filled_outside(self.current_state) > 0:
                return True
            if holes(self.current_state) > 0:
                return True
        #always active fail states
        if state.stability() is False: #we loose if its unstable
            return True
        if state.score(F1score) != 1 and state.possible_actions() == []: #we loose if we aren't finished and have no options
            return True

        return False

    def is_win(self,state = None):
        if state is None:
            state = self.current_state
        if state.score(F1score) == 1 and state.stability():
            return True
        return False

    def score(self,state=None,scoring_function=None):
        if state is None:
            state = self.current_state
        if self.is_fail(state):
            return self.fail_penalty
        if self.is_win(state):
            return self.win_reward
        return state.score(scoring_function)

    def F1score(self,state=None):
        if state is None:
            state = self.current_state
        return state.score(F1score)

    def possible_actions(self,state=None):
        if state is None:
            state = self.current_state
        return state.possible_actions()

    def stability(self,state=None):
        if state is None:
            state = self.current_state
        return state.stability()



    class State():
        """This subclass contains a (possible or current) state of the world and implements a range of functions to score it, namely for F1 (how much of the figure are we filling out) and physical stability. It also generates possible actions. The blockworld classes wrap around this class.
        Hashes of this class are orderinvariant as to the order the blocks were placed (but not their positions). Use 'order_sensitive_hash' for an hash that takes placement order of blocks into account."""
        def __init__(self,world,blocks):
            self.world = world
            self.blocks = blocks
            self.world_width = self.world.dimension[1]
            self.world_height = self.world.dimension[0]
            self.blockmap = np.zeros((self.world_height, self.world_width),dtype=int) ## bitmap for placement of blocks
            self._update_map_with_blocks(blocks) #read the blocks into the blockmap
            self._stable = None
            self._cached_hash = None #Cached hash value. It's only filled once we actually generate a hash and invalidates when the blockmap is updated. ⚠️ It is NOT updated when the blockmap/block list is touched manually! ⚠️
        
        def __eq__(self,other):
            """The order of the blocks does not matter, as they have their location attached. So the sorted list should be equal between two states which consist of the same blocks no matter the order in which they were placed"""
            return self.order_invariant_hash() == other.order_invariant_hash()
        
        def __hash__(self):
                return self.order_invariant_hash()

        def clear(self):
            """Clears the various cached values of the state and updates the blockmap."""
            caches = [self._F1score,self._stable,self._cached_hash]
            for cache in caches:
                try:
                    del(cache)
                except:
                    pass
            self._update_map_with_blocks(self.blocks)

        def order_invariant_blockmap(self):
            """Returns an np.array of the blockmap that ignores the order in which the blocks were placed, ie outputs the same blockmap for all the states that have the same kinds of blocks in the same locations, but not the same order in which these blocks have been placed.
            Use (A==B).all() to compare two numpy arrays for equality."""
            return self._get_new_map_from_blocks(self.order_invariant_blocks())
        
        def order_invariant_blocks(self):
            """Returns ordered list of blocks that have been placed to ensure the same order in the list even if they have been placed in a different order."""
            def _block_key(block):
                """Helper function returns a float to sort blocks by y location first, then x location."""
                return float(block.y + .5/(block.x+1))
            return sorted(self.blocks,key=_block_key,reverse=True)
        
        def order_invariant_hash(self):
            """String of order invariant blockmap"""
            if self._cached_hash is None: #create hash if necessary
                self._cached_hash = self.order_invariant_blockmap().tostring() #Long hex string
                # self._cached_hash = self.order_invariant_blockmap().__str__() #Slower, but human readable
            return self._cached_hash

        def order_sensitive_hash(self):
            return self.blockmap.tostring()

        def _update_map_with_blocks(self, blocks, delete=False):
            """Fills the blockmap with increasing numbers for each block. 0 is empty space. Original blockmap behavior can be achieved by blockmap > 0."""
            self._cached_hash = None #invalidate the hash
            for b in blocks:
                new_number = 0 if delete else (np.max(self.blockmap)+1) #numbers increase 
                self.blockmap[(b.y-b.height)+1: b.y+1, b.x:(b.x+b.width)] = new_number 
        
        def _get_new_map_from_blocks(self, blocks):
            """Same as _update_map_with_blocks, only that it returns a new blockmap without touching the blockmap of the state."""
            blockmap = np.zeros((self.world_height, self.world_width),dtype=int)
            i = 0
            for b in blocks:
                i+=1
                blockmap[(b.y-b.height)+1: b.y+1, b.x:(b.x+b.width)] = i 
            return blockmap

        def score(self,scoring_function):
            """Returns the score according to the the scoring function that is passed. """
            return scoring_function(self)

        def stability(self,visual_display=False):
            """Runs physics engine to determine stability. Returns true or false, but could be adapted for noisy simulations. Caches it's value."""
            if self._stable is not None:
                #return cached value
                return self._stable
            bwworld = self.state_to_bwworld()
            self._stable =  display_world.test_world_stability(bwworld,RENDER=visual_display)  == 'stable'
            return self._stable

        def possible_actions(self):
            """Generates all actions that are possible in this state independent of whether the block is stable or within the silhouette. Simply checks whether the block is in bounds. 
            Format of action is (BaseBlock from block_library, x location of lower left)."""
            possible_actions = []
            for base_block in self.world.block_library:
                for x in range(self.world_width): #starting coordinate is bottom left
                    #check whether it overlaps the left boundary
                    if x + base_block.width <= self.world_width:
                         #and whether it overlaps the top boundary by simply looking if the block at the top is free
                        if self.blockmap[0 : base_block.height, x : x+base_block.width] .sum() == 0:
                            possible_actions.append((base_block,x))
            return possible_actions

        def visual_display(self,blocking=False,silhouette=None):
            """Shows the state in a pretty way."""
            pyplot.close('all')
            pyplot.pcolormesh(self.blockmap[::-1], cmap='hot_r',vmin=0,vmax=20)
            if silhouette is not None:
                #we print the target silhouette as transparent overlay
                pyplot.pcolormesh(silhouette[::-1], cmap='Greens',alpha=0.20,facecolors='grey',edgecolors='black')
            pyplot.show(block=blocking)

        def state_to_bwworld(self):
            """converts state to Blockworld.world.
                Helper function for block_construction/stimuli/blockworld_helpers.py testing code"""
            wh = self.world.dimension[0] #world height
            bwworld = blockworld_helpers.World(world_height=self.world.dimension[0],world_width=self.world.dimension[1])
            for b in self.blocks:
                bwworld.add_block(b.width,b.height,b.x,wh-b.y-1)
            return bwworld

class Block:
    '''
        Adapted from block_construction/stimuli/blockworld_helpers.py
        Creates Block objects that are instantiated in a world
        x and y define the position of the BOTTOM LEFT corner of the block
        
        Defines functions to calculate relational properties between blocks
    '''
    def __init__(self, base_block, x, y):
        self.base_block = base_block # defines height, width and other functions
        #bottom left coordinate
        self.x = x
        self.y = y
        self.height = base_block.height
        self.width = base_block.width
        self.verts = base_block.translate(base_block.base_verts,x,y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.height == other.height and self.width == other.width

    def __str__(self):
        """width, height at (x,y)"""
        return "{}x{} at ({},{})".format(self.width,self.height,self.x,self.y)
    
    #Block Relational Properties
    def above(self, other):
        ''' Test whether this block is fully above another block.
        
            Returns true iff the height of the bottom face of this block 
            is greater than or equal to the top face of other block.
        '''
        return (self.y >= other.y + other.height)
    
    def below(self, other):
        ''' Test whether this block is fully below another block.
            
            Returns true iff the height of the top face of this block 
            is less than or equal to the bottom face of other block.
        '''
        return (self.y + self.height <= other.y)
    
    def leftof(self, other):
        ''' Test whether this block is fully to the left of another block
        
            Returns true iff the height of the bottom face of this block 
            is greater than or equal to the top face of other block.
        '''
        return (self.x + self.width <= other.x)
    
    def rightof(self, other):
        ''' Test whether this block is fully to the right of another block.
            
            Returns true iff the height of the top face of this block 
            is less than or equal to the bottom face of other block.
        '''
        return (self.x >= other.x + other.width)
    
    def sides_touch(self, other):
        ''' Test to see whether this block sits touching sides of another block.
            Corner to corner treated as not touching.
        '''
        y_overlap = not self.above(other) and not self.below(other) 
        buttressing_side = self.x == other.x + other.width or other.x == self.x + self.width
        return y_overlap and buttressing_side
    
    def vertical_touch(self, other):
        ''' Test to see whether this block sits top to bottom against other block or bottom to top against other block.
            Corner to corner treated as not touching.
        '''
        x_overlap = not self.leftof(other) and not self.rightof(other) 
        buttressing_up = self.y == other.y + other.height or other.y == self.y + self.height
        buttressing_down = self.y == other.y - other.height or other.y == self.y - self.height
        buttressing = buttressing_down or buttressing_up
        return x_overlap and buttressing
    
    def touching(self, other):
        ''' Test to see if this block is touching another block.
            Corner to corner treated as not touching.
        '''
        return self.sides_touch(other) or self.vertical_touch(other)        

    def abs_overlap(self, other, horizontal_overlap=True):
        ''' horizontal- are we measuring horizontal overlap?
        '''
        if horizontal_overlap and self.vertical_touch(other):
            return min([abs(self.x + self.width - other.x), abs(other.x + other.width - self.x)])
        elif not horizontal_overlap and self.sides_touch(other):
            return min([abs(self.y + self.height - other.y), abs(other.y + other.height - self.y)])
        else:
            return 0
        
    def partially_supported_by(self, other):
        ''' True if the base of this block is touching the top of the other block 
        '''
        return self.above(other) and (self.abs_overlap(other) > 0)
    
    def completely_supported_by(self, other):
        ''' True if the whole of the base of this block is touching the top of the other block 
        '''
        return self.above(other) and (self.abs_overlap(other) == self.width)
    
    def dual_supported(self, block_a, block_b):
        ''' Is this block partially supported by both blocks a and b?
        '''
        return self.partially_supported(block_a) and self.partially_supported(block_b)
    
    
    '''
    Other useful properties:
    - 
    
    '''

class BaseBlock:
    '''
    Base Block class for defining a block object with attributes.             
    Adapted from block_construction/stimuli/blockworld_helpers.py
    '''
    
    def __init__(self, width=1, height=1, shape='rectangle', color='gray'):
        self.base_verts = np.array([(0, 0), 
                            (0, 1 * height), 
                            (1 * width, 1 * height), 
                            (1 * width, 0), 
                            (0,0)])
        self.width = width
        self.height = height
        self.shape = shape
        self.color = color
    
    def __str__(self):
        return('('+str(self.width) + 'x' + str(self.height)+')')

    def init(self):
        self.corners = self.get_corners(self.base_verts)
        self.area = self.get_area(shape=self.shape) 
        
    def translate(self, base_verts, dx, dy):
        '''
        input:
            base_verts: array or list of (x,y) vertices of convex polygon. 
                    last vertex = first vertex, so len(base_verts) is num_vertices + 1
            dx, dy: distance to translate in each direction
        output:
            new vertices
        '''
        new_verts = copy.deepcopy(base_verts)
        new_verts[:,0] = base_verts[:,0] + dx
        new_verts[:,1] = base_verts[:,1] + dy
        return new_verts

    def get_corners(self,base_verts):
        '''
        input: list or array of block vertices in absolute coordinates
        output: absolute coordinates of top_left, bottom_left, bottom_right, top_right
        '''
        corners = {}
        corners['bottom_left'] = verts[0]
        corners['top_left'] = verts[1]
        corners['top_right'] = verts[2]
        corners['bottom_right'] = verts[3]
        return corners

    def get_area(self,shape='rectangle'):
        '''
        input: w = width 
            h = height           
            shape = ['rectangle', 'square', 'triangle']
        output
        '''
        ## extract width and height from dims dictionary 
        if shape in ['rectangle','square']:
            area = self.width*self.height
        elif shape=='triangle':
            area = self.width*self.height*0.5
        else:
            print('Shape type not recognized. Please use recognized shape type.')
        return area   


"""Scoring functions. These should be passed to the scoring function of the state. Note that these operate on the blockmap, not the blocks."""

def F1score(state):
    """Returns the F1 score relative to the target silhouette defined in the corresponding world. If the silhouette is empty, this produces division by 0 errors and returns NaN."""
    if hasattr(state,'_F1score'):
        return state._F1score
    s = sys.float_info[3] #smallest possible float to prevent division by zero. Not the prettiest of hacks
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    precision = np.sum(built & target)/(np.sum(built) + s) 
    recall = np.sum(built & target)/(np.sum(target) + s) 
    F1score = 2 * (precision * recall)/(precision + recall + s)
    state._F1score = F1score
    return F1score

def precision(state):
    s = sys.float_info[3] #smallest possible float to prevent division by zero. Not the prettiest of hacks
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    return np.sum(built & target)/(np.sum(built) + s) 

def recall(state):
    s = sys.float_info[3] #smallest possible float to prevent division by zero. Not the prettiest of hacks
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    return np.sum(built & target)/(np.sum(target) + s) 

def weighted_precision_recall(state,precision_weight=1):
    """Simply the weighted average of precision and recall. GIve a higher weigth to precision discourage building outside the structure."""
    return (precision * precision_weight + recall)/(precision_weight+1)

def filled_inside(state):
    """Returns the number of cells built in inside the silhouette"""
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    return np.sum(built & target)

def filled_outside(state):
    """Returns the number of cells built outside the silhouette"""
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    return np.sum(built & (1-target))

def silhouette_score(state):
    """Returns a score that encourages the filling out of the silhouette gives penalty for placing blocks outside of it. 1 if silhouette perfectly filled, penalty for building outside weighted by size of silhouette."""
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    ssize = np.sum(target)
    reward = np.sum(built & target)/ssize
    penalty = np.sum(built & (1-target))/ssize
    return reward + penalty * state.world.fail_penalty

def random_scoring(state):
    """Implements the random agent. Returns 1 for every block placement that is in the silhouette and -1 otherwise."""
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    if np.sum((1-target) & built) > 1:
        return -1
    else:
        return 1

def legal(state):
    """Implements the random agent. Returns 1 for every block placement that is in the silhouette and -1 otherwise."""
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    if np.sum((1-target) & built) > 1:
        return False
    else:
        return True

def holes(state):
    """Returns the number of holes in the structure that are covered with a block to prevent the model from creating holes it cannot fill."""
    target = state.world.silhouette > 0
    built = (state.blockmap > 0) * 2
    mapped = target + built
    holes = 0
    for x in range(built.shape[1]): # we don't need to check the bottom
        for y in range(built.shape[0]-1):
            if mapped[y,x] == 3: #if we have a cell with blck and in silhouette
                 #if blocks below is not built and in silhouette
                holes = holes + np.sum(mapped[y+1:,x] == 1)
                break #since we're going through from the top, we don't need to iterate further
    return holes

def silhouette_hole_score(state):
    """Implements the heuristic that the agent should not cover empty space in the silhouette because it later can't build there"""
    return silhouette_score(state) + state.world.fail_penalty * holes(state)

def F1_stability_score(state):
    """Returns F1 with check for stability."""
    return F1score(state) + state.world.fail_penalty * (1 - state.stability())

def silhouette_hole_stability_score(state):
    """Silhouette & hole heuristics with stability"""
    return silhouette_hole_score(state) + state.world.fail_penalty * (1 - state.stability())

def sparse(state):
    """Returns 1 if the silhouette is perfectly built & stable and 0 otherwise."""
    return float(state.world.is_win(state))