import utils.display_world as display_world
import utils.blockworld_helpers as blockworld_helpers
import socketio
import subprocess
import utils.matter_server as matter_server
import string
from random import randint
import random
import time
import datetime
import json
import copy
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import pylab, mlab, pyplot
from PIL import Image
import numpy as np
from utils.world import World
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_dir)


class Blockworld(World):
    """This class implements the blockworld as defined in https://github.com/cogtoolslab/block_construction. It manages state, allows for transition and scoring function of states both by F1 score and stability. Stability is calculated using the box2d as opposed to matter in the browser version. The stable/unstable distinction in the cases here is simple enough that differences between physics engines should not matter. 

    Fast failure means ending returning failure for states in which the agent has built outside the silhouette or left holes that can't be filled. Enable this to spare the agent the misery of having to fill the map with blocks if it can't win anymore. 

    `legal_action_space` returns only legal (meaning block fully in silhouette, but not necessarily stable) action rather than all possible blocks that can be placed. Note that this is the default: if agents should reason about all possible block placements, use `False`.

    All the important functions are implemented by the State class, which represents a particular configuration of the world. The world merely manages the current state and wraps around the functions of the class State to keep compatibility with previous uses of the class.

    Default values are chosen to correspond to the silhouette 2 study, see block_construction/experiments/silhouette_2.

    Dimensions are in y,x. The origin is top left (in accordance with numpy arrays.

    Physics provider should either be "box2d" (legacy) or "matter" or an instantiated matter_server.Physics_Server with a socket to a running physics server which uses matter.js for compatibility with the human experiments (see `matter_server.js`).

    Note: if the physics provider is created by the world object, the world object will not be garbabe collected and `__del__()` will need to be called manually. This is because the physics provider has a socketio.Client which is not garbage collected.

    """

    def __init__(self, dimension=(8, 8), silhouette=None, block_library=None, fast_failure=False, legal_action_space=True, physics=True, physics_provider="matter"):
        self.dimension = dimension
        # Defines dimensions of possible blocks.
        if block_library is None:  # the default block library is the one from the silhouette 2 study
            block_library = [
                BaseBlock(1, 2),
                BaseBlock(2, 1),
                BaseBlock(2, 2),
                BaseBlock(2, 4),
                BaseBlock(4, 2),
            ]
        self.block_library = block_library
        # load the target silhouette as numpy array
        self.silhouette = self.load_silhouette(silhouette)
        # save the full silhouette in case we have to modify the actual silhouette in the case of subgoal planning
        self.full_silhouette = np.copy(self.silhouette)
        # generate a new state with no blocks in it
        self.current_state = Blockworld.State(self, [])
        # set world state to fail if the agent has built outside the silhouette or left a hole
        self.fast_failure = fast_failure
        self.legal_action_space = legal_action_space  # only return legal actions?
        self.physics = physics  # turn physics on or off?
        self.destroy_physics_server = False # do we need to kill the server? (only matters if we're using matter and created it)
        if physics:
            if physics_provider == "box2d":
                self.physics_provider = "box2d"
            elif type(physics_provider) == matter_server.Physics_Server:
                self.physics_provider = physics_provider
            elif physics_provider == "matter":
                # we create the physics provider ourself
                self.physics_provider = matter_server.Physics_Server(
                    y_height=self.dimension[1])
                self.destroy_physics_server = True
            else:
                raise Exception(
                    "Physics provider must be either 'box2d' or 'matter' or a socketio.Client")

    def __str__(self):
        """String representation of the world"""
        return self.__class__.__name__

    def __del__(self):
        """Destroy the world"""
        if self.destroy_physics_server:
            self.physics_provider.kill_server()
        
    def copy(self):
        return copy.deepcopy(self)

    def transition(self, action, state=None, force=False):
        """Takes an action and a state and returns the resulting state without applying it to the current state of the world."""
        if state is None:
            state = self.current_state
        # if action not in self.possible_actions(state): #this might be slow and could be taken out for performance sake
            # print("Action is not in possible actions")
        # determine where the new block would land
        baseblock, x = action  # unpacking the action
        if baseblock is None:
            return state
        # check for legality of transition
        if not force:
            if action not in self.current_state.possible_actions(legal=False):
                raise Exception("Action not possible")
        # determine y coordinates of block
        # find the lowest free row in the tower
        try:
            y = int(
                np.where(state.blockmap[:, x:x+baseblock.width].any(axis=1))[0][0]) - 1
        except IndexError:
            y = self.dimension[0]-1
        # create new block
        new_block = Block(baseblock, x, y)
        # create new state
        new_state = Blockworld.State(self, state.blocks + [new_block])
        return new_state

    def status(self):
        """Expanded status function also returns a reason for failure"""
        if self.is_win():
            return "Win", "None"
        if self.is_fail():
            if self.stability() == False:
                return "Fail", "Unstable"
            if self.current_state.possible_actions() == []:
                return "Fail", "Full"
            # the two conditions below are for fast failure only
            if filled_outside(self.current_state) > 0:
                return "Fail", "Outside"
            if holes(self.current_state) > 0:
                return "Fail", "Holes"
        return "Ongoing", "None"

    def load_silhouette(self, silhouette):
        if type(silhouette) is np.ndarray:
            if silhouette.shape == self.dimension:
                return silhouette
            else:
                # print("Silhouette dimensions", silhouette.shape, "don't match world dimensions. Setting world dimensions to match.")
                self.dimension = silhouette.shape
                return silhouette
        elif silhouette is None:  # No silhouette returns an empty field.
            return np.zeros(self.dimension)
        else:
            # TODO implement importing from file
            raise Warning("Use other function in file to import from file")

    def set_silhouette(self, silhouette, is_full=False):
        """Sets new silhouette (as bitmap) and flushes the current states cache. Doesn't overwrite full silhuouette by default"""
        self.silhouette = silhouette
        if is_full:
            self.full_silhouette = silhouette
        self.current_state.clear()

    """Simple functions inherited from the class World"""

    def apply_action(self, action, force=False):
        if not force:
            if action not in self.current_state.possible_actions(legal=False):
                raise Exception("Action not possible")
        self.current_state = self.transition(
            action, self.current_state, force=force)

    def is_fail(self, state=None):
        if state is None:
            state = self.current_state
        # should we fail the trial if it can't be completed to save time? Suggested by David.
        if self.fast_failure:
            if filled_outside(self.current_state) > 0:
                return True
            if holes(self.current_state) > 0:
                return True
        # always active fail states
        if state.stability() is False:  # we loose if its unstable
            return True
        # we loose if we aren't finished and have no options
        if state.score(F1score) != 1 and state.possible_actions() == []:
            return True

        return False

    def is_win(self, state=None):
        if state is None:
            state = self.current_state
        if state.score(F1score) == 1 and state.stability():
            return True

    def is_full_win(self, state=None):
        if state is None:
            state = self.current_state
        # temporarily replace the silhouette with the full one, restore after checking
        old_silhouette = np.copy(self.silhouette)
        self.silhouette = self.full_silhouette
        if state.score(F1score) == 1 and state.stability():
            self.silhouette = old_silhouette
            return True
        self.silhouette = old_silhouette
        return False

    def score(self, state=None, scoring_function=None):
        if state is None:
            state = self.current_state
        if self.is_fail(state):
            return self.fail_penalty
        if self.is_win(state):
            return self.win_reward
        return state.score(scoring_function)

    def F1score(self, state=None):
        if state is None:
            state = self.current_state
        return state.score(F1score)

    def possible_actions(self, state=None, legal=None):
        if state is None:
            state = self.current_state
        return state.possible_actions(legal=legal)

    def legal_actions(self, state=None):
        if state is None:
            state = self.current_state
        return state.legal_actions()

    def stability(self, state=None):
        if state is None:
            state = self.current_state
        return state.stability()

    class State():
        """This subclass contains a (possible or current) state of the world and implements a range of functions to score it, namely for F1 (how much of the figure are we filling out) and physical stability. It also generates possible actions. The blockworld classes wrap around this class.
        Hashes of this class are orderinvariant as to the order the blocks were placed (but not their positions). Use 'order_sensitive_hash' for an hash that takes placement order of blocks into account."""

        def __init__(self, world, blocks):
            self.world = world
            self.blocks = blocks
            self.world_width = self.world.dimension[1]
            self.world_height = self.world.dimension[0]
            # bitmap for placement of blocks
            self.blockmap = np.zeros(
                (self.world_height, self.world_width), dtype=int)
            # read the blocks into the blockmap
            self._update_map_with_blocks(blocks)
            self._stable = None
            self._cached_hash = None  # Cached hash value. It's only filled once we actually generate a hash and invalidates when the blockmap is updated. ⚠️ It is NOT updated when the blockmap/block list is touched manually! ⚠️
            self._legal_actions = None  # Cached actions. It's only filled once we actually generate a hash and invalidates when the blockmap is updated. ⚠️ It is NOT updated when the blockmap/block list is touched manually! ⚠️
            self._possible_actions = None  # Cached actions. It's only filled once we actually generate a hash and invalidates when the blockmap is updated. ⚠️ It is NOT updated when the blockmap/block list is touched manually! ⚠️

        def __eq__(self, other):
            """The order of the blocks does not matter, as they have their location attached. So the sorted list should be equal between two states which consist of the same blocks no matter the order in which they were placed"""
            return self.order_invariant_hash() == other.order_invariant_hash()

        def __hash__(self):
            return self.order_invariant_hash()

        def transition(self, action):
            """Takes an action and returns the resulting state without applying it to the current state of the world."""
            return self.world.transition(action, self)

        def clear(self):
            """Clears the various cached values of the state and updates the blockmap."""
            self._stable = None
            self._legal_actions = None
            self._possible_actions = None
            try:
                del(self._F1score)
            except:
                pass
            try:
                del(self._cached_hash)
            except:
                pass
            # self._update_map_with_blocks(self.blocks)

        def order_invariant_blockmap(self):
            """Returns an np.array of the blockmap that ignores the order in which the blocks were placed, ie outputs the same blockmap for all the states that have the same kinds of blocks in the same locations, but not the same order in which these blocks have been placed.
            Use (A==B).all() to compare two numpy arrays for equality."""
            return self._get_new_map_from_blocks(self.order_invariant_blocks())

        def order_invariant_blocks(self):
            """Returns ordered list of blocks that have been placed to ensure the same order in the list even if they have been placed in a different order."""
            def _block_key(block):
                """Helper function returns a float to sort blocks by y location first, then x location."""
                return float(block.y + .5/(block.x+1))
            return sorted(self.blocks, key=_block_key, reverse=True)

        def order_invariant_hash(self):
            """String of order invariant blockmap"""
            if self._cached_hash is None:  # create hash if necessary
                self._cached_hash = self.order_invariant_blockmap().tostring()  # Long hex string
                # self._cached_hash = self.order_invariant_blockmap().__str__() #Slower, but human readable
            return self._cached_hash

        def order_sensitive_hash(self):
            return self.blockmap.tostring()

        def _update_map_with_blocks(self, blocks, delete=False):
            """Fills the blockmap with increasing numbers for each block. 0 is empty space. Original blockmap behavior can be achieved by blockmap > 0."""
            self._cached_hash = None  # invalidate the hash
            for b in blocks:
                new_number = 0 if delete else (
                    np.max(self.blockmap)+1)  # numbers increase
                self.blockmap[(b.y-b.height)+1: b.y+1,
                              b.x:(b.x+b.width)] = new_number

        def _get_new_map_from_blocks(self, blocks):
            """Same as _update_map_with_blocks, only that it returns a new blockmap without touching the blockmap of the state."""
            blockmap = np.zeros(
                (self.world_height, self.world_width), dtype=int)
            i = 0
            for b in blocks:
                i += 1
                blockmap[(b.y-b.height)+1: b.y+1, b.x:(b.x+b.width)] = i
            return blockmap

        def score(self, scoring_function):
            """Returns the score according to the the scoring function that is passed. """
            return scoring_function(self)

        def stability(self, visual_display=False):
            """Runs physics engine to determine stability. Returns true or false, but could be adapted for noisy simulations. Caches it's value."""
            if self.world.physics is False:
                # turning off physics means everything is stable
                return True
            if self._stable is not None and not visual_display:
                # return cached value
                return self._stable
            # we actually need to run the physics engine
            if self.world.physics_provider == "box2d":
                bwworld = self.state_to_bwworld()
                self._stable = display_world.test_world_stability(
                    bwworld, RENDER=visual_display) == 'stable'
                pass
            else:
                assert type(
                    self.world.physics_provider) == matter_server.Physics_Server, "Physics provider must be a matter_server.Physics_Server object"
                self._stable = self.world.physics_provider.get_stability(
                    self.blocks)
            return self._stable

        def is_win(self):
            return self.world.is_win(state=self)

        def is_fail(self):
            return self.world.is_fail(state=self)

        def possible_actions(self, legal=None):
            """Generates all actions that are possible in this state independent of whether the block is stable or within the silhouette. Simply checks whether the block is in bounds. 
            Format of action is (BaseBlock from block_library, x location of lower left)."""
            if legal is None:
                legal = self.world.legal_action_space
            if legal:  # return legal actions instead
                return self.legal_actions()
            if self._possible_actions is not None:
                return self._possible_actions
            possible_actions = []
            for base_block in self.world.block_library:
                # starting coordinate is bottom left. The block can't possible overlap the right side.
                for x in range(self.world_width-base_block.width+1):
                    # and whether it overlaps the top boundary by simply looking if the block at the top is free
                    if np.sum(self.blockmap[0: base_block.height, x: x+base_block.width]) == 0:
                        possible_actions.append((base_block, x))
            self._possible_actions = possible_actions
            return possible_actions

        def legal_actions(self):
            """Returns the subset of possible actions where the placed block is fully within the silhouette. Returns [] if the current state is already non-legal."""
            if self._legal_actions is not None:
                return self._legal_actions
            legal_actions = [a for a in self.possible_actions(legal=False) if legal(
                self.transition(a))]  # need legal false here to prevent infinite recursion
            return legal_actions

        def visual_display(self, blocking=False, silhouette=None):
            """Shows the state in a pretty way. Silhouette is shown as dotted outline."""
            pyplot.close('all')
            plt.figure(figsize=(4, 4))
            pyplot.pcolor(self.blockmap[::-1], cmap='hot_r',
                          vmin=0, vmax=20, linewidth=0, edgecolor='none')
            if silhouette is None:  # try to get silhouette
                try:
                    silhouette = self.world.silhouette
                except:
                    pass
            if silhouette is not None:
                # we print the target silhouette as transparent overlay
                pyplot.pcolor(silhouette[::-1], cmap='binary', alpha=.8, linewidth=2, facecolor='none',
                              edgecolor='black', capstyle='round', joinstyle='round', linestyle=':')
            pyplot.show(block=blocking)

        def state_to_bwworld(self):
            """converts state to Blockworld.world.
                Helper function for block_construction/stimuli/blockworld_helpers.py testing code"""
            wh = self.world.dimension[0]  # world height
            bwworld = blockworld_helpers.World(
                world_height=self.world.dimension[0], world_width=self.world.dimension[1])
            for b in self.blocks:
                bwworld.add_block(b.width, b.height, b.x, wh-b.y-1)
            return bwworld

        def is_improvement(self, action):
            """Pass an action. Returns True if that action increases F1 score, False otherwise"""
            return F1score(self) < F1score(self.transition(action))

        def is_improvement_or_equal(self, action):
            """Pass an action. Returns True if that action doesn't decrease F1 score, False otherwise"""
            return F1score(self) <= F1score(self.transition(action))


class Block:
    '''
        Adapted from block_construction/stimuli/blockworld_helpers.py
        Creates Block objects that are instantiated in a world
        x and y define the position of the BOTTOM LEFT corner of the block

        Defines functions to calculate relational properties between blocks
    '''

    def __init__(self, base_block, x, y):
        self.base_block = base_block  # defines height, width and other functions
        # bottom left coordinate
        self.x = x
        self.y = y
        self.height = base_block.height
        self.width = base_block.width
        self.verts = base_block.translate(base_block.base_verts, x, y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.height == other.height and self.width == other.width

    def __str__(self):
        """width, height at (x,y)"""
        return "{}x{} at ({},{})".format(self.width, self.height, self.x, self.y)

    # Block Relational Properties
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
        buttressing_side = self.x == other.x + \
            other.width or other.x == self.x + self.width
        return y_overlap and buttressing_side

    def vertical_touch(self, other):
        ''' Test to see whether this block sits top to bottom against other block or bottom to top against other block.
            Corner to corner treated as not touching.
        '''
        x_overlap = not self.leftof(other) and not self.rightof(other)
        buttressing_up = self.y == other.y + \
            other.height or other.y == self.y + self.height
        buttressing_down = self.y == other.y - \
            other.height or other.y == self.y - self.height
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
                                    (0, 0)])
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
        new_verts[:, 0] = base_verts[:, 0] + dx
        new_verts[:, 1] = base_verts[:, 1] + dy
        return new_verts

    def get_corners(self, base_verts):
        '''
        input: list or array of block vertices in absolute coordinates
        output: absolute coordinates of top_left, bottom_left, bottom_right, top_right
        '''
        corners = {}
        corners['bottom_left'] = base_verts[0]
        corners['top_left'] = base_verts[1]
        corners['top_right'] = base_verts[2]
        corners['bottom_right'] = base_verts[3]
        return corners

    def get_area(self, shape='rectangle'):
        '''
        input: w = width 
            h = height           
            shape = ['rectangle', 'square', 'triangle']
        output
        '''
        # extract width and height from dims dictionary
        if shape in ['rectangle', 'square']:
            area = self.width*self.height
        elif shape == 'triangle':
            area = self.width*self.height*0.5
        else:
            print('Shape type not recognized. Please use recognized shape type.')
        return area


"""Scoring functions. These should be passed to the scoring function of the state. Note that these operate on the blockmap, not the blocks."""


def F1score(state, force=False):
    """Returns the F1 score relative to the target silhouette defined in the corresponding world. If the silhouette is empty, this produces division by 0 errors and returns NaN.

    By default, the F1 score is cached for performance. Use force if the blockmap has been manually changed."""
    if hasattr(state, '_F1score') and not force:
        return state._F1score
    # smallest possible float to prevent division by zero. Not the prettiest of hacks
    s = sys.float_info[3]
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    precision = np.sum(built & target)/(np.sum(built) + s)
    recall = np.sum(built & target)/(np.sum(target) + s)
    F1score = 2 * (precision * recall)/(precision + recall + s)
    state._F1score = F1score
    return F1score


def precision(state):
    # smallest possible float to prevent division by zero. Not the prettiest of hacks
    s = sys.float_info[3]
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    return np.sum(built & target)/(np.sum(built) + s)


def recall(state):
    # smallest possible float to prevent division by zero. Not the prettiest of hacks
    s = sys.float_info[3]
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    return np.sum(built & target)/(np.sum(target) + s)


def weighted_precision_recall(state, precision_weight=1):
    """Simply the weighted average of precision and recall. GIve a higher weigth to precision discourage building outside the structure."""
    return (precision(state) * precision_weight + recall(state))/(precision_weight+1)


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
    """Returns True if the current blockmap is legal and false otherwise."""
    target = state.world.silhouette > 0
    built = state.blockmap > 0
    if np.sum((1-target) & built) > 0:
        return False
    else:
        return True


def holes(state):
    """Returns the number of cells that are in the silhouette, but not built, but have something built on top of them. This tracks the number of holes."""
    target = state.world.silhouette > 0
    built = (state.blockmap > 0) * 2
    mapped = target + built
    holes = 0
    for x in range(built.shape[1]):  # we don't need to check the bottom
        for y in range(built.shape[0]-1):
            if mapped[y, x] == 3:  # if we have a cell with blck and in silhouette
                # if blocks below is not built and in silhouette
                holes = holes + np.sum(mapped[y+1:, x] == 1)
                break  # since we're going through from the top, we don't need to iterate further
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


def cells_left(state):
    """The number of cells not yet filled out in the silhouette"""
    return np.sum(((state.world.silhouette > 0) - (state.blockmap > 0)) > 0)
