import os
import sys

proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from random import choice
import utils.blockworld as blockworld
from model.BFS_Lookahead_Agent import *
import model.utils.decomposition_functions
import copy
import numpy as np

BAD_SCORE = -10**20
MAX_NUMBER_OF_SUBGOALS = 64

"""
NOTE
The corresponding code in decomposition_functions has changed and this will need to be adapted to work

NOTE this is out of date and most likely superseded by the subgoal planning agent class. Consider deleting.
"""

class Resource_Rational_Subgoal_Planning_Agent(BFS_Lookahead_Agent):
    """Implements n subgoal lookahead planning. Works by sampling the lower level agent and using that to score sequences of actions. """

    def __init__(self,
                         world=None,
                         decomposer = None,
                         lookahead = 1,
                         include_subsequences=True,
                         c_weight = 1000,
                         S_treshold=0.8,
                         S_iterations=1,
                         lower_agent = BFS_Lookahead_Agent(only_improving_actions=True),
                         random_seed = None
                         ):
            self.world = world
            self.lookahead = lookahead
            self.include_subsequences = include_subsequences # only consider sequences of subgoals exactly `lookahead` long or ending on final decomposition
            self.c_weight = c_weight
            self.S_threshold = S_treshold #ignore subgoals that are done in less than this proportion
            self.S_iterations = S_iterations #how often should we run the simulation to determine S?
            self.lower_agent = lower_agent
            self.random_seed = random_seed
            if decomposer is None:
                try:
                    decomposer = model.utils.decomposition_functions.Horizontal_Construction_Paper(self.world.full_silhouette)
                except AttributeError: # no world has been passed, will need to be updated using decomposer.set_silhouette
                    decomposer = model.utils.decomposition_functions.Horizontal_Construction_Paper(None) 
            self.decomposer = decomposer
            self._cached_subgoal_evaluations = {} #sets up cache for  subgoal evaluations

    def __str__(self):
            """Yields a string representation of the agent"""
            return str(self.get_parameters())

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {**{
            'agent_type':self.__class__.__name__,
            'lookeahead':self.lookahead,
            'decomposition_function':self.decomposer.__class__.__name__,
            'include_subsequences':self.include_subsequences,
            'c_weight':self.c_weight,
            'S_threshold':self.S_threshold,
            'S_iterations':self.S_iterations,
            'random_seed':self.random_seed
            }, **{"lower level: "+key:value for key,value in self.lower_agent.get_parameters().items()}}
    
    def set_world(self, world):
        super().set_world(world)
        self.decomposer.set_silhouette(world.full_silhouette)
        self._cached_subgoal_evaluations = {} #clear cache

    def act(self,steps=1,verbose=False):
        """Finds subgoal plan, then builds the first subgoal. Steps here refers to subgoals (ie. 2 steps is acting the first two planned subgoals). Pass -1 to steps to execute the entire subgoal plan.
        
        NOTE: only the latest decomposed silhouette is saved by experiment runner: the plan needs to be extracted from the saved subgoal sequence."""
        if self.random_seed is None: self.random_seed = randint(0,99999)
        # get best sequence of subgoals
        sequence,sg_planning_cost = self.plan_subgoals(verbose=verbose)
        # finally plan and build all subgoals in order
        cur_i = 0
        lower_level_cost = 0
        lower_level_info = []
        lower_level_actions = []
        self.lower_agent.world = self.world
        while cur_i < len(sequence) and cur_i != steps: 
            current_subgoal = sequence.subgoals[cur_i]
            self.world.set_silhouette(current_subgoal.target)
            self.world.current_state.clear() #clear caches
            while self.world.status()[0] == "Ongoing":
                actions, info = self.lower_agent.act()
                lower_level_cost += info['states_evaluated']
                lower_level_info.append(info)
                lower_level_actions+=actions
            cur_i += 1
            self.world.set_silhouette(self.world.full_silhouette) # restore full silhouette to the world we're acting with
        return lower_level_actions,{'states_evaluated':lower_level_cost,
                                'sg_planning_cost':sg_planning_cost,
                                '_subgoal_sequence':sequence,
                                'decomposed_silhouette': current_subgoal.target}

    def plan_subgoals(self,verbose=False):
        """Plan a sequence of subgoals. First, we need to compute a sequence of subgoals however many steps in advance (since completion depends on subgoals). Then, we compute the cost and value of every subgoal in the sequence. Finally, we choose the sequence of subgoals that maximizes the total value over all subgoals within."""
        self.decomposer.set_silhouette(self.world.full_silhouette) #make sure that the decomposer has the right silhouette
        sequences = self.decomposer.get_sequences(state = self.world.current_state,length=self.lookahead,filter_for_length=not self.include_subsequences)
        if verbose: 
            print("Got",len(sequences),"sequences:")
            for sequence in sequences:
                print([g.name for g in sequence])
        # we need to score each in sequence (as it depends on the state before)
        number_of_states_evaluated = self.score_subgoals_in_sequence(sequences,verbose=verbose)
        # now we need to find the sequences that maximizes the total value of the parts according to the formula $V_{Z}^{g}(s)=\max _{z \in Z}\left\{R(s, z)-C_{\mathrm{Alg}}(s, z)+V_{Z}^{g}(z)\right\}$
        return self.choose_sequence(sequences,verbose=verbose),number_of_states_evaluated #return the sequence of subgoals with the highest score
    
    def choose_sequence(self, sequences,verbose=False):
        """Chooses the sequence that maximizes $V_{Z}^{g}(s)=\max _{z \in Z}\left\{R(s, z)-C_{\mathrm{Alg}}(s, z)+V_{Z}^{g}(z)\right\}$ including weighing by lambda"""
        scores = [None]*len(sequences)
        for i in range(len(sequences)):
            scores[i] = self.score_sequence(sequences[i])
            if verbose: print("Scoring sequence",i+1,"of",len(sequences),"->",[g.name for g in sequences[i]],"score:",scores[i])
        if verbose: print("Chose sequence:\n",sequences[scores.index(max(scores))])
        top_indices = [i for i in range(len(scores)) if scores[i] == max(scores)]
        top_sequences = [sequences[i] for i in top_indices]
        seed(self.random_seed) #fix random seed
        return choice(top_sequences)

    def score_sequence(self,sequence):
        """Compute the value of a single sequence with precomputed S,R,C. Assigns BAD_SCORE. If no solution can be found, the one with highest total reward is chosen."""
        score = 0
        penalized = False
        for subgoal in sequence:
            if subgoal.C is None or subgoal.S is None or subgoal.S < self.S_threshold or subgoal.R <= 0: 
                # we have a case where the subgoal computation was aborted early or we should ignore the subgoal because the success rate is too low or the reward is zero (subgoal already done or empty)
                penalized = True
            try: 
                # compute regular score
                subgoal_score =  subgoal.R - subgoal.C * self.c_weight
            except: 
                    #missing C—happens only in the case of fast_fail
                subgoal_score = subgoal.R
                penalized = True
            score += subgoal_score
        return score + (BAD_SCORE * penalized)
  
    def score_subgoals_in_sequence(self,sequences,verbose=False):
        """Add C,R,S to the subgoals in the sequences"""
        number_of_states_evaluated = 0
        seq_counter = 0 # for verbose printing
        for sequence in sequences: #reference or copy?
            seq_counter += 1 # for verbose printing
            sg_counter = 0 # for verbose printing
            prior_world = self.world
            for subgoal in sequence:
                sg_counter += 1 # for verbose printing
                #get reward and cost and success of that particular subgoal and store the resulting world
                R = self.reward_of_subgoal(subgoal.target,prior_world.current_state.blockmap) 
                S,C,winning_world,total_cost,stuck = self.success_and_cost_of_subgoal(subgoal.target,prior_world,iterations=self.S_iterations)
                number_of_states_evaluated += total_cost
                if verbose: 
                    print("For sequence",seq_counter,'/',len(sequences),
                    "scored subgoal",
                    sg_counter,'/',len(sequence),"named",
                    subgoal.name,
                    "with C:"+str(C)," R:"+str(R)," S:"+str(S))
                #store in the subgoal
                subgoal.R = R + stuck * BAD_SCORE
                subgoal.C = C
                subgoal.S = S
                #if we can't solve it to have a base for the next one, we break
                if winning_world is None:
                    break
        return number_of_states_evaluated

    def reward_of_subgoal(self,decomposition,prior_blockmap):
        """Gets the unscaled reward of a subgoal: the area of a figure that we can fill out by completing the subgoal in number of cells beyond what is already filled out."""
        return np.sum((decomposition * self.world.full_silhouette) - (prior_blockmap > 0))

    def success_and_cost_of_subgoal(self,decomposition,prior_world = None, iterations=1,max_steps = 20,fast_fail = False):
        """The cost of solving for a certain subgoal given the current block construction"""
        if prior_world is None:
            prior_world = self.world
        # generate key for cache
        key = decomposition * 100 - (prior_world.current_state.order_invariant_blockmap() > 0)
        key = key.tostring() #make hashable
        if key in self._cached_subgoal_evaluations:
            # print("Cache hit for",key)
            cached_eval = self._cached_subgoal_evaluations[key]
            return cached_eval['S'],cached_eval['C'],cached_eval['winning_world'],1,cached_eval['stuck'] #returning 1 as lookup cost, not the cost it tool to calculate the subgoal originally
        current_world = copy.deepcopy(prior_world)
        costs = 0
        wins = 0
        winning_world = None
        stuck = 0 # have we taken no actions in all iterations?
        for i in range(iterations):
            temp_world = copy.deepcopy(current_world)
            temp_world.set_silhouette(decomposition)
            temp_world.current_state.clear() #clear caches
            self.lower_agent.world = temp_world
            steps = 0
            while temp_world.status()[0] == 'Ongoing' and steps < max_steps:
                _,info = self.lower_agent.act()
                costs += info['states_evaluated']
                steps += 1
            wins += temp_world.status()[0] == 'Win'
            if steps == 0: #we have no steps, which means that the subgoal will lead to infinite costs
                stuck += 1
            if temp_world.status()[0] == 'Win':
                winning_world = copy.deepcopy(temp_world)
            #break early to save performance in case of fail
            if fast_fail and temp_world.status()[0] == 'Fail':
                 #store cached evaluation
                 #NOTE that this will lead to a state being "blacklisted" if it fails once
                cached_eval = {'S':wins/iterations,'C':costs/iterations,'winning_world':winning_world,'stuck':stuck == i}
                self._cached_subgoal_evaluations[key] = cached_eval
                return 0,None,None,costs,stuck == iterations
        #store cached evaluation
        cached_eval = {'S':wins/iterations,'C':costs/iterations,'winning_world':winning_world,'stuck':stuck == iterations}
        self._cached_subgoal_evaluations[key] = cached_eval
        return wins/iterations,costs/iterations,winning_world,costs/iterations,stuck == iterations


class Full_Sample_Subgoal_Planning_Agent(Resource_Rational_Subgoal_Planning_Agent):
    """Same as subgoal planning agent, only that we act the entire sequence of subgoals after planning and plan the entire sequence."""

    def __init__(self,
                        world=None,
                         decomposer = None,
                         c_weight = 1000,
                         S_treshold=0.8,
                         S_iterations=1,
                         lower_agent = BFS_Lookahead_Agent(only_improving_actions=True),
                         random_seed = None):
        super().__init__(world=world, decomposer=decomposer, lookahead=MAX_NUMBER_OF_SUBGOALS, include_subsequences=False, c_weight=c_weight, S_treshold=S_treshold, S_iterations=S_iterations, lower_agent=lower_agent, random_seed=random_seed)

    def act(self, verbose=False):
        """Plans and acts entire sequence"""
        return super().act(steps=MAX_NUMBER_OF_SUBGOALS, verbose=verbose)
