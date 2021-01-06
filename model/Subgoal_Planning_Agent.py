import os
import sys

proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from random import choice
import utils.blockworld as blockworld
from model.BFS_Agent import *
import model.utils.decomposition_functions
import copy
import numpy as np

UNSOLVABLE_PENALTY = 999999999999
MAX_STEPS = 20

class Subgoal_Planning_Agent(BFS_Agent):
    """Implements n subgoal planning. Works by running the lower level agent until it has found a solution or times out.. """

    def __init__(self,
                         world=None,
                         decomposer = None,
                         sequence_length = 1, #consider sequences up to this length
                         step_size = 1, #how many subgoals to act. Negative to act from end of plan
                         include_subsequences=True,
                         c_weight = 1/1000,
                         max_cost=10**3, #maximum cost before we give up trying to solve a subgoal
                         lower_agent = BFS_Agent(only_improving_actions=True),
                         random_seed = None
                         ):
            self.world = world
            self.sequence_length = sequence_length
            self.include_subsequences = include_subsequences # only consider sequences of subgoals exactly `lookahead` long or ending on final decomposition
            self.step_size = step_size
            self.c_weight = c_weight
            self.max_cost = max_cost
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
            'lookeahead':self.sequence_length,
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

    def act(self,steps=None,verbose=False):
        """Finds subgoal plan, then builds the first n subgoals. Pass none to build plan length. Steps here refers to subgoals (ie. 2 steps is acting the first two planned subgoals). Pass -1 to steps to execute the entire subgoal plan.
        
        NOTE: only the latest decomposed silhouette is saved by experiment runner: the plan needs to be extracted from the saved subgoal sequence."""
        if self.random_seed is None: self.random_seed = randint(0,99999)
        # get best sequence of subgoals
        sequence = self.plan_subgoals(verbose=verbose)
        if steps is None:
            if self.step_size <= 0:
                steps = len(sequence) + self.step_size
            else:
                steps = self.step_size
        # finally plan and build all subgoals in order
        cur_i = 0
        actions = []
        solution_cost = 0
        last_silhouette = None
        for sg in sequence:
            if cur_i == steps: break #stop after the nth subgoal
            if sg.actions is None: # no actions could be found
                # raise Warning("No actions could be found for subgoal "+str(sg.name))
                print("No actions could be found for subgoal "+str(sg.name))
                continue
            for action in sg.actions:
                self.world.apply_action(action,force=True) #applying the actions to the world — we need force here because the reference of the baseblock objects aren't the same
                actions.append(action)
                solution_cost += sg.solution_cost
                last_silhouette = sg.target
        return actions,{
                                'solution_cost':solution_cost,
                                'sequence_cost':sequence.sequence_cost(),
                                'all_sequences_cost':'XXX XXX XXX', #TODO implement
                                '_subgoal_sequence':sequence,
                                'decomposed_silhouette': last_silhouette
                                }

    def plan_subgoals(self,verbose=False):
        """Plan a sequence of subgoals. First, we need to compute a sequence of subgoals however many steps in advance (since completion depends on subgoals). Then, we compute the cost and value of every subgoal in the sequence. Finally, we choose the sequence of subgoals that maximizes the total value over all subgoals within."""
        self.decomposer.set_silhouette(self.world.full_silhouette) #make sure that the decomposer has the right silhouette
        sequences = self.decomposer.get_sequences(state = self.world.current_state,length=self.sequence_length,filter_for_length=not self.include_subsequences)
        if verbose: 
            print("Got",len(sequences),"sequences:")
            for sequence in sequences:
                print([g.name for g in sequence])
        # we need to score each in sequence (as it depends on the state before)
        self.fill_subgoals_in_sequence(sequences,verbose=verbose)
        # now we need to find the sequences that maximizes the total value of the parts according to the formula $V_{Z}^{g}(s)=\max _{z \in Z}\left\{R(s, z)-C_{\mathrm{Alg}}(s, z)+V_{Z}^{g}(z)\right\}$
        return self.choose_sequence(sequences,verbose=verbose) #return the sequence of subgoals with the highest score
    
    def choose_sequence(self, sequences,verbose=False):
        """Chooses the sequence that maximizes $V_{Z}^{g}(s)=\max _{z \in Z}\left\{R(s, z)-C_{\mathrm{Alg}}(s, z)+V_{Z}^{g}(z)\right\}$ including weighing by lambda"""
        scores = [None]*len(sequences)
        for i in range(len(sequences)):
            scores[i] = sequences[i].V(self.c_weight)
            if verbose: print("Scoring sequence",i+1,"of",len(sequences),"->",[g.name for g in sequences[i]],"score:\t\t",scores[i])
        top_indices = [i for i in range(len(scores)) if scores[i] == max(scores)]
        top_sequences = [sequences[i] for i in top_indices]
        seed(self.random_seed) #fix random seed
        chosen_sequence = choice(top_sequences)
        if verbose: print("Chose sequence:",chosen_sequence.names(),"with score",chosen_sequence.V(self.c_weight))
        return chosen_sequence
  
    def fill_subgoals_in_sequence(self,sequences,verbose=False):
        seq_counter = 0 # for verbose printing
        for sequence in sequences:
            seq_counter += 1 # for verbose printing
            sg_counter = 0 # for verbose printing
            current_world = self.world
            for subgoal in sequence:
                sg_counter += 1 # for verbose printing
                #get reward and cost and success of that particular subgoal and store the resulting world
                subgoal.prior_world = copy.deepcopy(current_world)
                self.solve_subgoal(subgoal,verbose=verbose)
                if verbose: 
                    print("For sequence",seq_counter,'/',len(sequences),
                    "scored subgoal",
                    sg_counter,'/',len(sequence),"named",
                    subgoal.name,
                    "with C:"+str(subgoal.C)," R:"+str(subgoal.R())," actions:"+str(subgoal.actions))
                #if we can't solve it to have a base for the next one, we break
                if subgoal.C == UNSOLVABLE_PENALTY:
                    break
                current_world = subgoal.past_world #store the resulting world as the input to the next one

    def solve_subgoal(self,subgoal,verbose=False):
        """Tries as long as needed to find a single solution to the current subgoal"""
        if subgoal.prior_world is None:
            subgoal.prior_world = self.world
        # generate key for cache
        key = subgoal.key()
        if key in self._cached_subgoal_evaluations:
            # print("Cache hit for",key)
            #NOTE this will add the cost taken the first time during planning—ie we don't count the caching for the total cost calculation
            hit = self._cached_subgoal_evaluations[key]
            subgoal.past_world = hit.past_world
            subgoal.actions = hit.actions
            subgoal.C = hit.C
            subgoal.solution_cost = hit.solution_cost
            subgoal.planning_cost = hit.planning_cost
            subgoal.iterations = hit.iterations
            return subgoal
        total_costs = 0
        i = 0
        while total_costs < self.max_cost:
            temp_world = copy.deepcopy(subgoal.prior_world)
            temp_world.set_silhouette(subgoal.target)
            temp_world.current_state.clear() #clear caches
            if temp_world.current_state.possible_actions() == []:
                #we can't do anything in this world
                break
            self.lower_agent.world = temp_world
            steps = 0
            costs = 0
            actions = []
            while temp_world.status()[0] == 'Ongoing' and total_costs < self.max_cost and steps < MAX_STEPS:
                chosen_actions,info = self.lower_agent.act()
                actions += chosen_actions
                costs += info['states_evaluated']
                steps += 1
            total_costs += costs
            i += 1 #counting attempts
            if verbose:
                print("Attempted subgoal",subgoal.name,"on attempt",str(i),"with result",str(temp_world.status()),"and total cost",str(total_costs))
            if temp_world.status()[0] == 'Win':
                #we've found a solution! write it to the subgoal
                subgoal.past_world = copy.deepcopy(temp_world)
                subgoal.actions = actions
                subgoal.C = costs
                subgoal.solution_cost = costs
                subgoal.planning_costs = total_costs
                subgoal.iterations = i
                self._cached_subgoal_evaluations[key] = subgoal #TODO copy or reference?
                return subgoal
        # if we've made it here, we've failed to find a solution
        #store cached evaluation
        subgoal.solution_cost = UNSOLVABLE_PENALTY
        subgoal.C = UNSOLVABLE_PENALTY
        subgoal.planning_costs = total_costs
        subgoal.iterations = i
        self._cached_subgoal_evaluations[key] = subgoal #TODO copy or reference?
        return subgoal