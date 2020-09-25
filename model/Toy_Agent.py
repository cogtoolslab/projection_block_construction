import os
import sys
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from model.BFS_Agent import BFS_Agent

class Toy_Agent(BFS_Agent):
    """This class keeps the code for the original toy agent around that operates on sequences of action, not a tree structure. Mostly for documentation purposes."""
    def __init__(self, world, horizon = 2):
        self.world = world
        self.horizon = horizon
    
    def generate_actions(self,horizon = None,state = None):
        def _generate_actions_helper(self,horizon,state,cur_sequence):
            possible_actions = self.world.possible_actions(state)
            # anchor recursion
            if horizon <= 0 or possible_actions == []:
                return [cur_sequence]
            results = []
            for action in possible_actions:
                results += _generate_actions_helper(self,horizon-1,self.world.transition(action,state.copy()),cur_sequence+[action]) # recursive step
            return results
        #special case for myopic agent
        if self.horizon == 0:
            return [self.Action_sequence(actions) for actions in self.world.possible_actions(state)]

        if horizon is None:
            horizon = self.horizon
        if state is None:
            state = self.world.world_state.copy()
        action_sequences = _generate_actions_helper(self,horizon,state,[])
        return [self.Action_sequence(actions) for actions in action_sequences] #convert to action sequence object
    
    class Action_sequence():
        def __init__(self,actions,score = None):
            self.actions = actions
            self.score = score
       
    def score_actions(self,action_sequences,beginning_state = None):
        #special case for myopic agent
        if self.horizon == 0:
            for action_seq in action_sequences:
                action_seq.score = 0
            return action_sequences
        if beginning_state is None:
            beginning_state = self.world.world_state.copy()
        # if type(action_sequences[0] ) is not list: # if we get passed a single list, wrap it in another list 
        #     action_sequences = [action_sequences]
        for action_seq in action_sequences:
            #calculate final state
            state = beginning_state.copy()
            for action in action_seq.actions: #go to final state
                state = self.world.transition(action,state)
                if self.world.is_fail(state):
                    break
            score = self.world.score(state)
            action_seq.score = score
        return action_sequences
    
    def select_action_seq(self,scored_actions): #could implement softmax here
        max_score = max([action_seq.score for action_seq in scored_actions if action_seq.score is not None])
        max_action_seqs = [action_seq for action_seq in scored_actions if action_seq.score == max_score] #choose randomly from highest actions
        return max_action_seqs[randint(0,len(max_action_seqs)-1)]

    def act(self,steps = 1,verbose=False): 
        possible_actions = self.generate_actions()
        scored_actions = self.score_actions(possible_actions)
        #get the first action with maximum score
        if scored_actions == []:
            Warning("No actions to take")
            return
        max_action_seq = self.select_action_seq(scored_actions)
        print(max_action_seq)
        #let's act
        if steps is 'All': #we act the entire planning sequence
            for action in max_action_seq.actions:
                self.world.apply_action(action)
                print(action) 
                print(self.world.world_state ) 
        else: # if we have a limit of how many steps to take
            for i in range(steps):
                try:
                    self.world.apply_action(max_action_seq.actions[i])
                    print(max_action_seq.actions[i])
                    print(self.world.world_state )
                except IndexError:
                    Warning("Not enough steps in plan for step "+str(i))
        print(self.world.status())