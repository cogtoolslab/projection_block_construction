from model.Simulated_Lookahead_Subgoal_Planning_Agent import *

class Simulated_No_Subgoal_Planning_Agent(Simulated_Lookahead_Subgoal_Planning_Agent):
    """Simulates the action level planner on the basis of a full subgoal solver dataframe, just like the simulated lookahead planner. Simply picks and 'runs' the maximum subgoal."""

    def __init__(self,
    all_sequences=None,
    parent_agent=None,
    random_seed = None,
    note = ""):
        self.all_sequences = all_sequences
        self.parent_agent = parent_agent #the full decomposition agent we draw from
        try:
            self.world = parent_agent.world
        except AttributeError:
            #no parent, no world
            pass
        self.random_seed = random_seed
        if self.random_seed is None:
            self.random_seed = self.random_seed = randint(0,99999)
        self.note = note #just a label to keep track of the agent    
        self.step_size = 1

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        try:
            return {**{
            'agent_type':self.__class__.__name__,
            'random_seed':self.random_seed
            }, **{"parent: "+key:value for key,value in self.parent_agent.get_parameters().items()}}
        except AttributeError:
            #no parent yet
             return {
            'agent_type':self.__class__.__name__,
            'random_seed':self.random_seed
            }

    def generate_sequences(self,state=None):
        """Generates list of sequences to try, then generates them from all_sequences by splicing larger sequences as needed"""
        if state is None:
            state = self.world.current_state
        #generate sequences
        target_sequences = self.parent_agent.decomposer.get_sequences(state = state,length=1,filter_for_length=True)
        #get the full decomposition—we assume that it is the last one generated
        sequences = [target_sequences[-1]]
        return sequences

    def choose_sequence(self, sequences):
        return sequences[0]

