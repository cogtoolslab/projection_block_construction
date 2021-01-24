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
            'random_seed':self.random_seed,
            'note':self.note
            }, **{"parent: "+key:value for key,value in self.parent_agent.get_parameters().items()}}
        except AttributeError:
            #no parent yet
             return {
            'agent_type':self.__class__.__name__,
            'random_seed':self.random_seed,
            'note':self.note
            }

    def act(self,state=None):
        """Simulates acting. Returns chosen sequence and the other sequences as well. This will update the blockmap in the current world state, but not the rest of the state! Cached values and the block list won't work."""
        if state is None:
            state = self.world.current_state
        #choose sequence
        chosen_seq = self.choose_sequence(self.all_sequences)
       #execute actions
        # finally plan and build all subgoals in order
        cur_i = 0
        actions = []
        solution_cost = 0
        partial_planning_cost = 0
        last_silhouette = None
        for sg in chosen_seq:
            if cur_i == self.step_size: break #stop after the nth subgoal
            if sg.actions is None: # no actions could be found
                # raise Warning("No actions could be found for subgoal "+str(sg.name))
                # print("No actions could be found for subgoal "+str(sg.name))
                continue
            for action in sg.actions:
                self.world.apply_action(action,force=True) #applying the actions to the world — we need force here because the reference of the baseblock objects aren't the same
                actions.append(action)
            cur_i += 1
            solution_cost += sg.solution_cost
            partial_planning_cost += sg.planning_cost
            last_silhouette = sg.target
        return actions,{
                                'partial_solution_cost':solution_cost, #solutions cost of steps acted
                                 'solution_cost':chosen_seq.solution_cost(),
                                'partial_planning_cost':partial_planning_cost, #planning cost of steps
                                'planning_cost':chosen_seq.planning_cost(),
                                'decomposed_silhouette': last_silhouette,
                                '_chosen_subgoal_sequence':chosen_seq}
  

    def choose_sequence(self, sequences):
        max_name = max(sequences[-1].names()) #get the complete decomposition name. Assumes they're ordered.
        return [seq for seq in sequences if seq.names() == [max_name]][0] #return the sequence that matches the complete decomp and is 1

