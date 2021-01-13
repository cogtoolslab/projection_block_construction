from model.Subgoal_Planning_Agent import *
from model.utils.decomposition_functions import Subgoal,Subgoal_sequence

class Simulated_Lookahead_Subgoal_Planning_Agent(Subgoal_Planning_Agent):
    """Simulates a lookahead agent on the data generated by `subgoal_generator_runner`. This is NOT an agent compatible with experiment runner like the others."""

    def __init__(self,
    all_sequences=None,
    parent_agent=None,
    sequence_length = 1,
     step_size = 1,
    include_subsequences=True,
    c_weight = 1/1000,
    random_seed = None,
    note = ""):
        self.all_sequences = all_sequences
        self.parent_agent = parent_agent #the full decomposition agent we draw from
        try:
            self.world = parent_agent.world
        except AttributeError:
            #no parent, no world
            pass
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.include_subsequences = include_subsequences
        self.c_weight = c_weight
        self.random_seed = random_seed
        if self.random_seed is None:
            self.random_seed = self.random_seed = randint(0,99999)
        self.note = note #just a label to keep track of the agent
    
    def set_parent_agent(self,parent_agent):
        self.parent_agent = parent_agent #the full decomposition agent we draw from
        self.world = parent_agent.world

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        try:
            return {**{
            'agent_type':self.__class__.__name__,
            'sequence_length':self.sequence_length,
            'include_subsequences':self.include_subsequences,
            'c_weight':self.c_weight,
            'step_size':self.step_size,
            'random_seed':self.random_seed
            }, **{"parent: "+key:value for key,value in self.parent_agent.get_parameters().items()}}
        except AttributeError:
            #no parent yet
             return {
            'agent_type':self.__class__.__name__,
            'sequence_length':self.sequence_length,
            'include_subsequences':self.include_subsequences,
            'c_weight':self.c_weight,
            'step_size':self.step_size,
            'random_seed':self.random_seed
            }

    def generate_sequences(self,state=None):
        """Generates list of sequences to try, then generates them from all_sequences by splicing larger sequences as needed"""
        if state is None:
            state = self.world.current_state
        #generate sequences
        target_sequences = self.parent_agent.decomposer.get_sequences(state = state,length=self.sequence_length,filter_for_length=not self.include_subsequences)
        last_bm = state.blockmap > 0
        target_sequences.reverse() 
        #fill sequences from all sequences
        sequences = []
        for target_sequence in target_sequences:
            target_names = target_sequence.names()
            Seq = None
            for sequence in self.all_sequences:
                found = False
                seq_names = sequence.names()
                #iterate all the parents sequences to find sublists
                for i in range(len(seq_names)):
                    if seq_names[i:i+len(target_names)] == target_names:
                        #found a candidate match—check if the preceding source matches
                        if (sequence.subgoals[i].source is None and np.sum(last_bm) == 0) or (np.all(sequence.subgoals[i].source == last_bm)):
                            Seq = Subgoal_sequence(sequence.subgoals[i:i+len(target_names)])
                            if Seq.complete():           
                                                  sequences.append(Seq)
                                                  found = True
                            break
                if found: break #we've already found a sequence, let's break
            #if we've found no winning sequence, save the last sequence that we've seen
            if Seq is not None: sequences.append(Seq)
            #we've gotten all the sequences we needed—return all and complete sequences
        return sequences

    def act(self,state=None):
        """Simulates acting. Returns chosen sequence and the other sequences as well. This will update the blockmap in the current world state, but not the rest of the state! Cached values and the block list won't work."""
        if state is None:
            state = self.world.current_state
        #get sequences
        all_sequences = self.generate_sequences(state=state)
        #choose sequence
        chosen_seq = self.choose_sequence(all_sequences)
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
        all_sequences_cost = sum([s.planning_cost() for s in all_sequences])
        return actions,{
                                'partial_solution_cost':solution_cost, #solutions cost of steps acted
                                 'solution_cost':chosen_seq.solution_cost(),
                                'partial_planning_cost':partial_planning_cost, #planning cost of steps
                                'planning_cost':chosen_seq.planning_cost(),
                                'all_sequences_planning_cost':all_sequences_cost, 
                                'decomposed_silhouette': last_silhouette,
                                '_all_subgoal_sequences':all_sequences,
                                '_chosen_subgoal_sequence':chosen_seq}
                            

    def choose_sequence(self,sequences):
        """Chooses a sequence. Overwrite self.c_weight and call again to generate different value"""
        if sequences == []:
            #cant choose from empty sequence
            return Subgoal_sequence([])
        scores = [None]*len(sequences)
        for i in range(len(sequences)):
            scores[i] = sequences[i].V(self.c_weight)
        top_indices = [i for i in range(len(scores)) if scores[i] == max(scores)]
        top_sequences = [sequences[i] for i in top_indices]
        seed(self.random_seed) #fix random seed
        chosen_sequence = choice(top_sequences)
        return chosen_sequence


