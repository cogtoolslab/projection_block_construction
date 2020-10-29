import os
import sys
proj_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,proj_dir)

from model.BFS_Agent import BFS_Agent, Ast_node, Ast_edge
import utils.blockworld as blockworld
 
class Beam_Search_Agent(BFS_Agent):
    """An agent using beam search—the agent expands only a certain number of promising nodes at each step and performs breadth first search over the resulting smaller tree. 
    https://en.wikipedia.org/wiki/Beam_search
    https://stackoverflow.com/questions/22273119/what-does-the-beam-size-represent-in-the-beam-search-algorithm
    The heuristic that judges which states are promising can be chosen (random would be naive sampling of the tree).
    Beam search returns the first result found. If no final solution is found before max_depth is reached, the best-so-far trajectory is returned. max_depth can be set to -1 to keep the algorithm running until the goal is found or no actions are possible.
    Physics is only taken into account by the heuristic function: use one that includes physics for dense physics (eg. blockworld.F1_stability_score)

    TODO: 
    - [ ] variable beam width
    - [ ] stochastic beam search
    """

    def __init__(self, world=None, beam_width = 1000,max_depth=20,heuristic = blockworld.F1_stability_score, only_improving_actions = False):
        self.world = world
        self.beam_width = beam_width
        self.heuristic = heuristic
        self.max_depth = max_depth
        self.only_improving_actions = only_improving_actions
    
    def __str__(self):
        """Yields a string representation of the agent"""
        return self.__class__.__name__+' beam_width: '+str(self.beam_width)+' heuristic: '+self.heuristic.__name__+' max_depth '+str(self.max_depth)

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            'agent_type':self.__class__.__name__,
            'heuristic':self.heuristic.__name__,
            'beam_width':self.beam_width
            }

    def act(self,steps = None, verbose = False):
        """By default, we perform one beam search and then perform the entire sequence that has been found.
        The ast represents the entire trajectory that we've explored so far."""
        #check if we even can act
        if self.world.status()[0] != 'Ongoing':
            print("Can't act with world in status",self.world.status())
            return
        step = 0
        edges,number_of_states_evaluated = self.beam_search(verbose)
        for edge in edges: #act in the world for each edge
            if self.only_improving_actions:
                # check if action improves the current state of the world 
                if not self.world.current_state.is_improvement(edge.action): 
                    break
            self.world.apply_action(edge.action)
            step += 1
            if verbose:
                print("Took step ",step," with action ",[str(a) for a in edge.action]," and got world state",self.world.current_state)
                self.world.current_state.visual_display(blocking=True,silhouette=self.world.silhouette)
            if step == steps: break #if we have acted x steps, stop acting
        if verbose:
            print("Done, reached world status: ",self.world.status())
        return [e.action for e in edges][:step],{'states_evaluated':number_of_states_evaluated}

    def beam_search(self,verbose=False):
        """Performs beam search. Each state is assigned a parent, which means that converging states are still treated as individual here. If we want converging states, we need a more sophisticated way of choosing the final trajectory and a constructor of states that prevents multiple instantions of the same world state.
        This function return a sequence of connecting Ast_edge objects, which contain the chosen trajectory through the space. If no winning action could be chosen, empty list is returned and no action is chosen.Could be extended to search the generated tree of most promising actions so far and take the best final state from this."""
        #make root of ast
        root = Ast_node(self.world.current_state)
        i = 0
        number_of_states_evaluated = 0
        current_nodes = [root] #contains the states at the current level of depth that we expand upon
        while i < self.max_depth:
            #Perform one expansion in beam search  
            candidate_edges = [] #contains possible edges (actions)
            for node in current_nodes:
                #add possible edges to the next set of candidate edges
                possible_actions = node.state.possible_actions()
                for action in possible_actions:
                    target = Ast_node(self.world.transition(action,node.state)) #get target state ast node object
                    edge = Ast_edge(action,node,target) #make edge
                    edge.target.parent_action = edge #add the parent action to allow for backtracking the found path
                    candidate_edges.append(edge)
                    number_of_states_evaluated += 1
            if verbose: print("Found",len(candidate_edges),"potential edges at depth",i,"for",len(current_nodes),"nodes")
            if candidate_edges == []: #if we can't act any further, end the search
                if verbose: print("No candidate edges, ending with current best node")
                current_nodes.sort(key=get_score_node)
                return backtrack(current_nodes[0]),number_of_states_evaluated #return the path to the best current state            
            current_nodes = []
            #select the most promising edges
            def get_score_edge(edge): #helper function to get the score of a target state according to heuristic function
                return self.heuristic(edge.target.state)
            def get_score_node(node): #helper function to get the score of a target state according to heuristic function
                return self.heuristic(node.state)
            candidate_edges.sort(key=get_score_edge,reverse=True) #sort the list according to the heuristic
            if verbose:
                print("Best 6 actions:",[",".join([str(a) for a in edge.action])+", "+str(get_score_edge(edge)) for edge in candidate_edges][:6])
            #add candidates to tree
            for candidate_edge in candidate_edges[:self.beam_width]:
                candidate_edge.source.actions.append(candidate_edge)
                current_nodes.append(candidate_edge.target)
            i+=1
            #check if we're done
            for node in current_nodes:
                if self.world.is_win(node.state):
                    #we have a winner state, returns its trajectory
                    if verbose: print("Found winning node")
                    return backtrack(node),number_of_states_evaluated
        #if we haven't found a winning state before reaching max depth, return best known state at current depth
        current_nodes.sort(key=get_score_node)
        if verbose: print("Didn't find winning node, returning best so far")
        return backtrack(current_nodes[0]),number_of_states_evaluated #return the path to the best current state
                
            
def backtrack(state):
    """Helper function. Takes a state and returns the sequence of action that got us there in the correct order by stepping through parent states."""
    action_sequence = []
    while state.parent_action is not None: #only the root state will have no parent action
        action_sequence.append(state.parent_action)
        state = state.parent_action.source
    action_sequence.reverse()
    return action_sequence
