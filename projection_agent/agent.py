from random import randint
from multiprocessing.dummy import Pool
from itertools import repeat
import blockworld

class Agent:
    """An agent. This class holds the scoring and decision functions and maintains beliefs about the values of the possible actions. An action can be whatever—it's left implicit for now—, but should be an iterable."""

    def __init__(self, world=None, horizon = 3, scoring = 'Final state', sparse=False,scoring_function=blockworld.silhouette_score):
        self.world = world
        self.horizon = horizon
        self.sparse = sparse
        self.scoring = scoring
        self.scoring_function = scoring_function

    def __str__(self):
        """Yields a string representation of the agent"""
        return 'type: '+self.__class__.__name__+' scoring: '+self.scoring_function.__name__+' horizon: '+str(self.horizon)+' scoring: '+self.scoring+' sparse?: '+str(self.sparse)

    def set_world(self,world):
        self.world = world

    class Ast_node():
        """AST means action space tree. This class serves as a tree for planning. The nodes are states of the world, and the edges are actions. The nodes store scores (if determined), the action merely deterministically move between states."""
        def __init__(self,state,score=None,stability=None,parent_action=None):
            # self.state = state.copy() #just to be sure
            self.state = state
            self.score = score
            self.stability = stability
            self.actions = []
            self.parent_action = parent_action

        def is_leaf(self):
            if len(self.actions) == 0:
                return True
            else:
                return False
       
        def add_action(self,action,target=None):
            if action in [act.action for act in self.actions]:
                Warning("Action ",action," is already in actionset for this node")
                pass
            else:
                if target is None: #figure out the target state
                    target = Agent.Ast_node(self.state.world.transition(action,self.state))
                action = Agent.Ast_edge(action,self,target) #create new action
                action.target.parent_action = action #set as parent for target state
                self.actions.append(action) #add action to node   
            return action #this is just for convenience, as the action is also added to the state

        def print_tree(self,level=0):
            """Prints out the tree to the command line in depth first order."""
            print(self.state," score: ",self.score," stability: ",self.stability,sep="") 
            for child,action in [(action.target,action.action) for action in self.actions]:
                print("\n|","____"*(level+1)," ",[str(b) for b in action]," → ",end="",sep="")#remove __str__ for non-blockworld?
                child.print_tree(level+1) #pass
        
    class Ast_edge():
        """AST means action space tree. This class simply consists of an action connecting to ast_nodes. The target state will have to be added by the constructing function."""
        def __init__(self,action,source,target=None):
            self.action = action
            self.source = source
            self.target = target
            if target is None:
                Warning("Node has empty target")

    def build_ast(self,state=None,horizon=None,verbose=False):
        """Builds ast from given state to a certain horizon. Returns root of tree."""
        def fill_node(node):
            possible_actions = node.state.possible_actions()
            for action in possible_actions:     
                #add action to current node with target state already filled out
                node.add_action(action,Agent.Ast_node(self.world.transition(action,node.state)))#add the result of applying the action
            
        if horizon is None:
            horizon = self.horizon
        #implement logic for infinite horizon (see score)
        if state is None:
            state = self.world.current_state
        root = Agent.Ast_node(state) #make root of tree
        current_nodes = [root]
        #breadth first compile the tree of possible actions exhaustively
        for i in range(horizon): #this just idles when no children are to be found #DEBUG was horizon -1 
            children_nodes = []
            for node in current_nodes:
                fill_node(node)
                children_nodes += [action.target for action in node.actions]
            current_nodes = children_nodes
            if verbose:
                print("Depth of AST:",i+1,",found",len(current_nodes),"nodes")
        return root

    def score_node(self,node,sparse=False,dense_stability=True):
        if dense_stability:
            node.stability = self.world.stability(node.state) 
        if sparse:
            #only return stability and score for final states
            if self.world.is_win(node.state):
                node.score = self.world.win_reward
            elif self.world.is_fail(node.state):
                node.score = self.world.fail_penalty
            else:
                node.score = 0
        else:
            #dense rewards
            node.score = self.world.score(node.state,self.scoring_function)

    def score_ast(self,root,horizon='All',sparse=None,dense_stability=None):
        """Iterate through the Ast and score all the nodes in it. Works in place. Can use sparse rewards or dense. We can also choose to not score stability—in that case stability is scored implictly by the world.score function that returns the preset world reward for win states. Dense stability scores the node at the end of planning. Dense reward only gives reward if the world is in a terminal state."""
        if sparse is None:
            sparse = self.sparse
        if dense_stability is None:
            dense_stability = not sparse

        if horizon == 'All':
            counter = -1
        else:
            counter = horizon
        current_nodes = [root]
        nodes = [root]
        while current_nodes  != [] and counter != 0:
            children_nodes = []
            for node in current_nodes:
                self.score_node(node,sparse,dense_stability)
                children_nodes += [action.target for action in node.actions]
            current_nodes = children_nodes
            counter = counter -1 

    def generate_action_sequences(self,ast_root = None,horizon = None,include_subsets = False,verbose=False):
        """Generate all possible sequences of actions for the given horizon. """
        def get_path(leaf):
            """Gets the path from leaf to root in order root -> leaf""" 
            if leaf.parent_action is None: # if we're already at the root, just return empty list (otherwise we'll return [[]])
                return []
            action_sequence = []
            current_node = leaf
            while current_node.parent_action is not None: #while we're not at the root node
                action_sequence += [current_node.parent_action]
                current_node = current_node.parent_action.source
            action_sequence.reverse() #reverse the order
            return [action_sequence]

        if horizon is None:
            horizon = self.horizon
        if ast_root is None:
            ast_root = self.build_ast(horizon,verbose=verbose)
            self.score_ast(ast_root)

        action_sequences = []
        counter = horizon+1 #off by one?
        current_nodes = [ast_root]
        while current_nodes  != [] and counter != 0:
            counter = counter -1 
            children_nodes = []
            for node in current_nodes:
                if include_subsets or node.is_leaf() or counter == 0:
                    action_sequences += get_path(node)
                children_nodes += [action.target for action in node.actions]
            current_nodes = children_nodes
        return [Agent.Action_sequence(act_seq,None) for act_seq in action_sequences]

    def score_action_sequences(self,action_sequences,method = 'Final state',verbose=False):
        """Adds scores to the action sequences. Possible scoring: Final state, Sum, Average, Fixed. Operates in place."""
        # if action_sequences[0].actions[0].source.score is None: #if the tree hasn't been scored yet
        #     print("Tree must be scored.")
        #     return None
        for act_seq in action_sequences: #score every action sequence
            if method == 'Final state':
                score = act_seq.actions[-1].target.score + (not act_seq.actions[-1].target.stability) * self.world.fail_penalty 
            elif method == 'Sum':
                score = 0
                for action in act_seq.actions:
                    score += action.target.score + (not action.target.stability) * self.world.fail_penalty
                score = sum(score)
            elif method == 'Average':
                score = 0
                counter = 0
                for action in act_seq.actions:
                    score += action.target.score + (not action.target.stability) * self.world.fail_penalty
                    counter += 1
                score = sum(score)/counter
            elif method == 'Fixed':
                score = 0
            else:
                Warning("Method not recognized")
                pass
            act_seq.score = score
            if verbose:
                print([a.action.__str__() for a in act_seq.actions]," score: ",score)
        
    class Action_sequence():
        def __init__(self,actions,score = None):
            self.actions = actions
            self.score = score

        def print_actseq(self):
            print("Action sequence:",[[str(b) for b in act.action] for act in self.actions],"Score:",self.score)
       
    def select_action_seq(self,scored_actions): #could implement softmax here
        max_score = max([action_seq.score for action_seq in scored_actions if action_seq.score is not None])
        max_action_seqs = [action_seq for action_seq in scored_actions if action_seq.score == max_score] #choose randomly from highest actions
        return max_action_seqs[randint(0,len(max_action_seqs)-1)]

    def act(self,steps = 1,planning_horizon =None,verbose=False,scoring=None): 
        """Make the agent act, including changing the world state. The agent deliberates once and then acts n steps. To get the agent to deliberate more than once, call action repeatedly. Setting the planning_horizon higher than the steps gives the agent foresight. To get dumb agent behavior, set scoring to Fixed."""
        if scoring is None:
            scoring = self.scoring
        if planning_horizon is None:
            planning_horizon = self.horizon
        if planning_horizon == 0: #special case for dumb agent that can't plan at all
            planning_horizon = steps
            scoring = 'Fixed'
        elif planning_horizon < steps:
            print("Planning horizon must be higher or equal to the steps or 0! Setting the horizon from",planning_horizon,"to",steps)
            planning_horizon = steps
        # steps += 1 #this is so we take the sensible number of steps
        #check if we even can act
        if self.world.status() != 'Ongoing':
            print("Can't act with world in status",self.world.status())
            return
        #make ast
        ast = self.build_ast(horizon=planning_horizon,verbose=verbose)  
        #score it
        self.score_ast(ast)
        #generate action sequences
        act_seqs = self.generate_action_sequences(ast,horizon=planning_horizon,include_subsets=False,verbose=verbose)
        if act_seqs == []: #if we can't act. Should be covered by world fail state above.
            print("No possible actions")
            return
        #score action sequences
        self.score_action_sequences(act_seqs,scoring)
        #choose an action sequence
        chosen_seq = self.select_action_seq(act_seqs)
        if verbose:
            self.Ast_node.print_tree(ast)
            for act_seq in act_seqs:
                act_seq.print_actseq()
            print("Chosen action sequence:",[[str(b) for b in a.action] for a in chosen_seq.actions], "with score: ",chosen_seq.score)
        #take the steps
        for step in range(min([steps,len(chosen_seq.actions)])): #If the chosen sequence is shorter than the steps, only go so far
            self.world.apply_action(chosen_seq.actions[step].action)
            if verbose:
                print("Took step ",step+1," with action ",[str(a) for a in chosen_seq.actions[step].action]," and got world state",self.world.current_state)
                self.world.current_state.visual_display(blocking=True,silhouette=self.world.silhouette)
        if verbose:
            print("Done, reached world status: ",self.world.status())
            # self.world.current_state.visual_display(blocking=True,silhouette=self.world.silhouette)
        return [[str(b) for b in a.action] for a in chosen_seq.actions]