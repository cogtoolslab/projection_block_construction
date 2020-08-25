from BFS_Agent import BFS_Agent, Ast_edge, Ast_node
import blockworld
import random
import copy
import math
import sys

class MCTS_Agent(BFS_Agent):
    """This agent derives from the brute tree search agent and implements Monte Carlo Tree Search.
    
    Planning cost is each state visited during light rollout."""
    #super()
    pass
    def __init__(self,world=None, horizon = 10000, random_seed=None):
        self.world = world
        self.horizon = horizon
        self.random_seed = random_seed
    
    def __str__(self):
        """Yields a string representation of the agent"""
        return self.__class__.__name__+' horizon: '+str(self.horizon)+' random seed: '+str(self.random_seed)

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            'agent_type':self.__class__.__name__,
            'horizon':self.horizon,
            'random_seed':self.random_seed
            }

    def MCTS(self,iterations,state=None,verbose=False):
        """Performs MCTS on the given state iterations times."""
        number_of_states_visited = 0
        if state is None:
            state = self.world.current_state
        root = MCTS_Agent.MCTS_Ast_node(state,self.world,random_seed=self.random_seed)
        for i in range(iterations):
            selected_node = root.selection()
            new_node = selected_node.expansion()
            outcome,_states_scored = new_node.simulation()
            number_of_states_visited += _states_scored
            new_node.backpropagation(outcome)
            if verbose:
                print(i,"of" ,iterations,"with result",str(outcome))
                root.print_tree
        if verbose:
            print("MCTS is done")
            root.print_tree()
        return root,number_of_states_visited

    def act(self,steps=-1,iterations=None,verbose=False,exploration_parameter=math.sqrt(2)):
        """Makes the agent act, including changing the world state. By default: the agent plans once, then acts until the end of planning. Not guaranteed to finish
         The agent samples according Kocsis, Levente; Szepesvári, Csaba (2006). "Bandit based Monte-Carlo Planning". """
        #Ensure that we have a random seed if none is set
        if self.random_seed is None: self.random_seed = random.randint(0,99999)
        if iterations is None:
            iterations = self.horizon
        if self.world.status()[0] != 'Ongoing':
            print("Can't act with world in status",self.world.status())
            return
        #perform MCTS
        cur,number_of_states_visited = self.MCTS(iterations,verbose=verbose)
        step = 0
        sequence_of_actions = []
        while self.world.status()[0] == 'Ongoing' and steps != 0:# and not cur.is_leaf():
            #choose highest scoring node
            scores = [action.target.MC_ratio[1] for action in cur.actions if action.target.actions is not []]
            if scores == []:
                #we haven't explored the tree further and didn't finish. Either run with a larger budget or run MCTS again.
                break
            index = scores.index(max(scores))
            action = [action for action in cur.actions if action.target.actions is not []][index]
            self.world.apply_action(action.action)
            sequence_of_actions.append(action)
            cur = action.target
            step += 1
            steps = steps - 1
            if verbose:
                print("Took step ",step," with action ",[str(a) for a in action.action],"and  score",max(scores)," and got world state",self.world.current_state)
                # cur.print_tree()
                self.world.current_state.visual_display(blocking=True,silhouette=self.world.silhouette)
        if verbose:
            print("Done, reached world status: ",self.world.status())
            # self.world.current_state.visual_display(blocking=True,silhouette=self.world.silhouette)
        return [[b for b in a.action] for a in sequence_of_actions][:step],number_of_states_visited

    class MCTS_Ast_node(Ast_node):
        """MCTS adaptation of Ast_node. MCTS steps are implemented in the node function for the subtree"""
       
        def __init__(self,state,world,score=None,stability=None,parent=None,MC_ratio=(0,0),random_seed=None):
            super().__init__(state,score,stability,parent)
            self.MC_ratio = MC_ratio
            self.world = world
            if random_seed is None: random_seed = random.randint(0,99999) 
            self.random_seed = random_seed 

        def UCT(self,c=math.sqrt(2)):
            """Per Kocsis, Levente; Szepesvári, Csaba (2006). "Bandit based Monte-Carlo Planning". In Fürnkranz, Johannes; Scheffer, Tobias; Spiliopoulou, Myra (eds.). Machine Learning: ECML 2006, 17th European Conference on Machine Learning, Berlin, Germany, September 18–22, 2006, Proceedings. Lecture Notes in Computer Science. 4212"""
            w = self.MC_ratio[0]
            n = self.MC_ratio[1] + sys.float_info[3] #smallest possible float to prevent division by zero. Not the prettiest of hacks
            #get total number of simulations run by parent node
            if self.parent_action is not None:
                parent = self.parent_action.source
                N = parent.MC_ratio[1]
            else: #root node special case
                N = n

            # current_nodes = [parent]
            # while current_nodes  != []:
            #     children_nodes = []
            #     for node in current_nodes:
            #         N += node.MC_ratio[1]
            #         children_nodes += [action.target for action in node.actions]
            #     current_nodes = children_nodes
            # self.UCT = w/n + c*math.sqrt(math.log(N)/n)
            return w/n + c*math.sqrt(math.log(N)/n)



        def selection(self):
            """Performs the selection step of MCTS. Selects child nodes until a leaf node is found. If not a final state, it is expanded."""
            cur = self
            while not cur.is_leaf():
                # children_nodes = [action.target for action in cur.actions]
                #sample from children_nodes using UCT weighting
                scored_children = [[action.target.UCT(),action.target] for action in cur.actions]
                max_score = max([a[0] for a in scored_children])
                random.seed(self.random_seed)
                cur = random.choice([a[1] for a in scored_children if a[0] == max_score])
            return cur

        def expansion(self):
            """Performs expansion step of MCTS. Checks if node is not final game state, and if so, creates a child node to perform expansion on."""
            if self.world.is_win(self.state) == True or self.world.is_fail(self.state) == True:
                # We can't expand on states that have ended the game, but we still need to backpropagate them
                # print("Expanding final state")
                return self
            #pick an edge to expand—just random sampling for now
            # pos_actions = self.state.possible_actions()
            #pick an edge to expand
            pos_actions = self.state.possible_actions()

            if pos_actions == []:
                #return self if this is a leaf node
                return self

            # we expand all children
            for action in pos_actions:
                self.add_action(action)
            # and sample one child state to return
            random.seed(self.random_seed)
            child = random.sample([a.target for a in self.actions],1)[0]
            return child

        def simulation(self,max=1000):
            """Performs the simulation step of MCTS by simulating a game from the current state by randomly choosing legal actions (ie. light playouts). Performs maximally max steps (set negative for infinite steps). Return True for win, False for loose and did not finish."""
            number_of_states_visited = 0
            w = copy.deepcopy(self.world)            
            w.current_state = self.state
            #heavy rollout
            # a = Agent(world=w,horizon=1,scoring_function=blockworld.random_scoring)
            # while w.status()[0] == 'Ongoing' and max != 0:
            #     a.act(1)
            #     max = max - 1
            # light rollout—random agent with legal moves
            while w.status()[0] == 'Ongoing' and max != 0:
                legal_actions = [a for a in w.possible_actions() if blockworld.legal(w.transition(a))]
                if legal_actions == []:
                    #if we have nowhere to go
                    # print("End of the  line",w.status())
                    # w.current_state.visual_display(blocking=True,silhouette=w.silhouette)
                    break
                action = random.sample(legal_actions,1)[0]
                # w.current_state.visual_display(blocking=True,silhouette=w.silhouette)
                w.apply_action(action)
                number_of_states_visited += 1
                max = max - 1
            return (True, number_of_states_visited) if w.status()[0] == 'Win' else (False, number_of_states_visited)

        def backpropagation(self,outcome):
            """Implements the backpropagtion step of MCTS"""
            self.MC_ratio = (self.MC_ratio[0]+outcome,self.MC_ratio[1]+1)
            # stop if self is root
            if self.parent_action is None:
                return
            #recursive call
            self.parent_action.source.backpropagation(outcome)

        def add_action(self,action,target=None):
            if action in [act.action for act in self.actions]:
                Warning("Action ",action," is already in actionset for this node")
                pass
            else:
                if target is None: #figure out the target state
                    target = MCTS_Agent.MCTS_Ast_node(self.state.world.transition(action,self.state),self.world,random_seed=self.random_seed)
                action = Ast_edge(action,self,target) #create new action
                action.target.parent_action = action #set as parent for target state
                self.actions.append(action) #add action to node   
            return action #this is just for convenience, as the action is also added to the state

        def print_tree(self,level=0):
                """Prints out the tree to the command line in depth first order."""
                print(self.state," score: ",self.score," stability: ",self.stability," ratio won/played: ",self.MC_ratio[0],"/",self.MC_ratio[1], " UCT: ",self.UCT(),sep="") 
                for child,action in [(action.target,action.action) for action in self.actions]:
                    print("\n|","____"*(level+1)," ",[str(b) for b in action]," → ",end="",sep="")#remove __str__ for non-blockworld?
                    child.print_tree(level+1) #pass


                


