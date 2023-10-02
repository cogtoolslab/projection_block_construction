class World:
    """This class maintains and holds the world state, action space and the transitions between them. This class assumes a world state as a dictionary with action space over the same. An action amounts to setting the key value to 1. For more complicated worlds, this can be inherited"""
    
    initial_state = {} #the initial state of the world
    win_states = [{}] #list of winning states
    fail_states = [{}] #list of failing states
    # action_space = [] #list of actions
    fail_penalty = -1000
    win_reward = 10

    def __init__(self):
        self.world_state = self.initial_state.copy()
        self.action_space = self.initial_state.keys #setting up the action space as  the parts of the world state

    def transition(self,action,state = None): 
        """takes an action and a state and returns the following state"""
        if state is None:
            state = self.world_state.copy()
        if action not in self.possible_actions(state):
            Warning("Action is not allowed")
        state = state.copy() # otherwise we change the passed state
        if self.is_fail(state) or self.is_win(state): #if we're in a final state we just return the final state
            return(state)
        # apply action
        try:
            state[action] = 1
        except KeyError:
            Warning("Action is not key of world state")
            pass
        return state
        
    def apply_action(self,action): #updates world state
        if self.status()[0] != "Ongoing":
            Warning("World is already in final state " + self.status())
        self.world_state = self.transition(action,self.world_state)
        return self.status() #return current status as string

    def is_fail(self,state = None):
        if state is None:
            state = self.world_state
        if state in self.fail_states:
            return True
        else:
            return False

    def is_win(self,state = None):
        if state is None:
            state = self.world_state
        if state in self.win_states:
            return True
        else:
            return False

    def status(self):
            if self.is_win():
                return "Win"
            if self.is_fail():
                return "Fail"
            return "Ongoing"

    def possible_actions(self,state=None): #returns actions whose corresponding worldstate is not 1 yet
        if state is None:
            state = self.world_state
        if self.is_fail(state) or self.is_win(state): #if we're in a final state there's no actions to take
            return []
        return [key for key,value in state.items() if value == 0]

    def score(self,state=None):
        if state is None:
            state = self.world_state
        if self.is_fail(state):
            return self.fail_penalty
        if self.is_win(state):
            return self.win_reward
        return 0
    
    def stability(self,state=None):
        if state is None:
            state = self.world_state
        if self.is_fail(state):
            return False
        return True
        


class Stonehenge(World):
    """Implements the simple stonehenge world from Dietz. A, B are columns, C is roof. """
    initial_state = {'A':0,'B':0,'C':0}
    win_states = [{'A':1,'B':1,'C':1}]
    fail_states = [{'A':0,'B':0,'C':1},
                                {'A':1,'B':0,'C':1},
                                {'A':0,'B':1,'C':1}]
    actions = ['A','B','C']