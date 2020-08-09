def Q_table():
    """This class stores a table of Q values to keep the code extensible. It allows for setting and getting the Q values as well as for getting the argmax over action for a Q(S,A)
    
    This class expects to be passed blockworld states and actions as they show up in the program and takes care to convert them to order–invariant representations internally.

    The default value of the Q function is 0
    
      The Q table is a dictionary, where the key is a state and the value is another dictionary where the key is an action and the value is the Q value.
        Actions are represented as tuple ('BaseBlock.__str__(),x_loc').
        This could be extended to create an array or some other datastructure, right now it's a dictionary so we don't need to build up the state space in advance. Hash functions would be the sane choice here.
        - [ ] use array with hash functions
        """

    def init(self,action_space=None,state_space=None, initial_value=0):
        self.initial_value = initial_value
        self.Qs = {}
        self.state_space = state_space
        self.action_space = action_space
    
    def get_Q(self,state,action):
        """Get Q value for state,action pair. Returns initial value if there is not yet a Q value saved."""
        try:
            return self.Qs[state_key(state)][action_key(action)]
        except KeyError:
            return self.initial_value
        
    def set_Q(self,state,action,value):
        """Sets Q value for state, action pair. Silently creates entry if it doesn't exist yet and updates with default value for action space.."""
        S = state_key(state)
        if S not in self.Qs: #if we haven't seen the state yet, initialize all possible actions with default values
            for a in self.action_space:
                self.Qs[S][action_key(a)] = self.initial_value
        #set the value
        self.Qs[S][action_key(action)] = value

    def max_Q(self,state):
        """Returns the maximum value for Q(s,a) where we take the maximum over all a already initialized in this state.
        - [x] What about uninitialized actions? -> Shouldn't exist per set_Q"""
        S = state_key(state)
        return self.Qs[S][self.argmax_Q(state)]

    def argmax_Q(self,state):
        """Returns the maximum action for Q(s,a) where we take the maximum over all a already initialized in this state.
        - [x] What about uninitialized actions? -> Shouldn't exist per set_Q"""
        S = state_key(state)
        return max(self.Qs[state], key=self.Qs[state].get)


def state_key(state):
    """Returns orderinvariant representation of state as string"""
    return state.order_invariant_blockmap().__str__()

def action_key(action):
    """Returns string representation of action"""
    return action