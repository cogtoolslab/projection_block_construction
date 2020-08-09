from BFS_Agent import BFS_Agent
import blockworld
from random import random,choice
from Q_table import Q_table,state_key,action_key

class Naive_Q_Agent(BFS_Agent):
    """This class implements naive Q learning. 
    Uses a state space over order-invariant blockmaps and action space over baseblocks times possible start locations.
    Then simply updates the Q values of those state action pairs with the same formula. 
    Reward functions can be passed as usual.
    You can (and probably should) specify a total number of episodes and a limit on steps that the agent can take.
    After the model achieves *streaks* many wins in a row, we terminate and consider the model fully trained.
    """
 
    def __init__(self,world = None,heuristic=blockworld.F1_stability_score,max_episodes=10**6,max_streaks=100,max_steps=40,explore_rate = 0.8 , learning_rate = 1, discount_factor = 1,):
        self.world = world
        self.max_episodes = max_episodes
        self.max_streaks = max_streaks
        self.max_steps = max_steps
        self.heuristic = heuristic
        self.explore_rate = explore_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def __str__(self):
        """Yields a string representation of the agent"""
        return 'type: '+self.__class__.__name__+' heuristic: '+self.heuristic.__name__+' max_episodes: '+str(self.max_steps)+' max_streaks: '+str(max_streaks)+' explore_rate: '+str(explore_rate)+' learning_rate: '+str(learning_rate)+' discount_factor: '+str(discount_factor)

    def train(self,max_episodes=None,max_streaks=None,max_steps=40,Qs=None,verbose=False):
        """Trains a Q matrix and returns it. It's not automatically set to the agent!
        Pass an existing table to Qs to update it, otherwise a new one is created."""
        if Qs is None:
            Qs = Q_table(action_space = self.world.possible_actions())
        if max_steps is None: max_steps = self.max_steps
        if max_episodes is None: max_episodes = self.max_episodes
        streaks = 0
        stats = {"episode_length":[],"final_reward":[]}
        # run episodes
        for episode_i in range(max_episodes):
            #reset world state
            current_state = self.world.current_state

            #act for however many steps
            for step_i in range(max_steps):
                #get current best action
                action = self.sample_action(current_state,Qs)
                #perform action
                next_state = self.world.transition(action,current_state)
                #update Q matrix with temporal difference learning
                old_Q = Qs.get_Q(current_state,action)
                max_Q_prime = Qs.max_Q(next_state)
                #calculate Q update
                new_Q = old_Q + self.learning_rate * (self.heuristic(current_state) + self.discount_factor * max_Q_prime - old_Q)
                Qs.set_Q(current_state,action,new_Q) # set the new Q value
                if verbose > 2:
                    print("Step {} in episode {} with reward {} and chosen action {} got Q {} for state {}".format(step_i,episode_i,self.heuristic(current_state),action_key(action),new_Q,state_key(current_state)))
                #update the current state
                current_state = next_state
                last_i = step_i
                #Have we reached a terminal state?
                if self.world.is_fail(current_state) or self.world.is_win(current_state): 
                    if verbose > 0 and self.world.is_win(current_state): print("You're Winner! 🏆")
                    #perform terminal Q update (which is just the reward of the terminal state) for all possible actions
                    Qs.fill_Q(current_state,self.heuristic(current_state))
                    break #terminate the episode

            #end of episode handling
            #update stats
            stats["episode_length"].append(last_i)
            stats["final_reward"].append(self.heuristic(current_state))
            if verbose > 0: #Run
                print("End of episode {} with reward {} and length {} and {} streaks".format(episode_i,self.heuristic(current_state),last_i,streaks))
            #if we've won
            if self.world.is_win(current_state): 
                streaks += 1 #increment streaks
            else: streaks = 0 #reset streaks :(
            if streaks > self.max_streaks: #if we've won enough streaks, we end training
                break
        return Qs,stats
    
    def sample_action(self,state,Qs,):
        """Sample a random action or choose the best possible action to take according to explore rate.
        This is where the policy driving the learning lives.
        - [ ] could be a lot more sophisticated (softmax etc)"""
        if random() > self.explore_rate: 
            #we chose the best action so far
            return Qs.argmax_Q(state)
        else:
            # we randomly sample an action that's possible in this state
            return choice(state.possible_actions())
        
    def act(self,verbose=False):
        """Trains the Q value for a while, then acts according to it by sampling the argmax of the Q value over the current state"""
        #train
        Qs,stats = self.train(verbose=verbose)
        while self.world.status()[0] == 'Ongoing':
            #get best action
            action = Qs.argmax_Q(self.world.current_state)
            self.world.apply_action(action)
            if verbose:
                print(self.world.status())
                self.world.current_state.visual_display(True,self.world.silhouette)
        print('Done,',self.world.status())

