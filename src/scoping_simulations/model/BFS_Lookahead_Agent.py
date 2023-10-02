import os
import sys
from random import randint, seed

import scoping_simulations.utils.blockworld as blockworld
from scoping_simulations.model.Agent import Agent

proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, proj_dir)


class BFS_Lookahead_Agent(Agent):
    """An agent. This class holds the scoring and decision functions and maintains beliefs about the values of the possible actions. An action can be whatever—it's left implicit for now—, but should be an iterable. Performs lookahead BFS search.

    `only_improving_actions` means that the agent only takes actions if the action improves F1 score. Note that the agent returns an empty action if no actions can be taken—this will lead to infinite loops with simple while loops!

    Dense stability means that the agent only considers stable states.

    All agent should return a dictionary after acting along with the chosen actions. That dictionary can be empty, but can also contain other information to be logged.
    """

    def __init__(
        self,
        world=None,
        horizon=2,
        scoring="Final_state",
        only_improving_actions=False,
        sparse=False,
        first_solution=True,
        scoring_function=blockworld.F1_stability_score,
        dense_stability=True,
        random_seed=None,
        label="BFS Lookahead",
    ):
        self.world = world
        self.horizon = horizon
        self.sparse = sparse
        self.first_solution = first_solution
        self.scoring = scoring
        self.scoring_function = scoring_function
        self.dense_stability = dense_stability
        self.random_seed = random_seed
        self.only_improving_actions = only_improving_actions
        self.label = label

    def __str__(self):
        """Yields a string representation of the agent"""
        return (
            self.__class__.__name__
            + " scoring: "
            + self.scoring_function.__name__
            + "first_solution: "
            + str(self.first_solution)
            + " horizon: "
            + str(self.horizon)
            + " scoring: "
            + self.scoring
            + " sparse?: "
            + str(self.sparse)
            + " random seed: "
            + str(self.random_seed)
            + " label: "
            + self.label
        )

    def get_parameters(self):
        """Returns dictionary of agent parameters."""
        return {
            "agent_type": self.__class__.__name__,
            "scoring_function": self.scoring_function.__name__,
            "horizon": self.horizon,
            "first_solution": self.first_solution,
            "scoring_type": self.scoring,
            "random_seed": self.random_seed,
            "label": self.label,
        }

    def build_ast(self, state=None, horizon=None, verbose=False):
        """Builds ast from given state to a certain horizon. Returns root of tree."""
        number_of_states_evaluated = 0

        def fill_node(node):
            possible_actions = node.state.possible_actions()
            for action in possible_actions:
                # add action to current node with target state already filled out
                # check for stability
                if (
                    self.dense_stability
                    and not self.world.transition(action, node.state).stability()
                ):
                    continue
                # add the result of applying the action
                node.add_action(
                    action, Ast_node(self.world.transition(action, node.state))
                )

        if horizon is None:
            horizon = self.horizon
        # implement logic for infinite horizon (see score)
        if state is None:
            state = self.world.current_state
        root = Ast_node(state)  # make root of tree
        current_nodes = [root]
        self.score_node(root, self.sparse, self.dense_stability)
        # breadth first compile the tree of possible actions exhaustively
        for i in range(horizon):  # this just idles when no children are to be found
            children_nodes = []
            for node in current_nodes:
                fill_node(node)  # add children etc
                # score the child nodes
                for action in node.actions:
                    self.score_node(
                        action.target, self.sparse, self.dense_stability
                    )  # score the node
                    number_of_states_evaluated += 1
                if self.first_solution and self.world.is_win(node.state):
                    # we have found a winning node, so we return the sequence of action to get there
                    if verbose:
                        print("Found winning node at depth of AST", i + 1)
                    return root, get_path(node), number_of_states_evaluated
                # this will only consider nodes that aren't known unstable. When not using dense_stability, we consider all children
                children_nodes += [
                    action.target
                    for action in node.actions
                    if node.stability is not False
                ]
            current_nodes = children_nodes
            if verbose:
                print("Depth of AST:", i + 1, ",found", len(current_nodes), "nodes")
        return root, None, number_of_states_evaluated

    def score_node(self, node, sparse=False, dense_stability=True):
        if dense_stability:
            node.stability = self.world.stability(node.state)
        else:
            node.stability = None
        if sparse:
            # only return stability and score for final states
            if self.world.is_win(node.state):
                node.score = self.world.win_reward
            elif self.world.is_fail(node.state):
                node.score = self.world.fail_penalty
            else:
                node.score = 0
        else:
            # dense rewards
            node.score = self.world.score(node.state, self.scoring_function)

    def score_ast(self, root, horizon="All", sparse=None, dense_stability=None):
        """Iterate through the Ast and score all the nodes in it. Works in place. Can use sparse rewards or dense. We can also choose to not score stability—in that case stability is scored implicitly by the world.score function that returns the preset world reward for win states. Dense stability scores the node at the end of planning. Dense reward only gives reward if the world is in a terminal state."""
        if sparse is None:
            sparse = self.sparse
        if dense_stability is None:
            dense_stability = not sparse

        if horizon == "All":
            counter = -1
        else:
            counter = horizon
        number_of_states_evaluated = 0
        current_nodes = [root]
        while current_nodes != [] and counter != 0:
            children_nodes = []
            for node in current_nodes:
                self.score_node(node, sparse, dense_stability)
                number_of_states_evaluated += 1
                children_nodes += [action.target for action in node.actions]
            current_nodes = children_nodes
            counter = counter - 1
        return number_of_states_evaluated

    def generate_action_sequences(
        self, ast_root=None, horizon=None, include_subsets=True, verbose=False
    ):
        """Generate all possible sequences of actions for the given horizon."""

        if horizon is None:
            horizon = self.horizon
        if ast_root is None:
            ast_root = self.build_ast(horizon, verbose=verbose)
            self.score_ast(ast_root)

        action_sequences = []
        counter = horizon + 1  # off by one?
        current_nodes = [ast_root]
        while current_nodes != [] and counter != 0:
            counter = counter - 1
            children_nodes = []
            for node in current_nodes:
                if include_subsets or node.is_leaf() or counter == 0:
                    action_sequences += get_path(node)
                children_nodes += [action.target for action in node.actions]
            current_nodes = children_nodes
        return [
            BFS_Lookahead_Agent.Action_sequence(act_seq, None)
            for act_seq in action_sequences
        ]

    def score_action_sequences(
        self, action_sequences, method="Final_state", verbose=False
    ):
        """Adds scores to the action sequences. Possible scoring: Final state, Sum, Average, Fixed. Operates in place."""
        # if action_sequences[0].actions[0].source.score is None: #if the tree hasn't been scored yet
        #     print("Tree must be scored.")
        #     return None
        for act_seq in action_sequences:  # score every action sequence
            if method == "Final_state":
                score = (
                    act_seq.actions[-1].target.score
                    + (not act_seq.actions[-1].target.stability)
                    * self.world.fail_penalty
                )
            elif method == "Sum":
                score = []
                for action in act_seq.actions:
                    score.append(
                        action.target.score
                        + (not action.target.stability) * self.world.fail_penalty
                    )
                score = sum(score)
            elif method == "Average":
                score = []
                counter = 0
                for action in act_seq.actions:
                    score.append(
                        action.target.score
                        + (not action.target.stability) * self.world.fail_penalty
                    )
                    counter += 1
                score = sum(score) / counter
            elif method == "Fixed":
                score = 0
            else:
                Warning("Method not recognized")
                pass
            act_seq.score = score
            if verbose:
                print([a.action.__str__() for a in act_seq.actions], " score: ", score)

    class Action_sequence:
        def __init__(self, actions, score=None):
            self.actions = actions
            self.score = score

        def print_actseq(self):
            print(
                "Action sequence:",
                [[str(b) for b in act.action] for act in self.actions],
                "Score:",
                self.score,
            )

    def select_action_seq(self, scored_actions):  # could implement softmax here
        max_score = max(
            [
                action_seq.score
                for action_seq in scored_actions
                if action_seq.score is not None
            ]
        )
        # choose randomly from highest actions
        max_action_seqs = [
            action_seq for action_seq in scored_actions if action_seq.score == max_score
        ]
        seed(self.random_seed)  # fix random seed
        return max_action_seqs[randint(0, len(max_action_seqs) - 1)]

    def act(self, steps=None, planning_horizon=None, verbose=False, scoring=None):
        """Make the agent act, including changing the world state. The agent deliberates once and then acts n steps. To get the agent to deliberate more than once, call action repeatedly. Setting the planning_horizon higher than the steps gives the agent foresight. To get dumb agent behavior, set scoring to Fixed."""
        # Ensure that we have a random seed if none is set
        if self.random_seed is None:
            self.random_seed = randint(0, 99999)
        if scoring is None:
            scoring = self.scoring
        if planning_horizon is None:
            planning_horizon = self.horizon
        if steps is None:
            # if None is provided, act one step
            steps = 1
        if steps == -1:
            # special case for acting as far as we plan
            steps = planning_horizon
        if planning_horizon == 0:  # special case for dumb agent that can't plan at all
            planning_horizon = steps
            scoring = "Fixed"
        if planning_horizon < steps:
            print(
                "Planning horizon must be higher or equal to the steps or 0! Setting the horizon from",
                planning_horizon,
                "to",
                steps,
            )
            planning_horizon = steps
        # check if we even can act
        if self.world.status()[0] != "Ongoing":
            print("Can't act with world in status", self.world.status())
            return [], {"states_evaluated": 0}
        # make ast and score the nodes in it
        ast, win_seq, number_of_states_evaluated = self.build_ast(
            horizon=planning_horizon, verbose=verbose
        )
        if self.first_solution and win_seq is not None:
            # we have stopped building the AST because we found a solution
            chosen_seq = BFS_Lookahead_Agent.Action_sequence(win_seq[0])
            # we want to take the entire sequence at once, so we overwrite steps to act
            steps = len(chosen_seq.actions)
        else:
            # generate action sequences
            act_seqs = self.generate_action_sequences(
                ast, horizon=planning_horizon, include_subsets=True, verbose=verbose
            )
            if (
                act_seqs == []
            ):  # if we can't act. Should be covered by world fail state above.
                print("No possible actions")
                return [], {"states_evaluated": number_of_states_evaluated}
            # score action sequences
            self.score_action_sequences(act_seqs, scoring)
            # choose an action sequence
            chosen_seq = self.select_action_seq(act_seqs)
            if verbose:
                Ast_node.print_tree(ast)
                for act_seq in act_seqs:
                    act_seq.print_actseq()
                print(
                    "Chosen action sequence:",
                    [[str(b) for b in a.action] for a in chosen_seq.actions],
                    "with score: ",
                    chosen_seq.score,
                )
        # take the steps
        # If the chosen sequence is shorter than the steps, only go so far
        for step in range(min([steps, len(chosen_seq.actions)])):
            if self.only_improving_actions:
                # check if action improves the current state of the world
                if not self.world.current_state.is_improvement(
                    chosen_seq.actions[step].action
                ):
                    steps = step  # for logging
                    break
            self.world.apply_action(chosen_seq.actions[step].action)
            if verbose:
                print(
                    "Took step ",
                    step + 1,
                    " with action ",
                    [str(a) for a in chosen_seq.actions[step].action],
                    " and got world state",
                    self.world.current_state,
                )
                self.world.current_state.visual_display(
                    blocking=True, silhouette=self.world.silhouette
                )
        if verbose:
            print("Done, reached world status: ", self.world.status())
            # self.world.current_state.visual_display(blocking=True,silhouette=self.world.silhouette)
        return [tuple([b for b in a.action]) for a in chosen_seq.actions[:steps]], {
            "states_evaluated": number_of_states_evaluated
        }
        # only returns however many steps we actually acted, not the entire sequence


class Ast_node:
    """AST means action space tree. This class serves as a tree for planning. The nodes are states of the world, and the edges are actions. The nodes store scores (if determined), the action merely deterministically move between states."""

    def __init__(self, state, score=None, stability=None, parent_action=None):
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

    def add_action(self, action, target=None):
        if action in [act.action for act in self.actions]:
            Warning("Action ", action, " is already in actionset for this node")
            pass
        else:
            if target is None:  # figure out the target state
                target = Ast_node(self.state.world.transition(action, self.state))
            action = Ast_edge(action, self, target)  # create new action
            action.target.parent_action = action  # set as parent for target state
            self.actions.append(action)  # add action to node
        return action  # this is just for convenience, as the action is also added to the state

    def print_tree(self, level=0):
        """Prints out the tree to the command line in depth first order."""
        print(
            self.state, " score: ", self.score, " stability: ", self.stability, sep=""
        )
        for child, action in [
            (action.target, action.action) for action in self.actions
        ]:
            # remove __str__ for non-blockworld?
            print(
                "\n|",
                "____" * (level + 1),
                " ",
                [str(b) for b in action],
                " → ",
                end="",
                sep="",
            )
            child.print_tree(level + 1)  # pass


class Ast_edge:
    """AST means action space tree. This class simply consists of an action connecting to ast_nodes. The target state will have to be added by the constructing function."""

    def __init__(self, action, source, target=None):
        self.action = action
        self.source = source
        self.target = target
        # could initialize the parent action of the target here—but that would break trees that converge on states again
        if target is None:
            Warning("Node has empty target")


def get_path(leaf):
    """Gets the path from leaf to root in order root -> leaf"""
    if (
        leaf.parent_action is None
    ):  # if we're already at the root, just return empty list (otherwise we'll return [[]])
        return []
    action_sequence = []
    current_node = leaf
    while current_node.parent_action is not None:  # while we're not at the root node
        action_sequence += [current_node.parent_action]
        current_node = current_node.parent_action.source
    action_sequence.reverse()  # reverse the order
    return [action_sequence]
