from agent import Agent
from world import Stonehenge

w = Stonehenge()
a = Agent(w,4)
# ast = a.build_ast()
# a.score_ast(ast)
# ast.print_tree()
# act_seqs = a.generate_action_sequences(ast,include_subsets=True)
# a.score_action_sequences(act_seqs,'Final state',True)
a.act(4,verbose=True)
