from agent import Agent
from world import Stonehenge

w = Stonehenge()
a = Agent(w,1)
# ast = a.build_ast()
# a.score_ast(ast)
# ast.print_tree()
# act_seqs = a.generate_action_sequences(ast,include_subsets=True)
# a.score_action_sequences(act_seqs,'Final state',True)
a.act(2,verbose=True,scoring='Fixed')
