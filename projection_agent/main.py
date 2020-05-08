from agent import Agent
import blockworld
import random

w = blockworld.Blockworld()
a = Agent(w,2)
s = w.current_state
print(s.possible_actions())
print(s.F1score())
# for i in range(7):
#     print(i)
#     w.apply_action((blockworld.BaseBlock(4,2),7))
#     print(w.current_state.block_map)
# while w.possible_actions() != []:
#     print(w.current_state.block_map)
#     a = random.choice(w.possible_actions())
#     print(a[0].__str__(),a[1])
#     w.apply_action(a)
# w.current_state.visual_display()



ast = a.build_ast()
a.score_ast(ast)
ast.print_tree()
