from agent import Agent
import blockworld
import random
from silhuouettes import stonehenge

w = blockworld.Blockworld(silhouette=stonehenge)
a = Agent(w,1)
s = w.current_state
print(s.possible_actions())
print(s.F1score())
print(s.stability())

for i in range(25):
    print("Iteration:",i)
    print(w.current_state.block_map)
    a = random.choice(w.possible_actions())
    print(a[0].__str__(),a[1])
    w.apply_action(a)
    print(w.current_state.block_map)
    w.current_state.visual_display(blocking=False)
    print("Stability: ",w.current_state.stability(visual_display=True))
# while w.possible_actions() != []:
#     print(w.current_state.block_map)
#     a = random.choice(w.possible_actions())
#     print(a[0].__str__(),a[1])
#     w.apply_action(a)
w.current_state.visual_display()



# ast = a.build_ast()
# a.score_ast(ast)
# ast.print_tree()
