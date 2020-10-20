# Tools for block construction
Aim of the project is to investigate how people and artificial agents use tools to intervene on the world to make planning easier.

This is based on the environment from [Will McCarthy’s block construction experiments](https://github.com/cogtoolslab/block_construction). Find writeups [here](https://github.com/cogtoolslab/tools_block_construction_LaTeX).

Two main classes: worlds and agents. See the classes and derived classes for details. For minimal example run something like:

```
from model.BFS_Agent import BFS_Agent
import utils.blockworld as bw
import utils.blockworld_library as bl

a = BFS_Agent()
w = bw.Blockworld(silhouette=bl.stonehenge_6_4,
	block_library=bl.bl_stonehenge_6_4)
a.set_world(w)
while w.status() == 'Ongoing':
    a.act(1,verbose=True)
print('Finished with world in state:",w.status())
```

Use `experiment_runner` to run suites of experiments and save them to a data frame in the current directory.

Requirements:
- Python > 3.7
- ptqdm
- pygame *(might not work correctly in a virtualenv)*
- Box2D
- numpy
- PIL
- matplotlib