# Projection in block construction
This is based on the environment from [Will McCarthy’s block construction experiments](https://github.com/cogtoolslab/block_construction). Aim of the project is to investigate how agents plan by selectively performing epistemic actions: actions that serve to make planning easier. Eventually, we’re planning to investigate projection: internal and external augmentations of the visual environment in the support of planning.

Two main classes: worlds and agents. See the classes and derived classes for details. For minimal example run something like:

```
from agent import Agent
import blockworld as bw
import blockworld_library as bl

a = Agent()
w = bw.Blockworld(silhouette=bl.stonehenge_6_4,
	block_library=bl.bl_stonehenge_6_4)
a.set_world(w)
while w.status() == 'Ongoing':
    a.act(-1,verbose=True)
print('Finished with world in state:",w.status())
```

Use `experiment_runner` to run suites of experiments and save them to a data frame in the current directory.