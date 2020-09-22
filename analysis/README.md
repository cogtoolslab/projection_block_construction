This directory contains pickled pandas dataframes.

These can be read in by 

```
df_paths = ['../dataframes/Astar_search.pkl',
           '../dataframes/breadth_search.pkl',
           '../dataframes/beam_search.pkl',
           '../dataframes/MCTS.pkl',
           '../dataframes/Naive_Q_search.pkl']
#load all experiments as one dataframe
df = pd.concat([pd.read_pickle(l) for l in df_paths])
```

### Glossary

**Run**: training (if applicable) and running one agent on one particular silhouette.

**Silhouette**: the particular outline (and set of baseblocks) that the agent has to reconstruct.

**State**: state of the blockworld environment consisting of the blocks that have already been placed in it.