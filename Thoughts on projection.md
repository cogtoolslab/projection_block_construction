# Projection

![image-20200406042220505](Projection Master.assets/image-20200406042220505.png)

## What is projection in the context of block construction?

Maybe better to think about a graduation from all planning to all action (ie greedy/myopic planning)

##How to model projection/simulation?

#### Planning horizon

*promising*

Length of the planning horizon determines the graduation between all planning (unlimited steps/how many it takes to solve the problem) and all action (1—just greedily chose the action).

Requires way of scoring closeness of current build to end goal, then for every step go over all possibilities and score them. 

Needs scoring function. Simple solution: % of silhouette filled with nothing outside of it. Doesn’t integrate stability, though. Do I need a special model for this?  It looks like there already is a scoring function for incomplete figures in Will stuff (as it gives you points)

Problem: branching factor for long planning horizons. Maybe stochastic tree search/evolutionary things/other clever RL solutions? 	

## How to disentangle projection and simulation in an experiment?

#### force participants to preselect the pieces they’ll put

#### only show the outline briefly

#### use something like bubble vision to make looking at the outline costly (ie make it a cost to show parts of it)

This allow us to see when people are looking ahead. Could be nicely implemented on the figure on the right. Would allow us to see which parts of the shadow are considered at what part in the planning process

Needs a cost—time? How would time translate to the model? Better to take out of total reward. Maybe we could use a Gaussian process model (somehow), as it tells us where to explore using variance?

Is this not just a test of working memory? Isn’t all of it?

#### control/track the use of holding pieces in place to see

Will has the data? Planning to do something like this? 

#### distribution of thinking time

how long do participants think before picking up each piece: long time in the beginning (simulation), or spaced between each new piece (action)?

## Papers

### Tenenbaum on different kinds of physics simulation

Heuristics, forward & backward simulation

### Vul