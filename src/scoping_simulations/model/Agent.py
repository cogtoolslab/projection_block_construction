from scoping_simulations.model.utils.Search_Tree import *


class Agent:
    """The super class of all agents. This class cannot be instantiated, instead instantiate specific agents."""

    def __init__(self, world=None, random_seed=None, label="Agent") -> None:
        self.world = world
        self.random_seed = random_seed
        self.label = label

    def __str__(self) -> str:
        """Yields a string representation of the agent"""
        return (
            self.__class__.__name__
            + " random seed: "
            + str(self.random_seed)
            + " label: "
            + self.label
        )

    def __del__(self) -> None:
        """Called when the object is deleted."""
        pass  # TODO: implement destruction of node physics server

    def set_world(self, world) -> None:
        """Sets the world for the agent."""
        self.world = world

    def get_parameters(self) -> dict:
        """Returns dictionary of agent parameters."""
        return {
            "agent_type": self.__class__.__name__,
            "random_seed": self.random_seed,
            "label": self.label,
        }

    def act(self, steps=None, verbose=False) -> None:
        """Makes the agent act, including changing the world state."""
        raise NotImplementedError("This method must be implemented by a subclass.")
