import pickle


class RefactorUnpickler(pickle.Unpickler):
    """
    In Oct 2023, the code was refactored into a package.
    This unpickler can be used to load pickles from before the refactor.
    """

    def find_class(self, module, name):
        if module == "utils":
            module = "scoping_simulations.utils"
        if module == "model.utils":
            module = "scoping_simulations.model.utils"
        if module == "utils.blockworld":
            module = "scoping_simulations.utils.blockworld"
        if module == "utils.matter_server":
            module = "scoping_simulations.utils.matter_server"
        if module == "stimuli.subgoal_tree":
            module = "scoping_simulations.stimuli.subgoal_tree"
        if module == "stimuli.subgoal_tree_node":
            module = "scoping_simulations.stimuli.subgoal_tree_node"
        if module == "model.utils.decomposition_functions":
            module = "scoping_simulations.model.utils.decomposition_functions"
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            print(
                f"Could not find the following module:\nModule: {module}, Name: {name}"
            )
            raise
