import unittest

from scoping_simulations.utils.blockworld import Blockworld
from scoping_simulations.utils.blockworld_library import (
    bl_nonoverlapping_simple,
    bl_nonoverlapping_simple_named,
)

blsn = bl_nonoverlapping_simple_named


class TestMatterServer(unittest.TestCase):
    # def test_empty(self):
    #     server = ms.Physics_Server()
    #     self.assertTrue(server.get_stability([]))

    # def test_stonehenge(self):
    #     world = Blockworld(block_library=bl_nonoverlapping_simple)
    #     # actions are baseblocks with an x coordinate as a tuple
    #     stonehenge = [
    #         (blsn['h2'], 1),
    #         (blsn['h2'], 3),
    #         (blsn['v3'], 1),
    #     ]
    #     for action in stonehenge:
    #         world.apply_action(action)
    #         self.assertTrue(world.stability())

    # def test_bad_stonehenge(self):
    #     world = Blockworld(block_library=bl_nonoverlapping_simple)
    #     # actions are baseblocks with an x coordinate as a tuple
    #     stonehenge = [
    #         (blsn['v2'], 1),
    #         (blsn['h3'], 1),
    #         (blsn['v2'], 3),
    #     ]
    #     stables = []
    #     for action in stonehenge:
    #         world.apply_action(action)
    #         stables.append(world.stability())
    #     print(stables)
    #     self.assertTrue(False in stables)
    #     self.assertEqual(stables, [True, False, False])

    # def test_single_overlap(self):
    #     world = Blockworld(block_library=bl_nonoverlapping_simple)
    #     # actions are baseblocks with an x coordinate as a tuple
    #     actions = [
    #         (blsn['v2'], 1),
    #         (blsn['h2'], 1),
    #     ]
    #     for action in actions:
    #         world.apply_action(action)
    #     self.assertFalse(world.stability())

    # def test_single_overlap_other_side(self):
    #     world = Blockworld(block_library=bl_nonoverlapping_simple)
    #     # actions are baseblocks with an x coordinate as a tuple
    #     actions = [
    #         (blsn['v2'], 2),
    #         (blsn['h2'], 1),
    #     ]
    #     for action in actions:
    #         world.apply_action(action)
    #     self.assertFalse(world.stability())

    def test_13_47(self):
        # this is one of the worlds listed as stable in the subgoal node, but should not be (as the intermediate states are instable).
        world = Blockworld(block_library=bl_nonoverlapping_simple)
        # actions are baseblocks with an x coordinate as a tuple
        actions = [
            (blsn["h3"], 1),
            (blsn["v2"], 1),
            (blsn["h2"], 2),
            (blsn["h2"], 3),
            (blsn["v3"], 4),
            (blsn["h3"], 1),
        ]
        stables = []
        for action in actions:
            world.apply_action(action)
            stables.append(world.stability())
        print(stables)
        self.assertTrue(False in stables)


if __name__ == "__main__":
    unittest.main()
