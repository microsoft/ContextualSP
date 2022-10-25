# import pytest

import sys
sys.path.append('../../../')
from strongsup.example import Example, Context
from strongsup.rlong.exploration_policy import AlchemyOraclePathFinder
from strongsup.rlong.state import RLongAlchemyState
from strongsup.rlong.world import RLongAlchemyWorld
from strongsup.rlong.value import RLongStateValue


class TestAlchemyExplorationPolicy(object):

    def test_exploration(self):
        initial_state = RLongAlchemyState.from_raw_string(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg')
        final_state = RLongAlchemyState.from_raw_string(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:_')
        num_steps = 2
        world = RLongAlchemyWorld(initial_state)
        context = Context(world, [[""], [""]])
        ex = Example(context, answer=[RLongStateValue(final_state)])
        print()
        print(' INIT:', initial_state)
        print('FINAL:', final_state)
        print('STEPS:', num_steps)
        path_finder = AlchemyOraclePathFinder(ex, debug=True)
        found = set()
        for path in path_finder.all_actual_paths:
            finalized = ex.context.executor.finalize(path.denotation)
            assert finalized[0].state == final_state
            found.add(' '.join(str(x) for x in path.decisions))
        assert 'all-objects -1 index 2 ADrain -1 H1 -1 H2 -1 H0' in found
        assert 'all-objects -1 index all-objects 2 index APour 1 H2 X1/1 ADrain' in found

if __name__ == '__main__':
    tester = TestAlchemyExplorationPolicy()
    tester.test_exploration()