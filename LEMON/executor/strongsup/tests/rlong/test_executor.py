# import pytest
import sys
sys.path.append('../../../')
from strongsup.rlong.executor import RLongExecutor
from strongsup.rlong.predicate import RLongPredicate
from strongsup.rlong.state import \
        RLongAlchemyState, RLongSceneState, RLongTangramsState, RLongUndogramsState


class RLongExecutorTester(object):

    def prepare_state(self, state):
        if not isinstance(state, self.STATE_CLASS):
            state = self.STATE_CLASS.from_raw_string(state)
        return state

    def prepare_lf(self, lf):
        if isinstance(lf, str):
            lf = lf.split()
        if not all(isinstance(x, RLongPredicate) for x in lf):
            lf = [x if isinstance(x, RLongPredicate) else RLongPredicate(x)
                    for x in lf]
        return lf

    def assert_good(self, initial_state, lf, final_state):
        initial_state = self.prepare_state(initial_state)
        final_state = self.prepare_state(final_state)
        # executor = RLongExecutor(initial_state, debug=True)
        executor = RLongExecutor(initial_state, debug=False)
        lf = self.prepare_lf(lf)
        # print(('=' * 10, lf, '=' * 10))
        # Direct execution
        denotation = executor.execute(lf)
        assert denotation.world_state == final_state
        # Token-by-token execution
        denotation = None
        for x in lf:
            denotation = executor.execute_predicate(x, denotation)
        assert denotation.world_state == final_state

    def assert_bad(self, initial_state, lf):
        initial_state = self.prepare_state(initial_state)
        # executor = RLongExecutor(initial_state, debug=True)
        executor = RLongExecutor(initial_state, debug=False)
        lf = self.prepare_lf(lf)
        # print(('=' * 10, lf, '=' * 10))
        # Direct execution
        try:
            denotation = executor.execute(lf)
            assert False, 'No error: denotation = {}'.format(denotation)
        except:
            pass
        # Token-by-token execution
        denotation = None
        for x in lf:
            try:
                denotation = executor.execute_predicate(x, denotation)
            except Exception as e:
                denotation = e
        assert isinstance(denotation, Exception)
    

################################

class TestAlchemyExecutor(RLongExecutorTester):
    STATE_CLASS = RLongAlchemyState

    def test_simple(self):
        # 1:ggg 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg throw out two units of first beaker 1:g 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg throw out fifth beaker 1:g 2:_ 3:_ 4:_ 5:_ 6 :ooo 7:gggg throw out first one 1:_ 2:_ 3:_ 4:_ 5:_ 6:ooo 7:gggg throw out orange beaker 1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:gggg throw out one unit of green 1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:ggg
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg',
                'all-objects 1 index 2 ADrain',
                '1:g 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg',
                'g PColor 1 index 2 ADrain',
                '1:g 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'r PColor 1 ADrain',
                '1:ggg 2:_ 3:_ 4:_ 5:o 6:ooo 7:gggg')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'all-objects 5 index r PColor APour',
                '1:ggg 2:_ 3:_ 4:ro 5:_ 6:ooo 7:gggg')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'all-objects -1 index X1/2 ADrain',
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gg')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'all-objects -1 index X1/2 ADrain all-objects -1 index 1 ADrain',
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:g')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'all-objects -1 index X1/2 ADrain -1 H1 1 ADrain',
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:g')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'all-objects -1 index X1/2 ADrain -1 H1 1 H2 -1 H0',
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:_')
        self.assert_good(
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:gggg',
                'all-objects -1 index X1/2 ADrain -1 H1 -1 H2 -1 H0',
                '1:ggg 2:_ 3:_ 4:r 5:o 6:ooo 7:_')


class TestSceneExecutor(RLongExecutorTester):
    STATE_CLASS = RLongSceneState

    def test_simple(self):
        # train-1100  1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo  a man in a green shirt and an orange hat stands near the middle and a man in a yellow shirt and an orange hat stands on the far right 1:r_ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo  a man in a red shirt and no hat enters and stands on the far left 1:r_ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:y_ 10:yo  a man in a yellow shirt and no hat joins and stands next to the man in the yellow shirt and orange hat  1:r_ 2:__ 3:__ 4:__ 5:__ 6:go 7:__ 8:__ 9:y_ 10:yo  the man in the yellow shirt and orange hat moves next to the man in the green shirt and orange hat, he stands on the right  1:r_ 2:__ 3:__ 4:__ 5:__ 6:go 7:yo 8:__ 9:y_ 10:__  a man in a green shirt and no hat joins and stands next to the man in the green shirt and orange hat  1:r_ 2:__ 3:__ 4:__ 5:g_ 6:go 7:yo 8:__ 9:y_ 10:__
        # Well the sentences were really wrong ...
        self.assert_good(
                '1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo',
                '1 r e ACreate',
                '1:r_ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo')
        self.assert_bad(
                '1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo',
                '7 r e ACreate')
        self.assert_good(
                '1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo',
                '1 r e ACreate 9 y e ACreate',
                '1:r_ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:y_ 10:yo')
        self.assert_good(
                '1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo',
                '1 r e ACreate y o DShirtHat PLeft y e ACreate',
                '1:r_ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:y_ 10:yo')
        self.assert_good(
                '1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo',
                '1 r e ACreate y o DShirtHat PLeft y e ACreate ' +
                'g PShirt g PShirt PLeft AMove',
                '1:r_ 2:__ 3:__ 4:__ 5:__ 6:go 7:__ 8:__ 9:y_ 10:yo')
        self.assert_good(
                '1:__ 2:__ 3:__ 4:__ 5:__ 6:__ 7:go 8:__ 9:__ 10:yo',
                '1 r e ACreate y o DShirtHat PLeft y e ACreate ' +
                'g PShirt g PShirt PLeft AMove ' +
                'all-objects 2 index ALeave',
                '1:r_ 2:__ 3:__ 4:__ 5:__ 6:__ 7:__ 8:__ 9:y_ 10:yo')
        # train-1101  1:bo 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__  the person in an orange hat moves to the left of the person in a red hat  1:__ 2:__ 3:bo 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__  he then disappears  1:__ 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__  then a person in orange appears on the far right  1:__ 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:o_  he then disappears  1:__ 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__  the person in blue in a red hat moves to the far left 1:br 2:__ 3:__ 4:__ 5:__ 6:__ 7:__ 8:__ 9:__ 10:__
        self.assert_good(
                '1:bo 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__',
                'o PHat r PHat PLeft AMove',
                '1:__ 2:__ 3:bo 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__')
        self.assert_good(
                '1:bo 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__',
                'o PHat r PHat PLeft AMove -1 H1 ALeave',
                '1:__ 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__')
        self.assert_good(
                '1:bo 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__',
                'o PHat r PHat PLeft AMove -1 H1 ALeave -1 o e ACreate',
                '1:__ 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:o_')
        self.assert_good(
                '1:bo 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__',
                'o PHat r PHat PLeft AMove -1 H1 ALeave -1 o e ACreate ' +
                '-1 H1 ALeave',
                '1:__ 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__')
        self.assert_good(
                '1:bo 2:__ 3:__ 4:br 5:__ 6:__ 7:__ 8:__ 9:__ 10:__',
                'o PHat r PHat PLeft AMove -1 H1 ALeave -1 o e ACreate ' +
                '-1 H1 ALeave b r DShirtHat 1 AMove',
                '1:br 2:__ 3:__ 4:__ 5:__ 6:__ 7:__ 8:__ 9:__ 10:__')


class TestTangramsExecutor(RLongExecutorTester):
    STATE_CLASS = RLongTangramsState

    def test_simple(self):
        # train-437 1:2 2:1 3:4 4:0 5:3 delete the second object from the left  1:2 2:4 3:0 4:3 delete the leftmost object  1:4 2:0 3:3 swap the leftmost and the rightmost objects 1:3 2:0 3:4 swap them again 1:4 2:0 3:3 add back the object we removed on step 1  1:1 2:4 3:0 4:3
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove',
                '1:2 2:4 3:0 4:3')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove',
                '1:4 2:0 3:3')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects -1 index ASwap',
                '1:3 2:0 3:4')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects -1 index ASwap ' +
                '-1 H1 -1 H2 ASwap',
                '1:4 2:0 3:3')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects -1 index ASwap ' +
                '-1 H1 -1 H2 ASwap 1 1 H1 AAdd',
                '1:1 2:4 3:0 4:3')
        # train-438 1:0 2:2 3:4 4:3 5:1 swap the second and third figures 1:0 2:4 3:2 4:3 5:1 remove the second figure  1:0 2:2 3:3 4:1 swap the second and third figures 1:0 2:3 3:2 4:1 remove the third figure 1:0 2:3 3:1 add back the figure removed in step 2, and place in the third space 1:0 2:3 3:4 4:1
        self.assert_good(
                '1:0 2:2 3:4 4:3 5:1',
                'all-objects 2 index all-objects 3 index ASwap',
                '1:0 2:4 3:2 4:3 5:1')
        self.assert_good(
                '1:0 2:2 3:4 4:3 5:1',
                'all-objects 2 index all-objects 3 index ASwap ' +
                'all-objects 2 index ARemove',
                '1:0 2:2 3:3 4:1')
        self.assert_good(
                '1:0 2:2 3:4 4:3 5:1',
                'all-objects 2 index all-objects 3 index ASwap ' +
                'all-objects 2 index ARemove ' +
                'all-objects 2 index all-objects 3 index ASwap ' +
                'all-objects 3 index ARemove ' +
                '3 2 H1 AAdd',
                '1:0 2:3 3:4 4:1')


class TestUndogramsExecutor(RLongExecutorTester):
    STATE_CLASS = RLongUndogramsState

    def test_simple(self):
        # train-437 1:2 2:1 3:4 4:0 5:3 delete the second object from the left  1:2 2:4 3:0 4:3 delete the leftmost object  1:4 2:0 3:3 swap the leftmost and the rightmost objects 1:3 2:0 3:4 swap them again 1:4 2:0 3:3 add back the object we removed on step 1  1:1 2:4 3:0 4:3
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove',
                '1:2 2:4 3:0 4:3')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove',
                '1:4 2:0 3:3')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects -1 index ASwap',
                '1:3 2:0 3:4')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects -1 index ASwap ' +
                '-1 H1 -1 H2 ASwap',
                '1:4 2:0 3:3')
        self.assert_good(
                '1:2 2:1 3:4 4:0 5:3',
                'all-objects 2 index ARemove all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects -1 index ASwap ' +
                '-1 H1 -1 H2 ASwap 1 1 H2 AAdd',
                '1:1 2:4 3:0 4:3')
        # train-438 1:0 2:2 3:4 4:3 5:1 swap the second and third figures 1:0 2:4 3:2 4:3 5:1 remove the second figure  1:0 2:2 3:3 4:1 swap the second and third figures 1:0 2:3 3:2 4:1 remove the third figure 1:0 2:3 3:1 add back the figure removed in step 2, and place in the third space 1:0 2:3 3:4 4:1
        self.assert_good(
                '1:0 2:2 3:4 4:3 5:1',
                'all-objects 2 index all-objects 3 index ASwap',
                '1:0 2:4 3:2 4:3 5:1')
        self.assert_good(
                '1:0 2:2 3:4 4:3 5:1',
                'all-objects 2 index all-objects 3 index ASwap ' +
                'all-objects 2 index ARemove',
                '1:0 2:2 3:3 4:1')
        self.assert_good(
                '1:0 2:2 3:4 4:3 5:1',
                'all-objects 2 index all-objects 3 index ASwap ' +
                'all-objects 2 index ARemove ' +
                'all-objects 2 index all-objects 3 index ASwap ' +
                'all-objects 3 index ARemove ' +
                '3 2 H2 AAdd',
                '1:0 2:3 3:4 4:1')

    def test_undo(self):
        # train-440 1:2 2:1 3:3 4:0 5:4 delete the rightmost figure 1:2 2:1 3:3 4:0 undo step 1 1:2 2:1 3:3 4:0 5:4 delete the 1st figure 1:1 2:3 3:0 4:4 swap the 1st and 3rd figure 1:0 2:3 3:1 4:4 undo step 4 1:1 2:3 3:0 4:4
        self.assert_good(
                '1:2 2:1 3:3 4:0 5:4',
                'all-objects -1 index ARemove',
                '1:2 2:1 3:3 4:0')
        self.assert_good(
                '1:2 2:1 3:3 4:0 5:4',
                'all-objects -1 index ARemove ' +
                '1 H1 1 H2 1 HUndo',
                '1:2 2:1 3:3 4:0 5:4')
        self.assert_good(
                '1:2 2:1 3:3 4:0 5:4',
                'all-objects -1 index ARemove ' +
                '1 H1 1 H2 1 HUndo ' +
                'all-objects 1 index ARemove ' +
                'all-objects 1 index all-objects 3 index ASwap ' +
                '4 H1 4 H2 4 HUndo',
                '1:1 2:3 3:0 4:4')

if __name__ == '__main__':
    
    # tester = TestAlchemyExecutor()
    # tester.test_simple()

    # tester = TestSceneExecutor()
    # tester.test_simple()

    # tester = TestTangramsExecutor()
    # tester.test_simple()

    tester = TestUndogramsExecutor()
    # tester.test_simple()
    tester.test_undo()