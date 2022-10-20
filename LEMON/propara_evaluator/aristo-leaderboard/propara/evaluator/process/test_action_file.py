import unittest
from collections import OrderedDict

from process.action_file import ActionFile
from process.constants import NO_ACTION as NO_ACT
from process.constants import NO_LOCATION as NO_LOC, CREATE, DESTROY, MOVE


class TestSummarize(unittest.TestCase):
    def test_load(self):
        # Spot-check values loaded from an action file
        actionfile = ActionFile.from_file('testfiles-0/answers.tsv')

        # Process 514
        self.assertEquals(
            OrderedDict([
                ('glacier', [NO_LOC, NO_LOC, NO_LOC, NO_LOC, NO_LOC, NO_LOC, 'area', 'area']),
                ('mass', [NO_LOC, NO_LOC, NO_LOC, NO_LOC, NO_LOC, 'area', 'area', 'area']),
                ('snow', ['area', 'area', 'area', 'area', NO_LOC, NO_LOC, NO_LOC, NO_LOC]),
            ]),
            actionfile.locations[514],
        )
        self.assertEquals(
            OrderedDict([
                ('glacier', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, NO_ACT, CREATE, NO_ACT]),
                ('mass', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, CREATE, NO_ACT, NO_ACT]),
                ('snow', [NO_ACT, NO_ACT, NO_ACT, DESTROY, NO_ACT, NO_ACT, NO_ACT]),
            ]),
            actionfile.actions[514],
        )
        self.assertEquals(7, actionfile.num_sentences[514])

        # Process 540
        self.assertEquals(
            OrderedDict([
                ('air', ['unk', 'unk', 'unk', 'bronchiole', 'alveolus', 'unk', 'unk', 'unk', 'unk', 'unk', 'unk']),
                ('carbon dioxide',
                 ['unk', 'unk', 'unk', 'unk', 'unk', 'bloodstream', 'bloodstream', 'alveolus', 'bronchiole', 'lung',
                  'body']),
                ('oxygen', ['unk', 'unk', 'unk', 'unk', 'unk', 'bloodstream', 'unk', 'unk', 'unk', 'unk', 'unk']),
            ]),
            actionfile.locations[540],
        )
        self.assertEquals(
            OrderedDict([
                ('air', [NO_ACT, NO_ACT, MOVE, MOVE, MOVE, NO_ACT, NO_ACT, NO_ACT, NO_ACT, NO_ACT]),
                ('carbon dioxide', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, MOVE, NO_ACT, MOVE, MOVE, MOVE, MOVE]),
                ('oxygen', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, MOVE, MOVE, NO_ACT, NO_ACT, NO_ACT, NO_ACT]),
            ]),
            actionfile.actions[540]
        )
        self.assertEquals(10, actionfile.num_sentences[540])

        if __name__ == '__main__':
            unittest.main()
