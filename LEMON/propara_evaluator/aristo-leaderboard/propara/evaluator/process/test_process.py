import unittest
from collections import OrderedDict

from process import process, Process, Conversion, Move, Input, Output
from process.constants import NO_ACTION as NO_ACT, NO_LOCATION as NO_LOC, CREATE, DESTROY, MOVE


class TestProcess(unittest.TestCase):

    def test_qa(self):
        p = Process(
            process_id=514,
            locations=OrderedDict([
                ('glacier', [NO_LOC, NO_LOC, NO_LOC, NO_LOC, NO_LOC, NO_LOC, 'area', 'area']),
                ('snow', ['area', 'area', 'area', 'area', NO_LOC, NO_LOC, NO_LOC, NO_LOC]),
                ('mass', [NO_LOC, NO_LOC, NO_LOC, NO_LOC, NO_LOC, 'area', 'area', 'area'])
            ]),
            actions=OrderedDict([
                ('glacier', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, NO_ACT, CREATE, NO_ACT]),
                ('snow', [NO_ACT, NO_ACT, NO_ACT, DESTROY, NO_ACT, NO_ACT, NO_ACT]),
                ('mass', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, CREATE, NO_ACT, NO_ACT])
            ]),
            num_steps=7,
        )
        self.assertEquals(p.inputs(), [
            Input(participants='snow')
        ])
        self.assertEquals(p.outputs(), [
            Output(participants='glacier'),
            Output(participants='mass')
        ])
        self.assertEquals(p.conversions(), [
            Conversion(destroyed='snow', created='mass', locations='area', step_id='4')
        ])
        self.assertEquals(p.moves(), [])

        p = Process(
            process_id=540,
            locations=OrderedDict([
                ('air', ['unk', 'unk', 'unk', 'bronchiole', 'alveolus', 'unk', 'unk', 'unk', 'unk', 'unk', 'unk']),
                ('oxygen', ['unk', 'unk', 'unk', 'unk', 'unk', 'bloodstream', 'unk', 'unk', 'unk', 'unk', 'unk']),
                ('carbon dioxide',
                 ['unk', 'unk', 'unk', 'unk', 'unk', 'bloodstream', 'bloodstream', 'alveolus', 'bronchiole', 'lung',
                  'body'])
            ]),
            actions=OrderedDict([
                ('air', [NO_ACT, NO_ACT, MOVE, MOVE, MOVE, NO_ACT, NO_ACT, NO_ACT, NO_ACT, NO_ACT]),
                ('oxygen', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, MOVE, MOVE, NO_ACT, NO_ACT, NO_ACT, NO_ACT]),
                ('carbon dioxide', [NO_ACT, NO_ACT, NO_ACT, NO_ACT, MOVE, NO_ACT, MOVE, MOVE, MOVE, MOVE])
            ]),
            num_steps=10,
        )
        self.assertEquals(p.inputs(), [])
        self.assertEquals(p.outputs(), [])
        self.assertEquals(p.conversions(), [])
        self.assertEquals(p.moves(), [
            Move(participants='air', location_before='unk', location_after='bronchiole', step_id='3'),
            Move(participants='air', location_before='bronchiole', location_after='alveolus', step_id='4'),
            Move(participants='air', location_before='alveolus', location_after='unk', step_id='5'),
            Move(participants='oxygen', location_before='unk', location_after='bloodstream', step_id='5'),
            Move(participants='oxygen', location_before='bloodstream', location_after='unk', step_id='6'),
            Move(participants='carbon dioxide', location_before='unk', location_after='bloodstream', step_id='5'),
            Move(participants='carbon dioxide', location_before='bloodstream', location_after='alveolus', step_id='7'),
            Move(participants='carbon dioxide', location_before='alveolus', location_after='bronchiole', step_id='8'),
            Move(participants='carbon dioxide', location_before='bronchiole', location_after='lung', step_id='9'),
            Move(participants='carbon dioxide', location_before='lung', location_after='body', step_id='10'),
        ])

    def test_is_this_action_seq_of_an_input(self):
        self.assertFalse(process._is_this_action_seq_of_an_input([NO_ACT, CREATE, DESTROY, NO_ACT]))
        self.assertFalse(process._is_this_action_seq_of_an_input([CREATE, DESTROY, NO_ACT, NO_ACT]))

    def test_summarize_participants(self):
        self.assertEquals('gasoline OR gas', process._summarize_participants('gasoline; gas'))
        self.assertEquals('gasoline OR gas', process._summarize_participants('gasoline;gas'))

    def test_split_participants(self):
        self.assertEquals(['gasoline', 'gas'], process._split_participants('gasoline; gas'))
        self.assertEquals(['gasoline', 'gas'], process._split_participants('gasoline;gas'))


if __name__ == '__main__':
    unittest.main()
