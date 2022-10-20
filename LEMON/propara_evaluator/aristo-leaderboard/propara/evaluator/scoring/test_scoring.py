import unittest

from process import ProcessSummary, Conversion, Move, Input, Output
from scoring import question, QuestionScores


class TestScoring(unittest.TestCase):

    def test_compare_locations(self):
        self.assertEquals(question._compare_locations('', ''), 1.0)
        self.assertEquals(question._compare_locations('', '-'), 0.0)
        self.assertEquals(question._compare_locations('?', '?'), 1.0)
        self.assertEquals(question._compare_locations('plant OR leaf', 'leaf'), 1.0)
        self.assertEquals(question._compare_locations('', 'leaf'), 0.0)
        self.assertEquals(question._compare_locations('-', 'leaf'), 0.0)
        self.assertEquals(question._compare_locations('plant  OR leaf', 'leaf'), 1.0)
        self.assertEquals(question._compare_locations('dew OR rain', 'water OR dew'), 1.0)
        self.assertEquals(question._compare_locations('dew', 'dew AND sun'), 0.5)
        self.assertEquals(question._compare_locations('dew AND sun', 'dew'), 0.5)
        self.assertEquals(question._compare_locations('dew AND sun', 'dew AND blah1 AND blah2'), 0.25)
        self.assertEquals(question._compare_locations('dew AND rain', 'water OR dew'), 0.5)
        self.assertEquals(question._compare_locations('water OR dew AND sun', 'dew OR rain'), 0.5)

    def test_score_tuple_question(self):
        answers = [
            Move(participants='plant OR leaf', location_before='root', location_after='earth', step_id='event2'),
            Move(participants='leaf', location_before='soil', location_after='mud', step_id='event2'),
        ]
        predictions = [
            Move(participants='plants OR leaf', location_before='root', location_after='earth', step_id='event2'),
            Move(participants='plant', location_before='mud OR plant', location_after='room OR earth', step_id='event2')
        ]
        predictions_longer = [
            Move(participants='plants OR leaf', location_before='root', location_after='earth', step_id='event2'),
            Move(participants='plant', location_before='mud OR plant', location_after='room OR earth',
                 step_id='event2'),
            Move(participants='tree', location_before='monkey', location_after='earth', step_id='event2'),
        ]
        predictions_shorter = [
            Move(participants='plants OR leaf', location_before='root', location_after='earth', step_id='event2'),
        ]

        self.assertEquals(question._score_moves(answers, predictions).F1(), 0.8333333333333333)
        self.assertEquals(question._score_moves(answers, predictions_shorter).F1(), 0.8)
        self.assertEquals(question._score_moves(answers, predictions_longer).F1(), 0.6666666666666666)
        self.assertEquals(question._score_moves(answers, []).F1(), 0.0)
        self.assertEquals(question._score_moves([], predictions).F1(), 0.0)
        self.assertEquals(question._score_moves([], []).F1(), 1.0)

    def test_score_conversion_pair(self):
        self.assertEquals(question._score_conversion_pair(
            Conversion(destroyed='animal OR monkey', created='tree', locations='branch', step_id='event1'),
            Conversion(destroyed='animal', created='tree', locations='branch', step_id='event1')
        ), 1.0)

        self.assertEquals(question._score_conversion_pair(
            Conversion(destroyed='plant OR leaf', created='root', locations='earth', step_id='event2'),
            Conversion(destroyed='leaf', created='root OR plant', locations='soil OR earth', step_id='event2'),
        ), 1.0)

        # plants should match plant.
        self.assertEquals(question._score_conversion_pair(
            Conversion(destroyed='plants OR leaf', created='root', locations='earth', step_id='event2'),
            Conversion(destroyed='leaf', created='root OR plant', locations='soil OR earth', step_id='event2'),
        ), 1.0)

        # identical conversion, but mismatching step_ids
        self.assertEquals(question._score_conversion_pair(
            Conversion(destroyed='foo', created='bar', locations='baz', step_id='eventX'),
            Conversion(destroyed='foo', created='bar', locations='baz', step_id='eventY'),
        ), 0.0)

    def test_score_input_pair(self):
        self.assertEquals(question._score_input_pair(
            Input(participants=''), Input(participants='-')
        ), 0)
        self.assertEquals(question._score_input_pair(
            Input(participants='plant OR leaf'), Input(participants='leaf')
        ), 1)
        self.assertEquals(question._score_input_pair(
            Input(participants='?'), Input(participants='?')
        ), 1)

    def test_calculate(self):
        score = QuestionScores.from_summaries(
            answer=ProcessSummary(
                1,
                inputs=[Input(participants='plant')],
                outputs=[Output(participants='plant OR leaf'), Output(participants='soil')],
                conversions=[],
                moves=[
                    Move(participants="plant OR leaf",
                         location_before="root",
                         location_after="event2",
                         step_id="event2")
                ]
            ),
            prediction=ProcessSummary(
                1,
                inputs=[Input(participants='tree')],
                outputs=[Output(participants='leaf'), Output(participants='mud')],
                conversions=[Conversion(destroyed='tree', created='root', locations='soil', step_id='event1')],
                moves=[Move(participants='plant', location_before='root', location_after='soil', step_id='event2')]
            ),
        )
        self.assertEquals(score.conversions.F1(), 0.0)

        score = QuestionScores.from_summaries(
            answer=ProcessSummary(
                1,
                inputs=[Input(participants='monkey'), Input(participants='ape')],
                outputs=[Output(participants='langur OR langer')],
                conversions=[
                    Conversion(destroyed='animal OR monkey', created='tree', locations='branch', step_id='event1')],
                moves=[],
            ),
            prediction=ProcessSummary(
                1,
                inputs=[Input(participants='langur'), Input(participants='ape')],
                outputs=[Output(participants='monkey')],
                conversions=[Conversion(destroyed='animal', created='tree', locations='branch', step_id='event1')],
                moves=[],
            ),
        )
        self.assertEquals(score.conversions.F1(), 1.0)

    def test_score_empty_answers(self):
        score = QuestionScores.from_summaries(
            answer=ProcessSummary(process_id=1, inputs=[], outputs=[], conversions=[], moves=[]),
            prediction=ProcessSummary(process_id=1, inputs=[], outputs=[], conversions=[], moves=[])
        )
        self.assertEquals(score.inputs.F1(), 1.0)
        self.assertEquals(score.outputs.F1(), 1.0)
        self.assertEquals(score.conversions.F1(), 1.0)

    def test_score(self):
        i1 = Input(participants='xxx')
        i2 = Input(participants='yyy')
        i3 = Input(participants='zzz')
        answers = [i1]
        predictions = [i2, i3]

        def scoring_function(answer: Input, prediction: Input) -> float:
            return 1.0

        score = question._score(answers, predictions, scoring_function)
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.F1(), 1.0)

    def test_score2(self):
        i1 = Input(participants='xxx')
        i2 = Input(participants='yyy')
        i3 = Input(participants='zzz')
        answers = [i1]
        predictions = [i2, i3]

        def scoring_function(answer: Input, prediction: Input) -> float:
            if (answer, prediction) == (i1, i2):
                return 1.0
            return 0.0

        score = question._score(answers, predictions, scoring_function)
        self.assertEqual(score.precision, 0.5)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.F1(), 2 / 3)


if __name__ == '__main__':
    unittest.main()

