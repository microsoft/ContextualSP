import os

import evaluator
import unittest
import tempfile
import typing


class TestAccuracy(unittest.TestCase):
    def test_EverythingCorrect(self):
        qa = {"Q1": "A", "Q2": "A", "Q3": "A"}
        p = {"Q1": ["A"], "Q2": ["A"], "Q3": ["A"]}

        self.assertEqual(3.0 / 3.0, evaluator.calculate_accuracy(qa, p))

    def test_EverythingWrong(self):
        qa = {"Q1": "A", "Q2": "A", "Q3": "A"}
        p = {"Q1": ["B"], "Q2": ["B"], "Q3": ["B"]}

        self.assertEqual(0.0 / 3.0, evaluator.calculate_accuracy(qa, p))

    def test_MixedResults(self):
        qa = {"Q1": "A", "Q2": "A", "Q3": "A"}
        p = {"Q1": ["A"], "Q2": ["A"], "Q3": ["B"]}

        self.assertEqual(2.0 / 3.0, evaluator.calculate_accuracy(qa, p))

    def test_PartialGuess(self):
        qa = {"Q1": "A", "Q2": "A", "Q3": "A"}
        p = {"Q1": ["A", "B"], "Q2": ["B"], "Q3": ["B"]}

        self.assertEqual(0.5 / 3, evaluator.calculate_accuracy(qa, p))

    def test_ExtraPredictions(self):
        qa = {"Q1": "A", "Q2": "A", "Q3": "A"}
        p = {"Q1": ["A"], "Q2": ["A"], "Q3": ["B"], "QExtra": ["X"]}

        with self.assertRaises(SystemExit) as context:
            evaluator.calculate_accuracy(qa, p)
        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_EXTRA)

    def test_MissingPredictions(self):
        qa = {"Q1": "A", "Q2": "A", "Q3": "A"}
        p = {"Q1": ["A"], "Q2": ["A"]}

        with self.assertRaises(SystemExit) as context:
            evaluator.calculate_accuracy(qa, p)
        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTION_MISSING)


def temp_file_with_contents(lines: typing.List[str]) -> str:
    t = tempfile.NamedTemporaryFile(mode='wt', delete=False)
    t.writelines(lines)
    t.close()
    return t.name


class TestReadAnswers(unittest.TestCase):
    def test_ReadAnswers(self):
        t = temp_file_with_contents([
            '{"id": "Q1", "answerKey": "A"}\n',
            '{"id": "Q2", "answerKey": "B"}\n',
            '{"id": "Q3", "answerKey": "C"}\n',
        ])
        answers = evaluator.read_answers(t)
        os.remove(t)

        self.assertEqual(answers, {"Q1": "A", "Q2": "B", "Q3": "C"})

    def test_ReadAnswersEmpty(self):
        t = temp_file_with_contents([])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_answers(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_ANSWERS_MALFORMED)

    def test_ReadAnswersCorrupted(self):
        t = temp_file_with_contents(['this is not json'])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_answers(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_ANSWERS_MALFORMED)

    def test_ReadAnswersRepeated(self):
        t = temp_file_with_contents([
            '{"id": "Q1", "answerKey": "A"}\n',
            '{"id": "Q1", "answerKey": "B"}\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_answers(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_ANSWERS_MALFORMED)


class TestReadPredictions(unittest.TestCase):
    def test_ReadPredictions(self):
        t = temp_file_with_contents([
            'Q1,A\n',
            '"Q2",A;B\n',
            'Q3,"A;B;C"\n',
        ])
        predictions = evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(predictions, {
            "Q1": ["A"],
            "Q2": ["A", "B"],
            "Q3": ["A", "B", "C"],
        })

    def test_ReadPredictionsMissingColumn(self):
        t = temp_file_with_contents([
            'Q1,A\n',
            '"Q2"\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)

    def test_ReadPredictionsRepeated(self):
        t = temp_file_with_contents([
            'Q1,A\n',
            'Q1,A\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)

    def test_ReadPredictionsCorruptedEmptyKey(self):
        t = temp_file_with_contents([
            ',A\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)

    def test_ReadPredictionsCorruptedEmptyLabels(self):
        t = temp_file_with_contents([
            'Q1,A;\n',
        ])
        with self.assertRaises(SystemExit) as context:
            p = evaluator.read_predictions(t)
            print(p)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)


if __name__ == '__main__':
    unittest.main()
