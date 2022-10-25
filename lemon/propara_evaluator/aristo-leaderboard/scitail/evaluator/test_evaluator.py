import os

import evaluator
import unittest
import tempfile
import typing


class TestAccuracy(unittest.TestCase):
    def test_EverythingCorrect(self):
        qa = {"P1": "E", "P2": "N", "P3": "N"}
        p = {"P1": "E", "P2": "N", "P3": "N"}

        self.assertEqual(3.0 / 3.0, evaluator.calculate_accuracy(qa, p))

    def test_EverythingWrong(self):
        qa = {"P1": "E", "P2": "N", "P3": "N"}
        p = {"P1": "N", "P2": "E", "P3": "E"}

        self.assertEqual(0.0 / 3.0, evaluator.calculate_accuracy(qa, p))

    def test_MixedResults(self):
        qa = {"P1": "E", "P2": "N", "P3": "N"}
        p = {"P1": "E", "P2": "N", "P3": "E"}

        self.assertEqual(2.0 / 3.0, evaluator.calculate_accuracy(qa, p))

    def test_ExtraPredictions(self):
        qa = {"P1": "E", "P2": "N", "P3": "N"}
        p = {"P1": "E", "P2": "N", "P3": "N", "PExtra": "E"}

        with self.assertRaises(SystemExit) as context:
            evaluator.calculate_accuracy(qa, p)
        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_EXTRA)

    def test_MissingPredictions(self):
        qa = {"P1": "E", "P2": "N", "P3": "N"}
        p = {"P1": "E", "P2": "N"}

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
            '{"id": "P1", "gold_label": "E"}\n',
            '{"id": "P2", "gold_label": "N"}\n',
            '{"id": "P3", "gold_label": "N"}\n',
        ])
        answers = evaluator.read_answers(t)
        os.remove(t)

        self.assertEqual(answers, {"P1": "E", "P2": "N", "P3": "N"})

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
            '{"id": "P1", "gold_label": "E"}\n',
            '{"id": "P1", "gold_label": "N"}\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_answers(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_ANSWERS_MALFORMED)


class TestReadPredictions(unittest.TestCase):
    def test_ReadPredictions(self):
        t = temp_file_with_contents([
            'P1,E\n',
            '"P2",N\n',
        ])
        predictions = evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(predictions, {
            "P1": "E",
            "P2": "N",
        })

    def test_ReadPredictionsMissingColumn(self):
        t = temp_file_with_contents([
            'P1,E\n',
            '"P2"\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)

    def test_ReadPredictionsRepeated(self):
        t = temp_file_with_contents([
            'P1,E\n',
            'P1,N\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)

    def test_ReadPredictionsCorruptedBadKey(self):
        t = temp_file_with_contents([
            'P1,X\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)

    def test_ReadPredictionsCorruptedEmptyKey(self):
        t = temp_file_with_contents([
            ',N\n',
        ])
        with self.assertRaises(SystemExit) as context:
            evaluator.read_predictions(t)
        os.remove(t)

        self.assertEqual(context.exception.code, evaluator.EXIT_STATUS_PREDICTIONS_MALFORMED)


if __name__ == '__main__':
    unittest.main()
