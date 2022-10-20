from typing import Dict, NamedTuple, Iterable

from evaluation.metric import Metric


class EvaluationAverages(NamedTuple):
    inputs: float
    outputs: float
    conversions: float
    moves: float
    overall: float


class Evaluation:
    def __init__(self, scores: Dict[int, "QuestionScores"]) -> None:  # type: ignore
        precision = Evaluation._precision(scores.values())
        recall = Evaluation._recall(scores.values())

        self.inputs = Metric(precision=precision.inputs, recall=recall.inputs)
        self.outputs = Metric(precision=precision.outputs, recall=recall.outputs)
        self.conversions = Metric(precision=precision.conversions, recall=recall.conversions)
        self.moves = Metric(precision=precision.moves, recall=recall.moves)
        self.overall = Metric(precision=precision.overall, recall=recall.overall)

    @staticmethod
    def _precision(scores: Iterable["QuestionScores"]) -> EvaluationAverages:  # type: ignore
        inputs = 0.0
        outputs = 0.0
        conversions = 0.0
        moves = 0.0

        num_processes = 0
        for score in scores:
            inputs += score.inputs.precision
            outputs += score.outputs.precision
            conversions += score.conversions.precision
            moves += score.moves.precision
            num_processes += 1

        inputs_avg = round(inputs / num_processes, 3)
        outputs_avg = round(outputs / num_processes, 3)
        conversions_avg = round(conversions / num_processes, 3)
        moves_avg = round(moves / num_processes, 3)

        overall = (inputs_avg + outputs_avg + conversions_avg + moves_avg) / 4

        return EvaluationAverages(
            inputs=inputs_avg,
            outputs=outputs_avg,
            conversions=conversions_avg,
            moves=moves_avg,
            overall=overall,
        )

    @staticmethod
    def _recall(scores: Iterable["QuestionScores"]) -> EvaluationAverages:  # type: ignore
        inputs = 0.0
        outputs = 0.0
        conversions = 0.0
        moves = 0.0

        num_processes = 0
        for score in scores:
            inputs += score.inputs.recall
            outputs += score.outputs.recall
            conversions += score.conversions.recall
            moves += score.moves.recall
            num_processes += 1

        inputs_avg = round(inputs / num_processes, 3)
        outputs_avg = round(outputs / num_processes, 3)
        conversions_avg = round(conversions / num_processes, 3)
        moves_avg = round(moves / num_processes, 3)

        overall = (inputs_avg + outputs_avg + conversions_avg + moves_avg) / 4

        return EvaluationAverages(
            inputs=inputs_avg,
            outputs=outputs_avg,
            conversions=conversions_avg,
            moves=moves_avg,
            overall=overall,
        )
