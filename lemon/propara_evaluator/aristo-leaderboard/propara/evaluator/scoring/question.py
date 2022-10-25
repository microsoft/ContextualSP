from typing import List, NamedTuple, Callable, TypeVar, Optional

from evaluation.metric import Metric
from text import terms
from process import ProcessSummary, Conversion, Move, Input, Output

# Question types used in functions here
QType = TypeVar("QType", Input, Output, Conversion, Move)


class QuestionScores(NamedTuple):
    inputs: Metric
    outputs: Metric
    conversions: Metric
    moves: Metric

    @classmethod
    def from_summaries(cls, answer: ProcessSummary, prediction: ProcessSummary):
        return cls(
            inputs=_score_inputs(answer.inputs, prediction.inputs),
            outputs=_score_outputs(answer.outputs, prediction.outputs),
            conversions=_score_conversions(answer.conversions, prediction.conversions),
            moves=_score_moves(answer.moves, prediction.moves),
        )


def _edgecases(answers: List[QType], predictions: List[QType]) -> Optional[Metric]:
    if len(answers) == 0 and len(predictions) == 0:
        return Metric(precision=1.0, recall=1.0)

    if len(answers) == 0:
        return Metric(precision=0.0, recall=1.0)

    if len(predictions) == 0:
        return Metric(precision=1.0, recall=0.0)

    return None


def _score_inputs(answers: List[Input], predictions: List[Input]) -> Metric:
    m = _edgecases(answers, predictions)
    if m:
        return m

    return _score(answers, predictions, _score_input_pair)


def _score_input_pair(answer: Input, prediction: Input) -> float:
    return _compare_participants(answer.participants, prediction.participants)


def _score_outputs(answers: List[Output], predictions: List[Output]) -> Metric:
    m = _edgecases(answers, predictions)
    if m:
        return m

    return _score(answers, predictions, _score_output_pair)


def _score_output_pair(answer: Output, prediction: Output) -> float:
    return _compare_participants(answer.participants, prediction.participants)


def _score_conversions(answers: List[Conversion], predictions: List[Conversion]) -> Metric:
    m = _edgecases(answers, predictions)
    if m:
        return m

    return _score(answers, predictions, _score_conversion_pair)


def _score_conversion_pair(answer: Conversion, prediction: Conversion) -> float:
    if answer.step_id != prediction.step_id:
        return 0.0

    return sum((_compare_locations(answer.locations, prediction.locations),
                _compare_participants(answer.destroyed, prediction.destroyed),
                _compare_participants(answer.created, prediction.created))) / 3


def _score_moves(answers: List[Move], predictions: List[Move]) -> Metric:
    m = _edgecases(answers, predictions)
    if m:
        return m

    return _score(answers, predictions, _score_move_pair)


def _score_move_pair(answer: Move, prediction: Move) -> float:
    if answer.step_id != prediction.step_id:
        return 0.0

    return sum((_compare_participants(answer.participants, prediction.participants),
                _compare_locations(answer.location_before, prediction.location_before),
                _compare_locations(answer.location_after, prediction.location_after))) / 3


def _compare_participants(answer: str, prediction: str) -> float:
    # Trivial match
    if answer == prediction:
        return 1.0

    prediction_terms = terms.extract_termsets(prediction)
    answer_terms = terms.extract_termsets(answer)

    # calculate Jaccard similarity score
    numerator = terms.terms_overlap(prediction_terms, answer_terms)
    denominator = len(prediction_terms) + len(answer_terms) - numerator

    return numerator / denominator


def _compare_locations(answer: str, prediction: str) -> float:
    if answer == prediction:
        return 1.0

    prediction_terms = terms.extract_termsets_with_normalization(prediction)
    answer_terms = terms.extract_termsets_with_normalization(answer)

    # calculate Jaccard similarity score
    numerator = terms.terms_overlap(prediction_terms, answer_terms)
    denominator = len(prediction_terms) + len(answer_terms) - numerator

    return numerator / denominator


# Score a pair of QType answers and predictions, such that:
#
#   precision = precision_numerator / len(predictions)
#   recall = recall_numerator / len(answers)
#
# The calculation of precision and recall numerators depends on the number of answers and predictions. In these
# examples, a1 and a2 are answers and p1, p2 and p3 are predictions. Combinations (like a2p3) indicate a score for the
# answer-prediction pair (like a2 and p3).
#
# Example 1: answers = [a1,a2] predictions = [p1]
#   precision_numerator = max(a1p1,  a2p1)
#   recall_numerator    = max(a1p1) + max(a2p1)
#
# Example 2: answers = [a1,a2] predictions = [p1,p2]
#   precision_numerator = max(a1p1, a2p1) + max(a1p2, a2p2)
#   recall_numerator    = max(a1p1, a2p1) + max(a1p2, a2p2)
#
# Example 3: answers = [a1,a2] predictions = [p1,p2,p3]
#   precision_numerator = max(a1p1, a2p1) + max(a1p2, a2p2) + max(a1p3, a2p3)
#   recall_numerator    = max(a1p1, a1p2, a1p3) + max(a2p1, a2p2, a2p3)
def _score(answers: List[QType], predictions: List[QType], scoring_function: Callable[[QType, QType], float]) -> Metric:
    precision_numerator = 0.0
    for p in predictions:
        max_score = 0.0
        for a in answers:
            max_score = max(max_score, scoring_function(a, p))
        precision_numerator += max_score

    # only compute recall numerator when number of predictions doesn't match number of expected answers
    recall_numerator = precision_numerator
    if len(predictions) != len(answers):
        recall_numerator = 0.0
        for a in answers:
            max_score = 0.0
            for p in predictions:
                max_score = max(max_score, scoring_function(a, p))
            recall_numerator += max_score

    if precision_numerator == 0.0:
        precision = 0.0
    else:
        precision = precision_numerator / (1.0 * len(predictions))

    if recall_numerator == 0.0:
        recall = 0.0
    else:
        recall = recall_numerator / (1.0 * len(answers))

    return Metric(precision=precision, recall=recall)
