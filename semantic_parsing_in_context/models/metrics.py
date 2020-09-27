# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from functools import partial
from typing import List
from typing import Optional, Dict
from typing import Tuple
import edit_distance
import numpy as np
import torch
from allennlp.training.metrics.metric import Metric
from overrides import overrides
from context.converter import ActionConverter
from context.db_context import SparcDBContext
from scripts.sparc_evaluate import evaluate


class MetricUtil:
    """
    This metric is deigned for full interaction matching ratio
    """

    def __init__(self, dataset_path=None):
        """
        Dataset path if provided for sql evaluation
        :param dataset_path:
        """
        self._total_value = 0.0
        self._count = 0
        # we package a complete evaluate function in `sparc_evaluate.py`
        if dataset_path is not None:
            self._evaluator = SQLEvaluator(dataset_path=dataset_path)
        else:
            self._evaluator = ActionEvaluator()

    def __call__(self, best_action_indices: List[List[int]],
                 gold_labels,
                 batch_size: int,
                 mask: Optional[torch.LongTensor],
                 db_contexts: List[SparcDBContext] = None,
                 action_mapping: List[List[str]] = None,
                 with_sql: bool = False) -> Tuple:
        # if with sql, we need the action mapping to restore the action mapping
        assert (action_mapping is not None) == with_sql
        assert (db_contexts is not None) == with_sql
        assert isinstance(gold_labels[0], str) == with_sql

        # convert best_final_states into best_action_indices
        return self.calculation(best_action_indices, gold_labels, batch_size, mask,
                                db_contexts, action_mapping, with_sql)

    def calculation(self, best_action_indices: List[List[int]],
                    gold_labels,
                    batch_size: int,
                    mask: Optional[torch.LongTensor],
                    db_contexts: List[SparcDBContext] = None,
                    action_mapping: List[List[str]] = None,
                    with_sql: bool = False,
                    soft_correct: bool = False):
        """
            This method is designed for check the correctness of metric measuring
        :param best_action_indices: predicted action sequence
        :param gold_labels: if ground-truth is SQL, the type should be `str`; otherwise, it should be torch.LongTensor.
        :param batch_size: batch size, for separate interaction accuracy calculation
        :param mask: action mask which has shape batch_size * inter_size, max_action_len
        :param db_contexts: db_context for mapping action str sequence into SQL
        :param action_mapping: int -> str, mapping action into corresponding string
        :param with_sql: whether evaluation under sql equality
        :param soft_correct: soft correct will return similarity(float) rather than correctness(integer)
        :return:
        """
        assert (action_mapping is not None) == with_sql
        assert (db_contexts is not None) == with_sql
        assert isinstance(gold_labels[0], str) == with_sql

        sen_mask = mask.sum(dim=1).ne(0)
        iter_size = len(gold_labels)
        assert iter_size % batch_size == 0
        inter_size = iter_size // batch_size
        # correct matrix
        if soft_correct:
            correct_mat = np.zeros((batch_size, inter_size), dtype=np.float)
        else:
            correct_mat = np.zeros((batch_size, inter_size), dtype=np.long)
        # iteration over all instances
        for i in range(iter_size):
            # if equal to 0, skip it
            if sen_mask[i] == 0:
                continue
            # for plain calculation
            if with_sql:
                # map predicted action into sql
                action_seq_ind = best_action_indices[i]
                action_seq_str = [action_mapping[i][ind] for ind in action_seq_ind]
                if len(action_seq_ind) == 0:
                    match_score = 0.0
                else:
                    converter = ActionConverter(db_context=db_contexts[i])
                    try:
                        action_sql = converter.translate_to_sql(action_seq_str)
                    except Exception as e:
                        logging.error("Converter error: {}".format(e))
                        action_sql = f'NULL'
                    match_score = self._evaluator.is_equal(action_sql, gold_labels[i], db_contexts[i].db_id)
            else:
                if soft_correct:
                    match_score = self._evaluator.similarity(best_action_indices[i], gold_labels[i], mask[i])
                else:
                    match_score = self._evaluator.is_equal(best_action_indices[i], gold_labels[i], mask[i])
            correct_mat[i // inter_size, i % inter_size] = match_score

        sen_mask = sen_mask.view(batch_size, inter_size).cpu().data.numpy()
        return correct_mat, sen_mask


class Evaluator:
    """
    Abstract class for evaluator
    """
    def __init__(self):
        pass

    def is_equal(self, predict, target, option) -> int:
        """
        Judge whether `predict` is equal to `target`
        :return: if equal return 1; otherwise, return 0.
        """
        pass

    def similarity(self, predict, target, option) -> float:
        """
        Calculate similarity between predict and target,
        :return: the similarity
        """
        pass


class SQLEvaluator(Evaluator):

    def __init__(self, dataset_path):
        super().__init__()
        self.evaluate_func = partial(evaluate,
                                     db_dir=os.path.join(dataset_path, 'database'),
                                     table=os.path.join(dataset_path, 'tables.json'))

    @overrides
    def is_equal(self, predict_sql, gold_sql, db_id) -> int:
        """
        Judge whether given predict sql is equal to ground truth under the db_id
        :return: if equal, return 1; otherwise, return 0
        """
        try:
            exact_match_score = self.evaluate_func(gold_sql, predict_sql, db_id)
        except Exception as e:
            logging.error("SQL Parse error: {}".format(e))
            logging.error("DB_id: {}, Gold_SQL: {}, Predicted SQL: {}".format(db_id, gold_sql, predict_sql))
            exact_match_score = 0
        return exact_match_score

    @overrides
    def similarity(self, predict_sql, gold_sql, db_id) -> float:
        raise NotImplementedError


class ActionEvaluator(Evaluator):

    @overrides
    def is_equal(self, predicted: List[int], targets: torch.LongTensor, target_mask: torch.LongTensor) -> int:
        """
        Judge whether given predict sql is equal to ground truth under the db_id
        :return: if equal, return 1; otherwise, return 0
        """
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        # remove padding ones
        actual_len = target_mask.sum()
        targets_trimmed = targets[:actual_len]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        is_correct = torch.equal(predicted_tensor, targets_trimmed)
        if is_correct:
            return 1
        else:
            return 0

    @overrides
    def similarity(self, predicted: List[int], targets: torch.LongTensor, target_mask: torch.LongTensor) -> float:
        # remove padding ones
        actual_len = target_mask.sum()
        targets_trimmed = targets[:actual_len]
        targets_trimmed = list(targets_trimmed.cpu().data.numpy())
        sm = edit_distance.SequenceMatcher(a=predicted, b=targets_trimmed)
        # get the edit distance similarity between two lists
        return sm.ratio()


@Metric.register("turn_average")
class TurnAverage(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self, prefix) -> None:
        self._total_seq_value = 0.0
        self._total_inter_value = 0.0
        self._total_turn_1_value = 0.0
        self._total_turn_2_value = 0.0
        self._total_turn_3_value = 0.0
        self._total_turn_4_value = 0.0

        self._seq_count = 0
        self._inter_count = 0
        self._turn_1_count = 0
        self._turn_2_count = 0
        self._turn_3_count = 0
        self._turn_4_count = 0
        # for record the metric
        self.prefix = prefix

    @overrides
    def __call__(self, correct_mat, mask_mat):
        """
        Parameters
        ----------
        correct_mat : ``np.matrix``
            has shape batch_size x inter_size, 1 means correct while 0 means error.
        mask_mat: ``np.matrix``
            has the same shape with correct mat, 0 means invalid and 1 means valid.
        """
        # return the sequence correct number
        correct_mat = correct_mat & mask_mat
        self._total_seq_value += np.count_nonzero(correct_mat == 1)
        self._seq_count += np.count_nonzero(mask_mat == 1)

        batch_size, maximum_inter_size = correct_mat.shape
        for i in range(maximum_inter_size):
            statistic_score = np.count_nonzero(correct_mat[:, i] == 1)
            statistic_count = np.count_nonzero(mask_mat[:, i] == 1)

            if i == 0:
                self._total_turn_1_value += statistic_score
                self._turn_1_count += statistic_count
            elif i == 1:
                self._total_turn_2_value += statistic_score
                self._turn_2_count += statistic_count
            elif i == 2:
                self._total_turn_3_value += statistic_score
                self._turn_3_count += statistic_count
            else:
                self._total_turn_4_value += statistic_score
                self._turn_4_count += statistic_count

        # only valid & incorrect, return 1
        not_correct_mat = np.logical_and(np.logical_not(correct_mat), mask_mat)
        # if anyone is 1, the result is 0
        correct_inter = 1 - not_correct_mat.max(axis=1)
        mask_inter = mask_mat.sum(axis=1) != 0
        correct_inter = correct_inter & mask_inter
        self._total_inter_value += np.count_nonzero(correct_inter == 1)
        self._inter_count += np.count_nonzero(mask_inter == 1)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_seq = self._total_seq_value / self._seq_count if self._seq_count > 0 else 0
        average_inter = self._total_inter_value / self._inter_count if self._inter_count > 0 else 0
        average_turn_1 = self._total_turn_1_value / self._turn_1_count if self._turn_1_count > 0 else 0
        average_turn_2 = self._total_turn_2_value / self._turn_2_count if self._turn_2_count > 0 else 0
        average_turn_3 = self._total_turn_3_value / self._turn_3_count if self._turn_3_count > 0 else 0
        average_turn_4 = self._total_turn_4_value / self._turn_4_count if self._turn_4_count > 0 else 0

        if reset:
            self.reset()

        return {
            f'{self.prefix}_exact_match': average_seq,
            # hidden them in TQDM report
            f'_{self.prefix}_inter_exact_match': average_inter,
            f'_{self.prefix}_turn_1_exact_match': average_turn_1,
            f'_{self.prefix}_turn_2_exact_match': average_turn_2,
            f'_{self.prefix}_turn_3_exact_match': average_turn_3,
            f'_{self.prefix}_turn_4_exact_match': average_turn_4,
        }

    @overrides
    def reset(self):
        self._total_inter_value = 0.0
        self._total_seq_value = 0.0
        self._total_turn_1_value = 0.0
        self._total_turn_2_value = 0.0
        self._total_turn_3_value = 0.0
        self._total_turn_4_value = 0.0

        self._inter_count = 0
        self._seq_count = 0
        self._turn_1_count = 0
        self._turn_2_count = 0
        self._turn_3_count = 0
        self._turn_4_count = 0
