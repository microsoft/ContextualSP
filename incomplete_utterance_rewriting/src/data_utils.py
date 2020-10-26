# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Author: Qian Liu (SivilTaram)
# Original Repo: https://github.com/microsoft/ContextualSP

from typing import List
from typing import Tuple

import nltk
from allennlp.training.metrics.metric import Metric
from nltk.translate.bleu_score import corpus_bleu
from overrides import overrides
from rouge import Rouge
from simplediff import diff


class SpecialSymbol:
    context_internal = '[SEP]'
    end_placeholder = '[END]'


@Metric.register('batch_average')
class BatchAverage(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, values: List):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        for value in values:
            self._total_value += value
            self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = (self._total_value / self._count if self._count > 0
                         else 0)
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0


@Metric.register('f_score')
class FScoreMetric(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self, prefix) -> None:
        self._total_inter_count = 0.0
        self._total_pred_count = 0.0
        self._total_ref_count = 0.0
        self._prefix = prefix

    @overrides
    def __call__(self, inter_list: List,
                 pred_list: List,
                 ref_list: List):
        for inter_count, pred_count, ref_count in zip(inter_list, pred_list, ref_list):
            self._total_inter_count += inter_count
            self._total_pred_count += pred_count
            self._total_ref_count += ref_count

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        precision = (self._total_inter_count / self._total_pred_count
                     if self._total_pred_count > 0 else 0)
        recall = (self._total_inter_count / self._total_ref_count
                  if self._total_ref_count > 0 else 0)
        fscore = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        if reset:
            self.reset()
        return {
            '_P' + self._prefix: precision,
            '_R' + self._prefix: recall,
            'F' + self._prefix: fscore
        }

    @overrides
    def reset(self):
        self._total_ref_count = 0.0
        self._total_pred_count = 0.0
        self._total_inter_count = 0.0


@Metric.register('corpus_bleu')
class CorpusBLEUMetric(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self) -> None:
        self._total_reference = []
        self._total_prediction = []

    @overrides
    def __call__(self, reference: List[str], prediction: List[str]):
        ref_list = [[ref.split(' ')] for ref in reference]
        pred_list = [pred.split(' ') for pred in prediction]
        self._total_reference.extend(ref_list)
        self._total_prediction.extend(pred_list)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        if len(self._total_prediction) > 0:
            bleu1s = corpus_bleu(self._total_reference, self._total_prediction, weights=(1.0, 0.0, 0.0, 0.0))
            bleu2s = corpus_bleu(self._total_reference, self._total_prediction, weights=(0.5, 0.5, 0.0, 0.0))
            bleu3s = corpus_bleu(self._total_reference, self._total_prediction, weights=(0.33, 0.33, 0.33, 0.0))
            bleu4s = corpus_bleu(self._total_reference, self._total_prediction, weights=(0.25, 0.25, 0.25, 0.25))
        else:
            bleu1s = 0
            bleu2s = 0
            bleu3s = 0
            bleu4s = 0

        if reset:
            self.reset()

        return {
            '_BLEU1': bleu1s,
            '_BLEU2': bleu2s,
            '_BLEU3': bleu3s,
            'BLEU4': bleu4s
        }

    @overrides
    def reset(self):
        self._total_reference = []
        self._total_prediction = []


class Scorer(object):

    @staticmethod
    def em_score(references, candidates):
        matches = []
        for ref, cand in zip(references, candidates):
            if ref == cand:
                matches.append(1)
            else:
                matches.append(0)
        return matches

    @staticmethod
    def rouge_score(references, candidates):
        """
        https://github.com/pltrdy/rouge
        :param references: list string
        :param candidates: list string
        :return:
        """
        rouge = Rouge()
        rouge1s = []
        rouge2s = []
        rougels = []
        for ref, cand in zip(references, candidates):
            if cand.strip() == '':
                cand = 'hello'
            rouge_score = rouge.get_scores(cand, ref)
            rouge_1 = rouge_score[0]['rouge-1']['f']
            rouge_2 = rouge_score[0]['rouge-2']['f']
            rouge_l = rouge_score[0]['rouge-l']['f']
            rouge1s.append(rouge_1)
            rouge2s.append(rouge_2)
            rougels.append(rouge_l)
        return rouge1s, rouge2s, rougels

    @staticmethod
    def restored_count(references, predictions, currents):

        def score_function(ref_n_gram, pred_n_gram, ref_restore, pred_restore):
            ref_restore = set(ref_restore)
            pred_restore = set(pred_restore)
            ref_n_gram = set([ngram_phrase for ngram_phrase in ref_n_gram if
                              set(ngram_phrase) & ref_restore])
            pred_n_gram = set([ngram_phrase for ngram_phrase in pred_n_gram if
                               set(ngram_phrase) & pred_restore])
            inter_count = len(ref_n_gram & pred_n_gram)
            pred_count = len(pred_n_gram)
            ref_count = len(ref_n_gram)
            return inter_count, pred_count, ref_count

        inter_count_1 = []
        pred_count_1 = []
        ref_count_1 = []

        inter_count_2 = []
        pred_count_2 = []
        ref_count_2 = []

        inter_count_3 = []
        pred_count_3 = []
        ref_count_3 = []

        for ref, cand, cur in zip(references, predictions, currents):
            ref_tokens = ref.split(' ')
            pred_tokens = cand.split(' ')
            cur_tokens = cur.split(' ')
            ref_restore_tokens = [token for token in ref_tokens if token not in
                                  cur_tokens]
            pred_restore_tokens = [token for token in pred_tokens if token not in
                                   cur_tokens]
            if len(ref_restore_tokens) == 0:
                continue
            ref_ngram_1 = list(nltk.ngrams(ref_tokens, n=1))
            pred_ngram_1 = list(nltk.ngrams(pred_tokens, n=1))
            inter_1, pred_1, ref_1 = score_function(ref_ngram_1, pred_ngram_1, ref_restore_tokens, pred_restore_tokens)

            ref_ngram_2 = list(nltk.ngrams(ref_tokens, n=2))
            pred_ngram_2 = list(nltk.ngrams(pred_tokens, n=2))
            inter_2, pred_2, ref_2 = score_function(ref_ngram_2, pred_ngram_2, ref_restore_tokens, pred_restore_tokens)

            ref_ngram_3 = list(nltk.ngrams(ref_tokens, n=3))
            pred_ngram_3 = list(nltk.ngrams(pred_tokens, n=3))
            inter_3, pred_3, ref_3 = score_function(ref_ngram_3, pred_ngram_3, ref_restore_tokens, pred_restore_tokens)

            inter_count_1.append(inter_1)
            pred_count_1.append(pred_1)
            ref_count_1.append(ref_1)
            inter_count_2.append(inter_2)
            pred_count_2.append(pred_2)
            ref_count_2.append(ref_2)
            inter_count_3.append(inter_3)
            pred_count_3.append(pred_3)
            ref_count_3.append(ref_3)

        return (inter_count_1, pred_count_1, ref_count_1,
                inter_count_2, pred_count_2, ref_count_2,
                inter_count_3, pred_count_3, ref_count_3)


def export_word_edit_matrix(context: List,
                            current_sen: List,
                            label_sen: List,
                            super_mode: str = 'before',
                            # if there requires multiple insert, we only
                            # keep the longest one
                            only_one_insert: bool = False):
    if isinstance(context, str):
        context_seq = list(context)
        current_seq = list(current_sen)
        label_seq = list(label_sen)
    else:
        context_seq = context
        current_seq = current_sen
        label_seq = label_sen
    applied_changes = diff(current_seq, label_seq)

    def sub_finder(cus_list, pattern, used_pos):
        find_indices = []
        for i in range(len(cus_list)):
            if cus_list[i] == pattern[0] and \
                    cus_list[i:i + len(pattern)] == pattern \
                    and i not in used_pos:
                find_indices.append((i, i + len(pattern)))
        if len(find_indices) == 0:
            return 0, 0
        else:
            return find_indices[-1]

    def cont_sub_finder(cus_list, pattern, used_pos):
        context_len = len(cus_list)
        pattern_len = len(pattern)
        for i in range(context_len):
            k = i
            j = 0
            temp_indices = []
            while j < pattern_len and k < context_len:
                if cus_list[k] == pattern[j][0] and \
                        cus_list[k:k + len(pattern[j])] == pattern[j] \
                        and k not in used_pos:
                    temp_indices.append((k, k + len(pattern[j])))
                    j += 1
                else:
                    k += 1
            if j == pattern_len:
                return zip(*temp_indices)
        else:
            return 0, 0

    rm_range = None
    ret_ops = []
    context_used_pos = []
    current_used_pos = []
    pointer = 0
    for diff_sample in applied_changes:
        diff_op = diff_sample[0]
        diff_content = diff_sample[1]
        if diff_op == '-':
            if rm_range is not None:
                ret_ops.append(['remove', rm_range, []])
            start, end = sub_finder(current_seq, diff_content, current_used_pos
                                    )
            rm_range = [start, end]
            current_used_pos.extend(list(range(start, end)))
        elif diff_op == '+':
            start, end = sub_finder(context_seq, diff_content, context_used_pos)
            # cannot find the exact match substring, we should identify the snippets
            if start == 0 and end == 0:
                inner_diff = diff(diff_content, context_seq)
                overlap_content = [inner_diff_sample[1] for
                                   inner_diff_sample in inner_diff if inner_diff_sample[0] == '=']
                if len(overlap_content) > 0:
                    # only take one insert
                    if len(overlap_content) == 1 or only_one_insert:
                        overlap_content = sorted(overlap_content, key=lambda x: len(x), reverse=True)[0]
                        start, end = sub_finder(context_seq, overlap_content,
                                                context_used_pos)
                    else:
                        start_end_tuple = cont_sub_finder(context_seq, overlap_content, context_used_pos)
                        # start is a list, end is also
                        start, end = start_end_tuple
                else:
                    start, end = 0, 0
            if not (start == 0 and end == 0):
                if isinstance(start, int):
                    add_ranges = [[start, end]]
                else:
                    add_ranges = list(zip(start, end))

                if rm_range is not None:
                    for add_range in add_ranges:
                        context_used_pos.extend(list(range(add_range[0], add_range[1])))
                        ret_ops.append(['replace', rm_range, add_range])
                    rm_range = None
                else:
                    for add_range in add_ranges:
                        if super_mode in ['before', 'both']:
                            ret_ops.append(['before', [pointer, pointer], add_range])
                        if super_mode in ['after', 'both']:
                            if pointer >= 1:
                                ret_ops.append(['after', [pointer - 1, pointer - 1], add_range])
        elif diff_op == '=':
            if rm_range is not None:
                ret_ops.append(['remove', rm_range, []])
            start, end = sub_finder(current_seq, diff_content, current_used_pos
                                    )
            current_used_pos.extend(list(range(start, end)))
            rm_range = None
            pointer = end
    return ret_ops


def transmit_seq(cur_str: str, context_str: str,
                 op_seq: List[Tuple[str, Tuple, Tuple]]) -> str:
    """
    Given an operation sequence as `add/replace`, context_start_end, cur_start_end, transmit the generated sequence
    :param op_seq:
    :return:
    """
    current_seq = cur_str.split(' ')
    context_seq = context_str.split(' ')

    for operation in op_seq:
        opera_op = operation[0]
        current_range = operation[1]
        context_range = operation[2]
        if opera_op == 'replace':
            current_seq[current_range[0]:current_range[1]] = context_seq[context_range[0]:context_range[1]]
        elif opera_op == 'before':
            current_seq[current_range[0]:current_range[0]] = context_seq[context_range[0]:context_range[1]]
        elif opera_op == 'after':
            current_seq[current_range[0] + 1: current_range[0] + 1] = context_seq[context_range[0]:context_range[1]]

    # remove current_seq
    ret_str = ' '.join(current_seq).strip()

    return ret_str


def get_class_mapping(super_mode: str):
    """
    Mapping mode into integer
    :param super_mode: before, after & both
    :return:
    """
    class_mapping = ['none', 'replace']
    if super_mode == 'both':
        class_mapping.extend(['before', 'after'])
    else:
        class_mapping.append(super_mode)
    return {k: v for v, k in enumerate(class_mapping)}
