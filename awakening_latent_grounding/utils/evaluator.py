from utils.nlp_utils import ValueMatch, is_adjective
import torch
import nltk
from collections import OrderedDict, defaultdict
from typing import Any, List, Dict, Tuple
from dataclasses import dataclass
from utils.data_types import *
from utils.data_iter import MetaIndex
from utils.schema_linker import *
from fuzzywuzzy import fuzz
import os


def reduce_alignment_matrix(alignment_matrix: torch.Tensor, mappings: List[int], max_length: int) -> torch.Tensor:
    new_alignment_matrix = torch.zeros((alignment_matrix.size(0), max_length), device=alignment_matrix.device)
    for i in range(alignment_matrix.size(0)):
        for j in range(alignment_matrix.size(1)):
            new_alignment_matrix[i][mappings[j]] += alignment_matrix[i][j]
    return new_alignment_matrix


def reduce_alignment_matrix_question_first(alignment_matrix: torch.Tensor, mappings: List[int],
                                           max_length: int) -> torch.Tensor:
    assert len(alignment_matrix) >= max_length
    new_alignment_matrix = torch.zeros((max_length, alignment_matrix.size(1)), device=alignment_matrix.device)
    for i in range(alignment_matrix.size(0)):
        for j in range(alignment_matrix.size(1)):
            new_alignment_matrix[mappings[i]][j] += alignment_matrix[i][j] / mappings.count(mappings[i])
    return new_alignment_matrix


def evaluate_linking(gold_align_labels: List[AlignmentLabel], pred_align_labels: List[AlignmentLabel],
                     enable_eval_types: List[SQLTokenType]):
    assert len(gold_align_labels) == len(pred_align_labels)
    eval_result = {}
    for eval_type in enable_eval_types:
        eval_result[eval_type] = defaultdict(int)
        for gold_label, pred_label in zip(gold_align_labels, pred_align_labels):
            if gold_label.align_type == eval_type:
                if gold_label == pred_label:
                    eval_result[eval_type]['tp'] += 1
                else:
                    eval_result[eval_type]['fn'] += 1

            if pred_label.align_type == eval_type:
                if gold_label != pred_label:
                    eval_result[eval_type]['fp'] += 1

    return eval_result


def get_precision_recall_and_f1(eval_result: Dict) -> Dict:
    metrics = {}
    for eval_type in eval_result:
        precision = eval_result[eval_type]['tp'] / (eval_result[eval_type]['tp'] + eval_result[eval_type]['fp']) if (
                                                                                                                                eval_result[
                                                                                                                                    eval_type][
                                                                                                                                    'tp'] +
                                                                                                                                eval_result[
                                                                                                                                    eval_type][
                                                                                                                                    'fp']) > 0 else 0
        recall = eval_result[eval_type]['tp'] / (eval_result[eval_type]['tp'] + eval_result[eval_type]['fn']) if (
                                                                                                                             eval_result[
                                                                                                                                 eval_type][
                                                                                                                                 'tp'] +
                                                                                                                             eval_result[
                                                                                                                                 eval_type][
                                                                                                                                 'fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        metrics[eval_type] = {'P': precision, 'R': recall, 'F1': f1}
    return metrics


def get_spider_alignments_from_labeling(nl_alignments: List[Dict], question: Utterance, schema: SpiderSchema) -> List[
    AlignmentLabel]:
    assert len(question.tokens) == len(nl_alignments)
    align_labels = []
    for q_idx in range(len(nl_alignments)):
        align_obj = nl_alignments[q_idx]
        if align_obj is None:
            align_labels.append(AlignmentLabel(question.tokens[q_idx], SQLTokenType.null, None, 1.0))
        elif align_obj['type'] == 'tbl':
            align_labels.append(AlignmentLabel(token=question.tokens[q_idx], align_type=SQLTokenType.table,
                                               align_value=schema.table_names_original[align_obj['id']].lower(),
                                               confidence=1.0))
        elif align_obj['type'] == 'col':
            col_full_name = schema.get_column_full_name(align_obj['id'])
            align_labels.append(
                AlignmentLabel(token=question.tokens[q_idx], align_type=SQLTokenType.column, align_value=col_full_name,
                               confidence=1.0))
        elif align_obj['type'] == 'val':
            col_full_name = schema.get_column_full_name(align_obj['id'])
            align_labels.append(AlignmentLabel(token=question.tokens[q_idx], align_type=SQLTokenType.value,
                                               align_value="VAL_{}".format(col_full_name), confidence=1.0))
        else:
            raise NotImplementedError()

    return align_labels


def get_spider_alignments_from_prediction(alignment_weights: torch.Tensor, question: Utterance, schema: SpiderSchema,
                                          values: List[ValueMatch], meta_index: MetaIndex, threshold: float = 0.1) -> \
List[AlignmentLabel]:
    alignment_weights[meta_index.num_tables] = 0.0  # Set column * to 0
    alignment_weights = alignment_weights.transpose(0, 1)
    assert len(alignment_weights) == len(question.tokens)
    align_labels = []
    for q_idx in range(len(alignment_weights)):
        max_idx = torch.argmax(alignment_weights[q_idx], dim=-1).item()
        confidence = alignment_weights[q_idx, max_idx]
        if confidence < threshold:
            align_label = AlignmentLabel(question.tokens[q_idx], SQLTokenType.null, None, 1 - confidence)
            align_labels.append(align_label)
            continue

        if max_idx < meta_index.num_tables:
            align_labels.append(
                AlignmentLabel(question.tokens[q_idx], SQLTokenType.table, schema.table_names_original[max_idx].lower(),
                               confidence))
        elif max_idx < meta_index.num_tables + meta_index.num_columns:
            align_labels.append(AlignmentLabel(question.tokens[q_idx], SQLTokenType.column,
                                               schema.get_column_full_name(max_idx - meta_index.num_tables),
                                               confidence))
        elif max_idx < meta_index.num_tables + meta_index.num_columns + meta_index.num_values:
            value_idx = max_idx - meta_index.num_tables - meta_index.num_columns
            align_labels.append(
                AlignmentLabel(question.tokens[q_idx], SQLTokenType.value, 'VAL_{}'.format(values[value_idx].column),
                               confidence))
        else:
            raise NotImplementedError()

    return align_labels


def post_process_alignment_labels(pred_align_labels: List[AlignmentLabel], gold_align_labels: List[AlignmentLabel]):
    new_pred_align_labels = []
    for i, (pred_align_label, gold_align_label) in enumerate(zip(pred_align_labels, gold_align_labels)):
        if gold_align_label.align_type == SQLTokenType.value and pred_align_label.align_type in [SQLTokenType.column,
                                                                                                 SQLTokenType.table]:
            new_label = AlignmentLabel(pred_align_label.token, align_type=SQLTokenType.null, align_value=None,
                                       confidence=pred_align_label.confidence)
            new_pred_align_labels += [new_label]
            continue

        if gold_align_label.align_type == SQLTokenType.null and pred_align_label.token.token.lower() in ['with', 'any',
                                                                                                         'without']:
            new_label = AlignmentLabel(pred_align_label.token, align_type=SQLTokenType.null, align_value=None,
                                       confidence=pred_align_label.confidence)
            new_pred_align_labels += [new_label]
            continue

        if gold_align_label.align_type == SQLTokenType.null and pred_align_label.align_type == SQLTokenType.column and is_adjective(
                pred_align_label.token.token.lower()):
            new_label = AlignmentLabel(pred_align_label.token, align_type=SQLTokenType.null, align_value=None,
                                       confidence=pred_align_label.confidence)
            new_pred_align_labels += [new_label]
            continue

        new_pred_align_labels.append(pred_align_label)

    return new_pred_align_labels


def get_wtq_alignments_from_labeling(nl_alignments: List[Dict], question: Utterance, schema: WTQSchema) -> List[AlignmentLabel]:
    assert len(question.tokens) == len(nl_alignments)
    align_labels = []
    for q_idx in range(len(question.tokens)):
        align_obj = nl_alignments[q_idx]
        if align_obj[0] == 'None':
            align_labels += [AlignmentLabel(question.tokens[q_idx], SQLTokenType.null, None, 1.0)]
        elif align_obj[0] == 'Keyword':
            align_labels += [AlignmentLabel(question.tokens[q_idx], SQLTokenType.keyword, align_obj[1][0], 1.0)]
        elif align_obj[0] == 'Column':
            col_id = schema.internal_name_to_id[align_obj[1]]
            align_labels += [AlignmentLabel(question.tokens[q_idx], SQLTokenType.column,
                                            schema.column_headers[schema.internal_to_header[col_id]], 1.0)]
        elif align_obj[0] == 'Literal':
            align_labels += [AlignmentLabel(question.tokens[q_idx], SQLTokenType.value, None, 1.0)]
        else:
            raise NotImplementedError("not supported alignment type: {}".format(align_obj))

    return align_labels


def get_wtq_alignments_from_prediction(alignment_weights: torch.Tensor, question: Utterance, schema: WTQSchema,
                                       meta_index: MetaIndex, threshold: float = 0.1, question_first=False) -> List[AlignmentLabel]:
    if question_first:
        alignment_weights = reduce_alignment_matrix_question_first(alignment_weights, question.get_piece2token(),
                                                                   len(question.tokens))
    else:
        alignment_weights = reduce_alignment_matrix(alignment_weights, question.get_piece2token(),
                                                    len(question.tokens)).transpose(0, 1)
    assert len(alignment_weights) == len(question.tokens)
    align_labels = []
    align_span = []
    for q_idx in range(len(alignment_weights)):
        max_idx = torch.argmax(alignment_weights[q_idx], dim=-1)
        confidence = alignment_weights[q_idx, max_idx]
        if confidence < threshold:
            align_label = AlignmentLabel(question.tokens[q_idx], SQLTokenType.null, None, 1 - confidence)
            align_labels.append(align_label)
            continue

        assert max_idx < meta_index.num_columns
        col_idx = meta_index.lookup_entity_id('col', int(max_idx))
        align_label = AlignmentLabel(question.tokens[q_idx], SQLTokenType.column, schema.column_headers[col_idx],
                                     confidence)
        align_labels.append(align_label)
    i = 0
    while i < len(align_labels):
        if align_labels[i].align_type == SQLTokenType.column:
            j = i + 1
            while j < len(align_labels):
                if align_labels[j].align_value == align_labels[i].align_value and \
                        align_labels[j].align_type == SQLTokenType.column:
                    j += 1
                else:
                    break
            align_span.append((i, j))
            i = j
            continue
        else:
            i += 1

    table_id = schema.table_id

    # load table from the same directory
    data_dir = os.getenv("PT_DATA_DIR", default="data/squall")
    table_content = json.load(open("{}/json/{}.json".format(data_dir, table_id), 'r'))
    all_values = []
    for internal_columns in table_content['contents']:
        for column in internal_columns:
            all_values += column['data']
    for span in align_span:
        find = 0
        for _label in align_labels[span[0]: span[1]]:
            if _label.align_value != align_labels[span[0]].align_value:
                find = 1
                break
        if find == 1:
            continue
        span_strs = [(span[0], span[1], " ".join([x.token.token for x in align_labels[span[0]: span[1]]]))]
        if span[0] - 1 >= 0:
            span_strs.append(
                (span[0] - 1, span[1], " ".join([x.token.token for x in align_labels[span[0] - 1: span[1]]])))
        if span[0] - 2 >= 0:
            span_strs.append(
                (span[0] - 2, span[1], " ".join([x.token.token for x in align_labels[span[0] - 2: span[1]]])))
        if span[1] + 1 < len(question.tokens):
            span_strs.append(
                (span[0], span[1] + 1, " ".join([x.token.token for x in align_labels[span[0]: span[1] + 1]])))
        if span[1] + 2 < len(question.tokens):
            span_strs.append(
                (span[0], span[1] + 2, " ".join([x.token.token for x in align_labels[span[0]: span[1] + 2]])))

        for value in all_values:
            if not isinstance(value, str):
                continue
            for sp1, sp2, span_str in span_strs:
                if value is not None and \
                        (fuzz.ratio(value.lower(), span_str.lower()) > 80):
                    if value.lower() in table_content['headers']:
                        continue
                    for label in align_labels[span[0]:span[1]]:
                        label.align_type = SQLTokenType.null
                    break
        for sp1, sp2, span_str in span_strs:
            if span_str in table_content['headers'][2:]:
                for idx in range(sp1, sp2):
                    align_labels[idx] = AlignmentLabel(question.tokens[idx], SQLTokenType.column, span_str, 1)

    return align_labels



@dataclass
class SpiderCase:
    schema: SpiderSchema
    question: Utterance
    goal_sql: SQLExpression
    enc_input_tokens: List[str]
    correct_dict: Dict[str, bool]

    identification_dict: Dict[str, Any]
    alignment_dict: Dict[str, torch.Tensor]
    gold_alignments: List[AlignmentLabel]
    pred_alignments: List[AlignmentLabel]
    metrics: Dict[str, Dict[str, float]]

    values: List[ValueMatch]

    def to_string(self):
        out_strs = []
        tokens = [token.token for token in self.question.tokens]
        out_strs.append(
            "Q: {}, Table = {}, Column = {}, Value = {}, All = {}".format(self.question.text, self.correct_dict['tbl'],
                                                                          self.correct_dict['col'],
                                                                          self.correct_dict['val'],
                                                                          self.correct_dict['all']))
        out_strs.append("Input tokens: {}".format(" ".join(self.enc_input_tokens)))
        out_strs.append(self.schema.to_string('\n'))

        out_strs.append("Gold SQL: {}".format(self.goal_sql.sql))

        if 'tbl' in self.identification_dict:
            for i, (tbl_id, gold_label, pred_label, pred_score) in enumerate(self.identification_dict['tbl']):
                tbl_name = self.schema.table_names_original[tbl_id]
                if gold_label == 0 and pred_label == 0:
                    continue
                out_strs.append(
                    "T {}: gold = {}, pred = {} / {:.3f}, Correct = {}".format(tbl_name, gold_label, pred_label,
                                                                               pred_score, gold_label == pred_label))
                if 'tbl' in self.alignment_dict:
                    align_vector = self.alignment_dict['tbl'][i]
                    assert len(align_vector) == len(tokens)
                    align_strs = ["{}/{:.3f}".format(token, weight.item()) for token, weight in
                                  zip(tokens, align_vector)]
                    out_strs += ["Alignment: {}".format(" ".join(align_strs))]

        if 'col' in self.identification_dict:
            for i, (col_id, gold_label, pred_label, pred_score) in enumerate(self.identification_dict['col']):
                if gold_label == 0 and pred_label == 0:
                    continue
                col_name = self.schema.get_column_full_name(col_id)
                out_strs.append(
                    "C {}: gold = {}, pred = {} / {:.3f}, Correct = {}".format(col_name, gold_label, pred_label,
                                                                               pred_score, gold_label == pred_label))
                if 'col' in self.alignment_dict:
                    align_vector = self.alignment_dict['col'][i]
                    assert len(align_vector) == len(tokens)
                    align_strs = ["{}/{:.3f}".format(token, weight.item()) for token, weight in
                                  zip(tokens, align_vector)]
                    out_strs += ["Alignment: {}".format(" ".join(align_strs))]

        if 'val' in self.identification_dict:
            for i, (val_id, gold_label, pred_label, pred_score) in enumerate(self.identification_dict['val']):
                if gold_label == 0 and pred_label == 0:
                    continue
                col_name = self.values[val_id].column
                val_str = "{}[{}:{}]".format(self.values[i].value, self.values[i].start, self.values[i].end)
                out_strs.append(
                    "V {}_{}: gold = {}, pred = {} / {:.3f}, Correct = {}".format(col_name, val_str, gold_label,
                                                                                  pred_label, pred_score,
                                                                                  gold_label == pred_label))
                if 'val' in self.alignment_dict:
                    align_vector = self.alignment_dict['val'][i]
                    assert len(align_vector) == len(tokens)
                    align_strs = ["{}/{:.3f}".format(token, weight.item()) for token, weight in
                                  zip(tokens, align_vector)]
                    out_strs += ["Alignment: {}".format(" ".join(align_strs))]

        out_strs.append('Gold Align: {}'.format(" ".join([str(align_label) for align_label in self.gold_alignments])))
        out_strs.append('Pred Align: {}'.format(" ".join([str(align_label) for align_label in self.pred_alignments])))
        for align_type in [SQLTokenType.table, SQLTokenType.column]:
            out_strs.append(
                "{} P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(str(align_type), self.metrics[align_type]['P'],
                                                                self.metrics[align_type]['R'],
                                                                self.metrics[align_type]['F1']))
        return '\n'.join(out_strs)


class SpiderEvaluator:
    def __init__(self) -> None:
        self.statistics = defaultdict(int)
        self.cases: List[SpiderCase] = []
        self.align_results = {SQLTokenType.table: defaultdict(int), SQLTokenType.column: defaultdict(int),
                              SQLTokenType.value: defaultdict(int)}

    def add_batch(self, inputs, outputs):
        batch_size = len(inputs['input_token_ids'])
        self.statistics['total_count'] += batch_size
        self.statistics['total_loss'] += outputs['loss'].item() * batch_size

        for i in range(batch_size):
            example = inputs['example'][i]
            question: Utterance = Utterance.from_json(example['question'])
            meta_index: MetaIndex = inputs['meta_index'][i]
            schema: SpiderSchema = SpiderSchema.from_json(example['schema'])
            gold_sql: SQLExpression = SQLExpression.from_json(example['sql'])
            values: List[ValueMatch] = [ValueMatch.from_json(x) for x in example['values']]

            gold_tbl_labels = inputs['table_labels'][i]
            pred_tbl_labels = torch.argmax(outputs['table_logits'][i], dim=-1)
            pred_tbl_scores = torch.softmax(outputs['table_logits'][i], dim=-1)
            tbl_correct = pred_tbl_labels.equal(gold_tbl_labels)
            tbl_identify_results = []
            for j in range(len(gold_tbl_labels)):
                tbl_identify_results += [(j, gold_tbl_labels[j].item(), pred_tbl_labels[j].item(),
                                          pred_tbl_scores[j, pred_tbl_labels[j]].item())]

            gold_col_labels = inputs['column_labels'][i]
            pred_col_labels = torch.argmax(outputs['column_logits'][i], dim=-1)
            pred_col_scores = torch.softmax(outputs['column_logits'][i], dim=-1)
            col_correct = pred_col_labels.equal(gold_col_labels)
            col_identify_results = []
            for j in range(len(gold_col_labels)):
                col_identify_results += [(j, gold_col_labels[j].item(), pred_col_labels[j].item(),
                                          pred_col_scores[j, pred_col_labels[j]].item())]

            val_correct = True
            gold_val_labels = inputs['value_labels'][i]
            val_identify_results = []
            if len(gold_val_labels) > 0:
                pred_val_labels = torch.argmax(outputs['value_logits'][i], dim=-1)
                pred_val_scores = torch.softmax(outputs['value_logits'][i], dim=-1)
                val_correct = pred_val_labels.equal(gold_val_labels)
                for j in range(len(gold_val_labels)):
                    val_identify_results += [(j, gold_val_labels[j].item(), pred_val_labels[j].item(),
                                              pred_val_scores[j, pred_val_labels[j]].item())]

            align_weights = {}
            if 'alignment_weights' in outputs:
                tbl_align_weights, col_align_weights, val_align_weights = meta_index.split(
                    outputs['alignment_weights'][i], dim=0)
                align_weights['tbl'] = tbl_align_weights
                align_weights['col'] = col_align_weights
                align_weights['val'] = val_align_weights

            gold_align_labels = get_spider_alignments_from_labeling(example['align_labels'], question, schema)
            # pred_align_labels = get_spider_alignments_from_prediction(outputs['alignment_weights'][i], question, schema, values, meta_index, threshold=0.15)
            # pred_align_labels = post_process_alignment_labels(pred_align_labels, gold_align_labels)

            identify_logits = {SQLTokenType.table: outputs['table_logits'][i],
                               SQLTokenType.column: outputs['column_logits'][i],
                               SQLTokenType.value: outputs['value_logits'][i]}
            align_weights2 = {SQLTokenType.table: align_weights['tbl'], SQLTokenType.column: align_weights['col'],
                              SQLTokenType.value: align_weights['val']}
            pred_align_labels = greedy_link_spider(identify_logits, align_weights2, question, schema, values,
                                                   threshold=0.3)

            align_result = evaluate_linking(gold_align_labels, pred_align_labels,
                                            [SQLTokenType.table, SQLTokenType.column, SQLTokenType.value])
            metrics = get_precision_recall_and_f1(align_result)
            for align_type in align_result:
                for key in align_result[align_type]:
                    self.align_results[align_type][key] += align_result[align_type][key]

            eval_case = SpiderCase(
                schema=schema,
                question=question,
                goal_sql=gold_sql,
                enc_input_tokens=inputs['input_tokens'][i],
                correct_dict={'tbl': tbl_correct, 'col': col_correct, 'val': val_correct,
                              'all': tbl_correct & col_correct},
                identification_dict={'tbl': tbl_identify_results, 'col': col_identify_results,
                                     'val': val_identify_results},
                alignment_dict=align_weights,
                gold_alignments=gold_align_labels,
                pred_alignments=pred_align_labels,
                metrics=metrics,
                values=values)

            self.cases += [eval_case]
            self.statistics['tbl_correct'] += tbl_correct
            self.statistics['col_correct'] += col_correct
            self.statistics['val_correct'] += val_correct
            self.statistics['overall_correct'] += tbl_correct & col_correct  # & val_correct

    def get_metrics(self, saved_file: str = None):
        metrics = OrderedDict()
        total_count = self.statistics['total_count']
        metrics['avg loss'] = self.statistics['total_loss'] / total_count
        metrics['table accuracy'] = self.statistics['tbl_correct'] / total_count
        metrics['column accuracy'] = self.statistics['col_correct'] / total_count
        metrics['value accuracy'] = self.statistics['val_correct'] / total_count
        metrics['overall accuracy'] = self.statistics['overall_correct'] / total_count

        align_metrics = get_precision_recall_and_f1(self.align_results)
        align_f1_string = ["acc_{:.3f}".format(metrics['overall accuracy'])]
        total_f1 = 0
        for align_type in self.align_results:
            total_f1 += align_metrics[align_type]['F1']
            metrics[str(align_type)] = " P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(align_metrics[align_type]['P'],
                                                                                     align_metrics[align_type]['R'],
                                                                                     align_metrics[align_type]['F1'])
            align_f1_string += ["{}_{:.3f}".format(align_type.abbr, align_metrics[align_type]['F1'])]
        align_f1_string = ".".join(align_f1_string)

        metrics['average F1'] = total_f1 / len(self.align_results)

        if saved_file is not None:
            with open(saved_file.replace(".txt", ".{}.txt".format(align_f1_string)), 'w', encoding='utf-8') as fw:
                for case in self.cases:
                    fw.write(case.to_string() + '\n\n')
        return metrics


@dataclass
class WTQCase:
    schema: WTQSchema
    gold_sql: SQLExpression
    pred_sql: SQLExpression
    question: Utterance
    enc_input_tokens: List[str]
    correct_dict: Dict[str, bool]

    identification_dict: Dict[str, Any]
    alignment_dict: Dict[str, torch.Tensor]
    gold_alignments: List[AlignmentLabel]
    pred_alignments: List[AlignmentLabel]
    metrics: Dict[str, Dict[str, float]]

    def to_string(self):
        out_strs = []
        tokens = self.question.get_piece2token()
        sql_correct = self.pred_sql is not None and self.pred_sql == self.gold_sql
        out_strs.append("Q: {}, Column = {}, All = {}, SQL = {}".format(self.question.text, self.correct_dict['col'],
                                                                        self.correct_dict['all'], sql_correct))
        out_strs.append("Gold SQL: {}".format(self.gold_sql.sql))
        if self.pred_sql is not None:
            out_strs.append("Pred SQL: {}".format(self.pred_sql.sql))

        out_strs.append("Encode Tokens: {}".format(" ".join(self.enc_input_tokens)))
        out_strs.append(self.schema.to_string())

        for i, (col_id, gold_label, pred_label, pred_score) in enumerate(self.identification_dict['col']):
            if gold_label == 0 and pred_label == 0:
                continue
            out_strs.append(
                "{}: gold = {}, pred = {} / {:.3f}, Correct = {}".format(col_id, gold_label, pred_label, pred_score,
                                                                         gold_label == pred_label))
            # if 'col' in self.alignment_dict:
            #     align_vector = self.alignment_dict['col'][i]
            #     assert len(align_vector) == len(tokens), print(align_vector.shape, tokens)
            #     align_strs = ["{}/{:.3f}".format(token, weight.item()) for token, weight in zip(tokens, align_vector)]
            #     out_strs += ["Alignment: {}".format(" ".join(align_strs))]

        out_strs.append('Gold Align: {}'.format(" ".join([str(align_label) for align_label in self.gold_alignments])))
        out_strs.append('Pred Align: {}'.format(" ".join([str(align_label) for align_label in self.pred_alignments])))
        for align_type in [SQLTokenType.column]:
            out_strs.append(
                "{} P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(str(align_type), self.metrics[align_type]['P'],
                                                                self.metrics[align_type]['R'],
                                                                self.metrics[align_type]['F1']))

        return '\n'.join(out_strs)


class WTQEvaluator:
    def __init__(self) -> None:
        self.statistics = defaultdict(int)
        self.cases: List[WTQCase] = []
        self.align_results = {SQLTokenType.column: defaultdict(int)}

    def add_batch(self, inputs, outputs):
        batch_size = len(inputs['input_token_ids'])
        self.statistics['total_count'] += batch_size
        self.statistics['total_loss'] += outputs['loss'].item() * batch_size

        for i in range(batch_size):
            example = inputs['example'][i]
            gold_sql: SQLExpression = SQLExpression.from_json(example['sql'])
            question: Utterance = Utterance.from_json(example['question'])
            meta_index: MetaIndex = inputs['meta_index'][i]
            schema: WTQSchema = WTQSchema.from_json(example['schema'])

            gold_col_labels = inputs['column_labels'][i]
            pred_col_labels = torch.argmax(outputs['column_logits'][i], dim=-1)
            pred_col_scores = torch.softmax(outputs['column_logits'][i], dim=-1)
            col_correct = pred_col_labels.equal(gold_col_labels)

            col_identify_results = []
            for j in range(len(gold_col_labels)):
                col_name = schema.column_headers[j]
                col_identify_results += [(col_name, gold_col_labels[j].item(), pred_col_labels[j].item(),
                                          pred_col_scores[j, pred_col_labels[j]].item())]

            align_weights = {}
            if 'alignment_weights' in outputs:
                col_align_weights = outputs['alignment_weights'][i]
                align_weights['col'] = col_align_weights

            gold_align_labels = get_wtq_alignments_from_labeling(example['align_labels'], question, schema)
            pred_align_labels = get_wtq_alignments_from_prediction(outputs['alignment_weights'][i], question, schema,
                                                                   meta_index, threshold=0.15)
            align_result = evaluate_linking(gold_align_labels, pred_align_labels, [SQLTokenType.column])

            metrics = get_precision_recall_and_f1(align_result)
            for align_type in align_result:
                for key in align_result[align_type]:
                    self.align_results[align_type][key] += align_result[align_type][key]

            pred_sql: SQLExpression = None
            if 'hypotheses' in outputs:
                hypothesis: SQLTokenHypothesis = outputs['hypotheses'][i]
                pred_sql = hypothesis.to_sql()

            sql_correct = pred_sql is not None and pred_sql == gold_sql

            eval_case = WTQCase(
                schema=schema,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                question=Utterance.from_json(example['question']),
                enc_input_tokens=inputs['input_tokens'][i],
                correct_dict={'col': col_correct, 'all': col_correct},
                identification_dict={'col': col_identify_results},
                alignment_dict=align_weights,
                gold_alignments=gold_align_labels,
                pred_alignments=pred_align_labels,
                metrics=metrics)

            self.cases += [eval_case]
            self.statistics['col_correct'] += col_correct
            self.statistics['overall_correct'] += col_correct
            self.statistics['LF_correct'] += sql_correct

    def get_metrics(self, saved_file: str = None):
        metrics = OrderedDict()
        total_count = self.statistics['total_count']
        metrics['Average loss'] = self.statistics['total_loss'] / total_count
        metrics['Column accuracy'] = self.statistics['col_correct'] / total_count
        metrics['overall accuracy'] = self.statistics['overall_correct'] / total_count
        metrics['SQL accuracy'] = self.statistics['LF_correct'] / total_count
        align_metrics = get_precision_recall_and_f1(self.align_results)
        for align_type in self.align_results:
            metrics[str(align_type)] = " P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(align_metrics[align_type]['P'],
                                                                                     align_metrics[align_type]['R'],
                                                                                     align_metrics[align_type]['F1'])

        if saved_file is not None:
            with open(saved_file, 'w', encoding='utf-8') as fw:
                fw.write('{}\n\n'.format("\n".join(
                    [f"{k} = {v:.4f}" if isinstance(v, float) else "{} = {}".format(k, str(v)) for k, v in
                     metrics.items()])))
                for case in self.cases:
                    fw.write(case.to_string() + '\n\n')
        return metrics
