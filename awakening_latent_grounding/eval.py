import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from collections import Counter
from models import *
from utils import *


def evaluate_squall(model: nn.Module, data_iter: DataLoader, enable_types: List[SQLTokenType], threshold: float):
    eval_results, eval_logs = {}, []
    for eval_type in enable_types:
        eval_results[eval_type] = defaultdict(int)

    statistics = Counter()
    with torch.no_grad():
        for model_input in data_iter:
            # model_input.pop('column_labels', None)
            model_output = model(**model_input)
            example = model_input['example'][0]
            gold_sql: SQLExpression = SQLExpression.from_json(example['sql'])
            meta_index: MetaIndex = model_input['meta_index'][0]
            question: Utterance = Utterance.from_json(example['question'])
            schema: WTQSchema = WTQSchema.from_json(example['schema'])

            eval_logs.append(
                "table id: {}, query id: {} question: {}\n".format(schema.table_id, example['id'], question.text))
            eval_logs.append("Schema: {}\n".format(schema.to_string()))
            eval_logs.append("Gold SQL: {}\n".format(str(gold_sql)))

            statistics['total_examples'] += 1
            pred_sql = None
            if 'hypotheses' in model_output:
                hypothesis: SQLTokenHypothesis = model_output['hypotheses'][0]
                pred_sql = hypothesis.to_sql(schema.table_id)

            sql_correct = pred_sql is not None and pred_sql.sql == gold_sql.sql
            eval_logs.append("Pred SQL: {}\n".format(str(pred_sql)))
            statistics['LF_correct'] += sql_correct

            gold_align_labels = get_wtq_alignments_from_labeling(example['align_labels'], question, schema)
            pred_align_labels = get_wtq_alignments_from_prediction(model_output['alignment_weights'][0], question,
                                                                   schema, meta_index, threshold=threshold)
            eval_logs.append(
                'Gold Align: {}\n'.format(" ".join([str(align_label) for align_label in gold_align_labels])))
            eval_logs.append(
                'Pred Align: {}\n'.format(" ".join([str(align_label) for align_label in pred_align_labels])))

            linking_result = evaluate_linking(gold_align_labels, pred_align_labels, enable_types)
            metrics = get_precision_recall_and_f1(linking_result)

            eval_logs.append("SQL(LF) = {}\n".format(sql_correct))
            for eval_type, eval_value in linking_result.items():
                eval_results[eval_type]['tp'] += eval_value['tp']
                eval_results[eval_type]['fn'] += eval_value['fn']
                eval_results[eval_type]['fp'] += eval_value['fp']
                eval_logs.append(
                    "{}: P = {:.3f}, R = {:.3f}, F1 = {:.3f}\n".format(str(eval_type), metrics[eval_type]['P'],
                                                                       metrics[eval_type]['R'],
                                                                       metrics[eval_type]['F1']))

            eval_logs.append('\n')

    LF_acc = statistics['LF_correct'] / statistics['total_examples']
    print("SQL LF accuracy: {:.3f} ({}/{})".format(LF_acc, statistics['LF_correct'], statistics['total_examples']))

    align_metrics = get_precision_recall_and_f1(eval_results)
    eval_result_string = ''
    out_eval_results = {'LF Accuracy': LF_acc}
    for eval_type in enable_types:
        print("{}: P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(str(eval_type), align_metrics[eval_type]['P'],
                                                               align_metrics[eval_type]['R'],
                                                               align_metrics[eval_type]['F1']))
        eval_result_string += "{}_F1_{:.3f}".format(str(eval_type), align_metrics[eval_type]['F1'])
        out_eval_results[str(eval_type)] = eval_results[eval_type]
        out_eval_results[str(eval_type)]['P'] = align_metrics[eval_type]['P']
        out_eval_results[str(eval_type)]['R'] = align_metrics[eval_type]['R']
        out_eval_results[str(eval_type)]['F1'] = align_metrics[eval_type]['F1']

    saved_path = args.checkpoint.replace('.pt', "eval_LF_acc_{:.3f}_align_thres{:.2f}_{}.txt".format(LF_acc, threshold,
                                                                                                     eval_result_string))
    open(saved_path, 'w', encoding='utf-8').writelines(eval_logs)
    json.dump(out_eval_results, open(args.checkpoint.replace('.pt', "eval_thres{}_results.json".format(threshold)), 'w',
                                     encoding='utf-8'))
    print("Evaluate over")


def postprocess_with_gold_sql(identify_logits: Dict[SQLTokenType, torch.Tensor],
                              alignments: Dict[SQLTokenType, torch.Tensor], gold_sql: SQLExpression,
                              schema: SpiderSchema, values: List[ValueMatch]):
    for tbl_idx in range(schema.num_tables):
        is_found = False
        for sql_token in gold_sql.tokens:
            if sql_token.token_type == SQLTokenType.table and schema.id_map[sql_token.value] == tbl_idx:
                is_found = True
                break
        if not is_found:
            identify_logits[SQLTokenType.table][tbl_idx][1] = -1000
            alignments[SQLTokenType.table][tbl_idx] = 0.0

    for col_idx in range(schema.num_columns):
        is_found = False
        for sql_token in gold_sql.tokens:
            if sql_token.token_type == SQLTokenType.column and schema.id_map[sql_token.value] == col_idx:
                is_found = True
                break
        if not is_found:
            identify_logits[SQLTokenType.column][col_idx][1] = -1000
            alignments[SQLTokenType.column][col_idx] = 0.0

    for val_idx in range(len(values)):
        is_found = False
        for sql_token in gold_sql.tokens:
            if sql_token.token_type == SQLTokenType.column and sql_token.column_name == values[val_idx].column.lower():
                is_found = True
                break
        if not is_found:
            identify_logits[SQLTokenType.value][val_idx][1] = -1000
            alignments[SQLTokenType.value][val_idx] = 0.0

    return identify_logits, alignments


def evaluate_spider(model: nn.Module, data_iter: DataLoader, enable_types: List[SQLTokenType], args):
    prefix = os.path.split(args.data_path)[-1]
    assert prefix in ['train', 'dev']
    print('Data Prefix: {}'.format(prefix))

    eval_results, eval_logs = {}, []
    for eval_type in enable_types:
        eval_results[eval_type] = defaultdict(int)

    statistics = Counter()
    slsql_align_labels, slsql_align_labels_all = [], []
    with torch.no_grad():
        for model_input in data_iter:
            model_output = model(**model_input)
            example = model_input['example'][0]
            meta_index: MetaIndex = model_input['meta_index'][0]
            question: Utterance = Utterance.from_json(example['question'])
            schema: SpiderSchema = SpiderSchema.from_json(example['schema'])
            values = [ValueMatch.from_json(v) for v in example['values']]
            sql: SQLExpression = SQLExpression.from_json(example['sql'])

            eval_logs.append("Q: {}\n".format(question.text))
            eval_logs.append("{}\n".format(schema.to_string()))
            eval_logs.append("Gold SQL: {}\n".format(sql.sql))

            statistics['total_examples'] += 1
            gold_align_labels = get_spider_alignments_from_labeling(example['align_labels'], question, schema)

            identify_logits = {SQLTokenType.table: model_output['table_logits'][0],
                               SQLTokenType.column: model_output['column_logits'][0],
                               SQLTokenType.value: model_output['value_logits'][0]}
            tbl_align_weights, col_align_weights, val_align_weights = meta_index.split(
                model_output['alignment_weights'][0].cpu())
            align_weights = {SQLTokenType.table: tbl_align_weights, SQLTokenType.column: col_align_weights,
                             SQLTokenType.value: val_align_weights}

            if args.with_gold_sql:
                identify_logits, align_weights = postprocess_with_gold_sql(identify_logits, align_weights, sql, schema,
                                                                           values)

            pred_align_labels = greedy_link_spider(identify_logits, align_weights, question, schema, values,
                                                   threshold=args.threshold)
            # pred_align_labels = generate_alignments_spider(align_weights, question, schema, values, threshold=args.threshold)

            assert len(pred_align_labels) == len(question.tokens)
            sql_align_label = [label.to_slsql(schema) for label in pred_align_labels]
            slsql_align_labels += [sql_align_label]
            # slsql_align_labels_all += [greedy_search_all_spider(identify_logits, align_weights, question, schema, values, threshold=args.threshold)]

            # pred_align_labels = post_process_alignment_labels(pred_align_labels, gold_align_labels)
            eval_logs.append(
                'Gold Align: {}\n'.format(" ".join([str(align_label) for align_label in gold_align_labels])))
            eval_logs.append(
                'Pred Align: {}\n'.format(" ".join([str(align_label) for align_label in pred_align_labels])))

            linking_result = evaluate_linking(gold_align_labels, pred_align_labels, enable_types)
            metrics = get_precision_recall_and_f1(linking_result)

            for eval_type, eval_value in linking_result.items():
                eval_results[eval_type]['tp'] += eval_value['tp']
                eval_results[eval_type]['fn'] += eval_value['fn']
                eval_results[eval_type]['fp'] += eval_value['fp']
                eval_logs.append(
                    "{}: P = {:.3f}, R = {:.3f}, F1 = {:.3f}\n".format(str(eval_type), metrics[eval_type]['P'],
                                                                       metrics[eval_type]['R'],
                                                                       metrics[eval_type]['F1']))

            eval_logs.append('\n')

    align_metrics = get_precision_recall_and_f1(eval_results)
    eval_result_string = []
    out_eval_results = {}
    for eval_type in enable_types:
        print("{}: P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(str(eval_type), align_metrics[eval_type]['P'],
                                                               align_metrics[eval_type]['R'],
                                                               align_metrics[eval_type]['F1']))
        eval_result_string += ["{}_{:.3f}".format(eval_type.abbr, align_metrics[eval_type]['F1'])]
        out_eval_results[str(eval_type)] = eval_results[eval_type]
        out_eval_results[str(eval_type)]['P'] = align_metrics[eval_type]['P']
        out_eval_results[str(eval_type)]['R'] = align_metrics[eval_type]['R']
        out_eval_results[str(eval_type)]['F1'] = align_metrics[eval_type]['F1']

    eval_saved_path = args.checkpoint.replace('.pt', "{}.threshold{:.2f}.{}.results.txt".format(prefix, args.threshold,
                                                                                                ".".join(
                                                                                                    eval_result_string)))
    align_saved_path = args.checkpoint.replace('.pt', ".{}_align.json".format(prefix))
    align_all_saved_path = args.checkpoint.replace('.pt', ".{}_align.all.json".format(prefix))

    if args.with_gold_sql:
        eval_saved_path = eval_saved_path.replace(".results.txt", ".gold_sql.results.txt")
        align_saved_path = align_saved_path.replace(".json", '.gold_sql.json')
        align_all_saved_path = align_all_saved_path.replace(".all.json", '.gold_sql.all.json')

    open(eval_saved_path, 'w', encoding='utf-8').writelines(eval_logs)
    save_json_objects(slsql_align_labels, align_saved_path)
    save_json_objects(slsql_align_labels_all, align_all_saved_path)
    print("Evaluate over")


def evaluate(args):
    config, model, data_iter = load_model_and_data_iter(args)
    if config['model'] in ['WTQAlignmentModel', 'WTQSeq2SeqModel']:
        evaluate_squall(model, data_iter, [SQLTokenType.column], args.threshold)
    elif config['model'] in ['SpiderAlignmentModel']:
        evaluate_spider(model, data_iter, [SQLTokenType.table, SQLTokenType.column, SQLTokenType.value], args)
    else:
        raise NotImplementedError()


def load_model_and_data_iter(args):
    ckpt_path = args.checkpoint
    device = torch.device(args.device)
    config = json.load(open(os.path.join(os.path.dirname(ckpt_path), 'config.json'), 'r', encoding='utf-8'))
    config['checkpoint'] = ckpt_path
    config['device'] = device

    model = load_model_from_checkpoint(**config)
    model.eval()
    print('-------------------Config-------------------')
    for key, val in config.items():
        print(key, val)
    print('load {} from {} over .'.format(config['model'], ckpt_path))

    bert_version = config['bert_version']
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    data_path = args.data_path + '.{}.json'.format(bert_version)
    data_iter = get_data_iterator_func(config['model'])(data_path, tokenizer, 1, device, False, False, 512, None)
    return config, model, data_iter


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint')
    parser.add_argument('-data', '--data_path')
    parser.add_argument('-dump_all', '--dump_all', action='store_true')
    parser.add_argument('-gpu', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-with_gold_sql', '--with_gold_sql', action='store_true')
    parser.add_argument('-threshold', '--threshold', type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
