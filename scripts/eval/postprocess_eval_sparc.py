import os
import json
import argparse
import subprocess


def postprocess(predictions, database_schema):
    correct = 0
    total = 0
    postprocess_sqls = {}

    for pred in predictions:
        db_id = pred['database_id']
        schema = database_schema[db_id]
        if db_id not in postprocess_sqls:
            postprocess_sqls[db_id] = []

        interaction_id = pred['interaction_id']
        turn_id = pred['index_in_interaction']
        total += 1

        pred_sql_str = ' '.join(pred['flat_prediction'])

        gold_sql_str = ' '.join(pred['flat_gold_queries'][0])
        if pred_sql_str == gold_sql_str:
            correct += 1

        postprocess_sql = pred_sql_str

        postprocess_sqls[db_id].append((postprocess_sql, interaction_id, turn_id))

    # print (correct, total, float(correct)/total)
    return postprocess_sqls


def read_prediction(pred_file):
    print('Read prediction from', pred_file)
    predictions = []
    with open(pred_file) as f:
        for line in f:
            pred = json.loads(line)
            predictions.append(pred)
    print('Number of predictions', len(predictions))
    return predictions


def read_schema(table_schema_path):
    with open(table_schema_path) as f:
        database_schema = json.load(f)

    database_schema_dict = {}
    for table_schema in database_schema:
        db_id = table_schema['db_id']
        database_schema_dict[db_id] = table_schema

    return database_schema_dict


def write_and_evaluate(postprocess_sqls, db_path, table_schema_path, gold_path, dataset):
    db_list = []
    with open(gold_path) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            db = line.strip().split('\t')[1]
            if db not in db_list:
                db_list.append(db)

    if dataset == 'sparc':
        cnt = 0
        output_file = 'output_temp.txt'
        with open(output_file, "w") as f:
            for db in db_list:
                for postprocess_sql, interaction_id, turn_id in postprocess_sqls[db]:
                    if turn_id == 0 and cnt > 0:
                        f.write('\n')
                    f.write('{}\n'.format(postprocess_sql))
                    cnt += 1

        command = 'python evaluation_sqa.py --db {} --table {} --etype match --gold {} --pred {}'.format(
            db_path,
            table_schema_path,
            gold_path,
            os.path.abspath(output_file))
        command += '; rm output_temp.txt'
    return command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sparc')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--pred_file', type=str, default='')
    args = parser.parse_args()

    if args.dataset == 'sparc':
        db_path = 'dataset\\database\\'
        table_schema_path = 'dataset\\tables.json'

        if args.split == 'dev':
            gold_path = 'dataset\\dev_gold.txt'

    pred_file = args.pred_file

    database_schema = read_schema(table_schema_path)
    predictions = read_prediction(pred_file)
    postprocess_sqls = postprocess(predictions, database_schema)

    command = write_and_evaluate(postprocess_sqls, db_path, table_schema_path, gold_path, args.dataset)

    # print(command)
    eval_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    with open(pred_file + '.eval', 'w') as f:
        f.write(eval_output.decode("utf-8"))
    print('Eval result in', pred_file + '.eval')
