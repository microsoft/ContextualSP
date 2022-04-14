import argparse
import json
import re
import subprocess
from collections import defaultdict
from re import RegexFlag
import networkx as nx
import torch

from genre.fairseq_model import GENRE, mGENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.trie import Trie
from semparse.sql.spider import load_original_schemas, load_tables
from semparse.worlds.evaluate_spider import evaluate as evaluate_sql
from step1_schema_linking import read_database_schema

database_dir='./data/spider/database'
database_schema_filename = './data/spider/tables.json'
schema_tokens, column_names, database_schemas = read_database_schema(database_schema_filename)

with open(f'./data/spider/dev.json', 'r', encoding='utf-8') as f:
    item = json.load(f)
sql_to_db = []
for i in item:
    sql_to_db.append(i['db_id'])


def post_processing_sql(p_sql, foreign_key_maps, schemas, o_schemas):
    foreign_key = {}
    for k, v in foreign_key_maps.items():
        if k == v:
            continue
        key = ' '.join(sorted([k.split('.')[0].strip('_'), v.split('.')[0].strip('_')]))
        foreign_key[key] = (k.strip('_').replace('.', '@'), v.strip('_').replace('.', '@'))

    primary_key = {}
    for t in o_schemas.tables:
        table = t.orig_name.lower()
        if len(t.primary_keys) == 0:
            continue
        column = t.primary_keys[0].orig_name.lower()
        primary_key[table] = f'{table}@{column}'

    p_sql = re.sub(r'(=)(\S+)', r'\1 \2', p_sql)
    p_sql = p_sql.split()

    columns = ['*']
    tables = []
    for table, column_list in schemas.schema.items():
        for column in column_list:
            columns.append(f"{table}@{column}")
        tables.append(table)

    # infer table from mentioned column
    all_from_table_ids = set()
    from_idx = where_idx = group_idx = order_idx = -1
    for idx, token in enumerate(p_sql):
        if '@' in token and token in columns:
            all_from_table_ids.add(schemas.idMap[token.split('@')[0]])
        if token == 'from' and from_idx == -1:
            from_idx = idx
        if token == 'where' and where_idx == -1:
            where_idx = idx
        if token == 'group' and group_idx == -1:
            group_idx = idx
        if token == 'order' and order_idx == -1:
            order_idx = idx

    #don't process nested SQL (more than one select)
    if len(re.findall('select', ' '.join(p_sql))) > 1 or len(all_from_table_ids) == 0:
        return ' '.join(p_sql)

    covered_tables = set()
    candidate_table_ids = sorted(all_from_table_ids)
    start_table_id = candidate_table_ids[0]
    conds = set()
    all_conds = []

    for table_id in candidate_table_ids[1:]:
        if table_id in covered_tables:
            continue
        try:
            path = nx.shortest_path(
                o_schemas.foreign_key_graph,
                source=start_table_id,
                target=table_id,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            covered_tables.add(table_id)
            continue

        for source_table_id, target_table_id in zip(path, path[1:]):
            if target_table_id in covered_tables:
                continue
            covered_tables.add(target_table_id)
            all_from_table_ids.add(target_table_id)
            col1, col2 = o_schemas.foreign_key_graph[source_table_id][target_table_id]["columns"]
            all_conds.append((columns[col1], columns[col2]))

            conds.add((tables[source_table_id],
                       tables[target_table_id],
                       columns[col1],
                       columns[col2]))

    all_from_table_ids = list(all_from_table_ids)

    try:
        tokens = ["from", tables[all_from_table_ids[0]]]
        for i, table_id in enumerate(all_from_table_ids[1:]):
            tokens += ["join"]
            tokens += [tables[table_id]]
            tokens += ["on", all_conds[i][0], "=", all_conds[i][1]]
    except:
        return ' '.join(p_sql)

    if where_idx != -1:
        p_sql = p_sql[:from_idx] + tokens + p_sql[where_idx:]
    elif group_idx != -1:
        p_sql = p_sql[:from_idx] + tokens + p_sql[group_idx:]
    elif order_idx != -1:
        p_sql = p_sql[:from_idx] + tokens + p_sql[order_idx:]
    elif len(p_sql[:from_idx] + p_sql[from_idx:]) == len(p_sql):
        p_sql = p_sql[:from_idx] + tokens

    return ' '.join(p_sql)


def extract_structure_data(plain_text_content: str):
    def sort_by_id(data):
        data.sort(key=lambda x: int(x.split('\t')[0][2:]))
        return data

    data = []
    original_schemas = load_original_schemas(database_schema_filename)
    schemas, eval_foreign_key_maps = load_tables(database_schema_filename)

    predict_outputs = sort_by_id(re.findall("^D.+", plain_text_content, RegexFlag.MULTILINE))
    ground_outputs = sort_by_id(re.findall("^T.+", plain_text_content, RegexFlag.MULTILINE))
    source_inputs = sort_by_id(re.findall("^S.+", plain_text_content, RegexFlag.MULTILINE))

    for idx, (predict, ground, source) in enumerate(zip(predict_outputs, ground_outputs, source_inputs)):
        predict_id, predict_score, predict_clean = predict.split('\t')
        ground_id, ground_clean = ground.split('\t')
        source_id, source_clean = source.split('\t')

        db_id = sql_to_db[idx]
        #try to postprocess the incomplete sql from
        # (1) correcting the COLUMN in ON_CLAUSE based on foreign key graph
        # (2) adding the underlying TABLE via searching shortest path
        predict_clean = post_processing_sql(predict_clean, eval_foreign_key_maps[db_id], original_schemas[db_id],
                                        schemas[db_id])

        data.append((predict_id[2:], source_clean.split('<Q>')[-1].strip(), ground_clean, predict_clean, db_id))

    return data


def evaluate(data):
    def evaluate_example(_predict_str: str, _ground_str: str):
        return re.sub("\s+", "", _predict_str.lower()) == re.sub("\s+", "", _ground_str.lower())

    correct_num = 0
    correct_tag_list = []
    total = 0

    tmp = []
    for example in data:
        idx, source_str, ground_str, predict_str, db_id = example
        total += 1

        try:
            sql_match = evaluate_sql(gold=ground_str.replace('@', '.'),
                                     predict=predict_str.replace('@', '.'),
                                     db_name=db_id,
                                     db_dir=database_dir,
                                     table=database_schema_filename)
        except:
            print(predict_str)
            sql_match = False

        if (sql_match or evaluate_example(predict_str, ground_str)):
            is_correct = True
            correct_num += 1
        else:
            is_correct = False

        tmp.append(is_correct)
        correct_tag_list.append(is_correct)

    print("Correct/Total : {}/{}, {:.4f}".format(correct_num, total, correct_num / total))
    return correct_tag_list, correct_num, total


def predict_and_evaluate(model_path, dataset_path, constrain):
    if constrain:
        data = predict_with_constrain(
            model_path=model_path,
            dataset_path=dataset_path
        )
    else:
        decode_without_constrain(
            model_path=model_path,
            dataset_path=dataset_path
        )
        with open('./eval/generate-valid.txt', "r", encoding="utf8") as generate_f:
            file_content = generate_f.read()
            data = extract_structure_data(file_content)

    correct_arr, correct_num, total = evaluate(data)
    with open('./eval/spider_eval.txt', "w", encoding="utf8") as eval_file:
        for example, correct in zip(data, correct_arr):
            eval_file.write(str(correct) + "\n" + "\n".join(
                [example[0], "db: " + example[-1], example[1], "gold: " + example[2], "pred: " + example[3]]) + "\n\n")
    return correct_num, total

def get_alias_schema(schemas):
    alias_schema = {}
    for db in schemas:
        schema = schemas[db].orig
        collect = []
        for i, (t, c) in enumerate(zip(schema['column_types'], schema['column_names_original'])):
            if c[0] == -1:
                collect.append('*')
            else:
                column_with_alias = "{0}@{1}".format(schema['table_names_original'][c[0]].lower(), c[1].lower())
                collect.append(column_with_alias)
        for t in schema['table_names_original']:
            collect.append(t.lower())
        collect.append("'value'")
        alias_schema[db] = collect
    return alias_schema

def predict_with_constrain(model_path, dataset_path):
    schemas, eval_foreign_key_maps = load_tables(database_schema_filename)
    original_schemas = load_original_schemas(database_schema_filename)
    with open(f'{dataset_path}/dev.src', 'r', encoding='utf-8') as f:
        item = [i.strip() for i in f.readlines()]
    with open(f'{dataset_path}/dev.tgt', 'r', encoding='utf-8') as f:
        ground = [i.strip() for i in f.readlines()]

    alias_schema = get_alias_schema(schemas)

    item_db_cluster = defaultdict(list)
    ground_db_cluster = defaultdict(list)
    source_db_cluster = defaultdict(list)

    num_example = 1034
    for db, sentence, g_sql in zip(sql_to_db[:num_example], item[:num_example], ground[:num_example]):
        source = sentence.split('<Q>')[-1].strip()
        item_db_cluster[db].append(sentence)
        ground_db_cluster[db].append(g_sql)
        source_db_cluster[db].append(source)

    source = []
    ground = []
    for db, sentence in source_db_cluster.items():
        source.extend(sentence)
    for db, g_SQL in ground_db_cluster.items():
        ground.extend(g_SQL)

    model = GENRE.from_pretrained(model_path).eval()
    if torch.cuda.is_available():
        model.cuda()

    result=[]
    for db, sentence in item_db_cluster.items():
        print(f'processing db: {db} with {len(sentence)} sentences')
        rnt=decode_with_constrain(sentence, alias_schema[db], model)
        result.extend([i[0]['text'] if isinstance(i[0]['text'], str) else i[0]['text'][0] for i in rnt])

    eval_file_path= f'./eval/generate-valid-constrain.txt'
    with open(eval_file_path, "w", encoding="utf8") as f:
        f.write('\n'.join(result))

    # result = []
    # with open(f'./eval/generate-valid-constrain.txt', "r", encoding="utf8") as f:
    #     for idx, (sent, db_id) in enumerate(zip(f.readlines(), sql_to_db)):
    #         result.append(sent.strip())

    data = []
    for predict_id, (predict_clean, ground_clean, source_clean, db_id) in enumerate(
            zip(result, ground, source, sql_to_db)):
        predict_clean = post_processing_sql(predict_clean, eval_foreign_key_maps[db_id], original_schemas[db_id],
                                   schemas[db_id])
        data.append((str(predict_id), source_clean.split('<Q>')[-1].strip(), ground_clean, predict_clean, db_id))


    return data


def decode_with_constrain(sentences, schema, model):
    trie = Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in schema
    ])

    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
        model,
        sentences,
        mention_trie=trie,
    )

    return model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )


def decode_without_constrain( model_path, dataset_path):
    cmd = f'fairseq-generate \
        --path {model_path}/model.pt  {dataset_path}/bin \
        --gen-subset valid \
        --nbest 1 \
        --max-tokens 4096 \
        --source-lang src --target-lang tgt \
        --results-path ./eval \
        --beam 5 \
        --bpe gpt2 \
        --remove-bpe \
        --skip-invalid-size-inputs-valid-test'

    subprocess.Popen(
        cmd, universal_newlines=True, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./models/spider_sl')
    parser.add_argument("--dataset_path", default='./dataset_post/spider_sl')
    parser.add_argument("--constrain", action='store_true')
    args = parser.parse_args()

    predict_and_evaluate(model_path=args.model_path,
                         dataset_path=args.dataset_path,
                         constrain=args.constrain)
