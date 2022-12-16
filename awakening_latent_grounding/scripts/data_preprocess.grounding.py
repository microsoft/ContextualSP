# %%
import logging
import sys
sys.path.append("..")
from utils import *
from tqdm import tqdm
import argparse
from transformers import BertTokenizer
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import os


bert_version = 'bert-large-uncased-whole-word-masking'
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(bert_version)
print('load Bert tokenizer over, vocab size = {}'.format(len(tokenizer)))
statistics = defaultdict(int)
spider_type_mappings = {
    'text': 'text',
    'time': 'time',
    'number': 'number',
    'boolean': 'boolean',
    'others': 'text'
}
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_dir = os.path.join(proj_dir, 'data', 'slsql')


# load schemas from database
def get_column_names_unique(column_names: List[Tuple[int, str]], table_names: List[str], primary_keys: List[int]) -> List[str]:
    column_names_dict = defaultdict(int)
    for tbl_idx, col_name in column_names:
        column_names_dict[col_name] += 1

    column_names_unique = []
    for c_idx, (tbl_idx, col_name) in enumerate(column_names):
        if tbl_idx == -1:
            column_names_unique.append(col_name)
            continue

        if column_names_dict[col_name] == 1:
            column_names_unique.append(col_name)
        elif c_idx in primary_keys:
            column_names_unique.append(col_name)
        else:
            tbl_name = table_names[tbl_idx]
            full_name = '{} . {}'.format(tbl_name, col_name)
            column_names_unique.append(full_name)
    assert len(column_names_unique) == len(column_names)
    return column_names_unique


def alt_tbl_name(tbl_name):
    tbl_name = tbl_name.split()
    if len(tbl_name) > 1 and tbl_name[0] == 'reference':
        tbl_name = tbl_name[1:]
    if len(tbl_name) > 1 and tbl_name[-1] == 'data':
        tbl_name = tbl_name[:-1]
    if len(tbl_name) > 1 and tbl_name[-1] == 'list':
        tbl_name = tbl_name[:-1]
    return ' '.join(tbl_name)


def remove_shared_prefix(col_name: str, tbl_name: str) -> str:
    col_tokens, tbl_tokens = col_name.split(), tbl_name.split()
    idx = 0
    while idx < len(col_tokens) and idx < len(tbl_tokens) and col_tokens[idx] == tbl_tokens[idx]:
        idx += 1
    return " ".join(col_tokens[idx:])


def get_column_name_normalized(column_lem_names: List[Tuple[int, str]], table_lem_names: List[str], verbose: bool = False):
    column_norm_names, table_norm_names = [], []
    for tbl_name in table_lem_names:
        table_norm_names.append(alt_tbl_name(tbl_name))

    for col_idx, (tbl_idx, col_name) in enumerate(column_lem_names):
        if col_name == '*':
            column_norm_names.append('*')
            continue
        col_norm_name = remove_shared_prefix(
            col_name, table_norm_names[tbl_idx])
        if col_norm_name != col_name and verbose:
            logging.info(" {}\t{}\t{}".format(
                table_norm_names[tbl_idx], col_name, col_norm_name))
        column_norm_names.append(col_norm_name)

    return column_norm_names, table_norm_names


def load_schema(obj: Dict) -> SpiderSchema:
    column_names_lemma = obj['column_names_lemma']
    table_names_lemma = obj['table_names_lemma']
    column_names_original = [x[1] for x in obj['column_names_original']]
    column_to_table, table_to_columns = {}, {}
    for col_idx, (tbl_idx, _) in enumerate(obj['column_names']):
        if tbl_idx not in table_to_columns:
            table_to_columns[tbl_idx] = []
        table_to_columns[tbl_idx].append(col_idx)
        column_to_table[col_idx] = tbl_idx

    col_norm_names, tbl_norm_names = get_column_name_normalized(
        column_names_lemma, table_names_lemma, True)
    return SpiderSchema(
        db_id=obj['db_id'],
        column_names=col_norm_names,
        column_types=obj['column_types'],
        column_names_lemma=[x[1] for x in column_names_lemma],
        column_names_original=column_names_original,
        table_names=tbl_norm_names,
        table_names_lemma=table_names_lemma,
        table_names_original=obj['table_names_original'],
        table_to_columns=table_to_columns,
        column_to_table=column_to_table,
        primary_keys=obj['primary_keys'],
        foreign_keys=obj['foreign_keys'])


def load_schemas(path: str):
    databases = json.load(open(path, 'r', encoding='utf-8'))
    schemas = {}
    for database in databases:
        schema = load_schema(database)
        schemas[schema.db_id] = schema
    return schemas


def load_value_matches(path: str) -> Dict[str, ValueMatcher]:
    db_columns = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            table = json.loads(line)
            db_id = table['db_name']
            table_name = table['table_name']
            for column in table['columns']:
                column = ("{}.{}".format(table_name, column['column_name']).lower(
                ), column['data_type'], column['values'])
                db_columns[db_id].append(column)

    db_matchers = {}
    for db, columns in db_columns.items():
        db_matchers[db] = ValueMatcher(columns)
    return db_matchers


def _is_in_column(idx, field: str, tokens: List[SQLToken]):
    if field.lower() == 'where':
        if idx + 2 >= len(tokens):
            return False
        if tokens[idx + 1].token_type == SQLTokenType.keyword and tokens[idx + 1].value.lower() == 'in':
            return True
        if tokens[idx + 2].token_type == SQLTokenType.keyword and tokens[idx + 2].value.lower() == 'in':
            return True

    if field.lower() == 'select':
        if idx - 2 >= 0 and tokens[idx - 2].token_type == SQLTokenType.keyword and tokens[idx - 2].value.lower() == 'in':
            return True

        # SELECT singer.name FROM singer WHERE singer.singer_id NOT IN ( SELECT song.singer_id FROM song )
        if idx - 3 >= 0 and tokens[idx - 3].token_type == SQLTokenType.keyword and tokens[idx - 3].value.lower() == 'in':
            return True

    return False


def _is_group_by_key_column(idx, field: str, tokens: List[SQLToken], schema: SpiderSchema):
    if field.lower() == 'group':
        if isinstance(tokens[idx], ColumnToken):
            key_code = schema.get_column_key_code(
                schema.id_map[tokens[idx].column_name])
            if key_code != 0:
                return True
    return False


def generate_identify_labels_from_sql(sql: SQLExpression, schema: SpiderSchema):
    identify_labels = defaultdict(list)
    is_from = False
    field = None
    value2column_count = defaultdict(int)

    # from_tables = []
    for i, token in enumerate(sql.tokens):
        if isinstance(token, KeywordToken):
            if token.keyword.lower() in ['from']:
                is_from = True
                continue
            if token.keyword.lower() in CLAUSE_KEYWORDS:
                field = token.keyword
                is_from = False
        elif isinstance(token, ColumnToken):
            if not is_from and token.column_name != '*' and not _is_in_column(i, field, sql.tokens) and not _is_group_by_key_column(i, field, sql.tokens, schema):
                identify_labels[str(SQLTokenType.column)
                                ].append(token.column_name)
        elif isinstance(token, TableToken):
            # if not is_from:
            #     identify_labels[str(SQLTokenType.table)].append(token.table_name)
            # else:
            #     from_tables.append(token.table_name)
            identify_labels[str(SQLTokenType.table)].append(token.table_name)
        elif isinstance(token, ValueToken):
            if not is_from and field != 'LIMIT':
                if token.columns is None or len(token.columns) != 1:
                    print(sql.sql, token.value, token.columns)
                identify_labels[str(SQLTokenType.value)].append(
                    (token.value, token.columns))
        else:
            raise NotImplementedError()

    if str(SQLTokenType.table) not in identify_labels:
        identify_labels[str(SQLTokenType.table)] = []

    if str(SQLTokenType.column) not in identify_labels:
        identify_labels[str(SQLTokenType.column)] = []

    for val, columns in identify_labels[str(SQLTokenType.value)]:
        for column in columns:
            value2column_count[column] += 1

    for count in value2column_count.values():
        statistics['max_value_count'] = max(
            statistics['max_value_count'], count)

    for key in identify_labels:
        if key != str(SQLTokenType.value):
            identify_labels[key] = list(set(identify_labels[key]))
    return identify_labels


def generate_identify_labels_from_align(ant: Dict, schema: SpiderSchema):
    identify_labels = defaultdict(list)
    for tok_idx, tok_ant in enumerate(ant):
        if tok_ant is None:
            continue
        e_type = tok_ant['type']
        e_idx = tok_ant['id']
        assert e_type in ['tbl', 'col', 'val']
        if e_type == 'tbl':
            identify_labels[str(SQLTokenType.table)].append(
                schema.table_names_original[e_idx].lower())
        elif e_type == 'col':
            identify_labels[str(SQLTokenType.column)].append(
                schema.get_column_full_name(e_idx))
        elif e_type == 'val':
            identify_labels[str(SQLTokenType.value)].append(
                'val_{}'.format(schema.get_column_full_name(e_idx)))
        identify_labels[(e_type, e_idx)].append(tok_idx)

    for key in identify_labels:
        identify_labels[key] = list(set(identify_labels[key]))
    return identify_labels


def generate_masking_ngrams(question: Utterance, schema: SpiderSchema) -> List[Tuple[int, int, str]]:
    if schema.db_id not in ngram_matchers:
        column_tokens = []
        for i, column in enumerate(schema.column_names):
            column_tokens.append(
                (schema.get_column_full_name(i), column.split(' ')))
        for i, table in enumerate(schema.table_names):
            column_tokens.append(
                (schema.table_names_original[i], table.split(' ')))
        ngram_matchers[schema.db_id] = NGramMatcher(column_tokens)

    ngram_matcher = ngram_matchers[schema.db_id]
    masking_ngrams = []
    for tok_idx in range(len(question.tokens)):
        masking_ngrams.append(
            (tok_idx, tok_idx, question.tokens[tok_idx].token))

    ngram_spans = set([])
    for q_i, q_j, _, _, _ in ngram_matcher.match([token.token for token in question.tokens]):
        ngram_spans.add((q_i, q_j))

    for q_i, q_j in sorted(list(ngram_spans), key=lambda x: x[1]-x[0], reverse=True):
        is_overlap = False
        for q_i2, q_j2, ngram in masking_ngrams:
            if q_i2 <= q_i and q_j2 >= q_j:
                is_overlap = True
                break
        if not is_overlap:
            ngram_ij = " ".join([x.token for x in question.tokens[q_i:q_j+1]])
            masking_ngrams.append((q_i, q_j, ngram_ij))

    return masking_ngrams


def resolve_values(question: Utterance, schema: SpiderSchema, sql: SQLExpression):
    value_matcher = value_matchers[schema.db_id]
    value_tokens = []
    values_dict = {}
    for token in sql.tokens:
        if isinstance(token, ValueToken) and len(token.columns) > 0:
            value_tokens.append(token)
            for column in token.columns:
                values_dict[(str(token.value).strip("\"").strip(
                    '%').lower(), column.lower())] = False

    value_matches = value_matcher.match(question.text_tokens, 0.8, 3)
    for value_match in value_matches:
        if (str(value_match.value).lower(), value_match.column.lower()) in values_dict:
            values_dict[(str(value_match.value), value_match.column)] = True
            value_match.label = True

    all_resolved = True
    for (value, column), resolved in values_dict.items():
        if not resolved:
            all_resolved = False
            logging.info('Value resolved: {}/{}/{}\t{}'.format(value,
                         schema.db_id, column, question.text))
    return all_resolved, value_matches


def fix_tok(tok):
    tok = tok.lower()
    if tok == '-lrb-':
        tok = '('
    elif tok == '-rrb-':
        tok = ')'
    elif tok == '\"':
        tok = '\''
    return tok


def process_squall_query(query: Dict):
    # Step1: process question tokens
    question = query['question']
    assert len(query['toks']) == len(query['lemma'])
    question_utterance = generate_utterance(tokenizer, question, [fix_tok(
        x) for x in query['toks']], [fix_tok(x) for x in query['lemma']])

    # Step 2: process tables & columns
    assert query['db_id'] in schemas
    schema: SpiderSchema = schemas[query['db_id']]
    processed_tables = []
    for tbl_idx, col_indices in schema.table_to_columns.items():
        # special column *
        if tbl_idx == -1:
            table_json = {
                'index': -1,
                'utterance': Utterance('*', tokens=[]).to_json(),
                'columns': None
            }
            processed_tables += [table_json]
            continue
        tbl_name = schema.table_names[tbl_idx]
        table_utterance = generate_utterance(tokenizer, tbl_name)

        processed_columns = []
        for col_idx in col_indices:
            column_type = schema.column_types[col_idx]
            assert column_type in spider_type_mappings, column_type
            column_utterance = generate_utterance(
                tokenizer, schema.column_names[col_idx])
            column_json = {
                'index': col_idx,
                'utterance': column_utterance.to_json(),
                'data_type': spider_type_mappings[column_type]
            }
            processed_columns += [column_json]

        table_json = {
            'index': tbl_idx,
            'utterance': table_utterance.to_json(),
            'columns': processed_columns
        }
        processed_tables += [table_json]

    # Parse SQL
    sql = parse_spider_sql(query['query'], schema)
    sql_logs.append(question)
    sql_logs.append(query['query'])
    sql_logs.append(sql.sql + '\n')

    value_resolved, matched_values = resolve_values(
        question_utterance, schema, sql)
    if not value_resolved:
        statistics['value_unresolved'] += 1

    # Generate alignment labels for our models
    assert len(query['ant']) == len(question_utterance.tokens)
    identify_labels = generate_identify_labels_from_sql(sql, schema)
    if len(identify_labels[str(SQLTokenType.table)]) == 0:
        print(question)
        print(sql.sql)
    masking_ngrams = generate_masking_ngrams(question_utterance, schema)
    processed_query = {
        'question': question_utterance.to_json(),
        'tables': processed_tables,
        'identify_labels': identify_labels,
        'align_labels': query['ant'],
        'sql': sql.to_json(),
        'schema': schema.to_json(),
        'masking_ngrams': masking_ngrams,
        'values': [v.to_json() for v in matched_values]
    }

    return processed_query


def _compare_identify_labels(example: Dict):
    question: Utterance = Utterance.from_json(example['question'])
    identify_labels_from_sql = example['identify_labels']
    schema: SpiderSchema = SpiderSchema.from_json(example['schema'])
    sql: SQLExpression = SQLExpression.from_json(example['sql'])
    identify_labels_from_align = generate_identify_labels_from_align(
        example['align_labels'], schema)
    cmp_results = []
    cmp_results.append("Q: {}\n".format(question.text))
    cmp_results.append("SQL: {}\n".format(sql.sql))
    cmp_results.append("Table Shared: {}\n".format(' '.join(sorted(set(identify_labels_from_sql[str(
        SQLTokenType.table)]) & set(identify_labels_from_align[str(SQLTokenType.table)])))))
    cmp_results.append("Table SQL: {}\n".format(' '.join(sorted(set(identify_labels_from_sql[str(
        SQLTokenType.table)]) - set(identify_labels_from_align[str(SQLTokenType.table)])))))
    cmp_results.append("Table Align: {}\n".format(' '.join(sorted(set(identify_labels_from_align[str(
        SQLTokenType.table)]) - set(identify_labels_from_sql[str(SQLTokenType.table)])))))
    cmp_results.append("Column Shared: {}\n".format(' '.join(sorted(set(identify_labels_from_sql[str(
        SQLTokenType.column)]) & set(identify_labels_from_align[str(SQLTokenType.column)])))))
    cmp_results.append("Column SQL: {}\n".format(' '.join(sorted(set(identify_labels_from_sql[str(
        SQLTokenType.column)]) - set(identify_labels_from_align[str(SQLTokenType.column)])))))
    cmp_results.append("Column Align: {}\n".format(' '.join(sorted(set(identify_labels_from_align[str(
        SQLTokenType.column)]) - set(identify_labels_from_sql[str(SQLTokenType.column)])))))
    cmp_results.append("Table: {}\n".format(set(identify_labels_from_sql[str(
        SQLTokenType.table)]) == set(identify_labels_from_align[str(SQLTokenType.table)])))
    cmp_results.append("Column: {}\n".format(set(identify_labels_from_sql[str(
        SQLTokenType.column)]) == set(identify_labels_from_align[str(SQLTokenType.column)])))
    cmp_results.append('\n')

    identify_labels_equal['table'] += int(set(identify_labels_from_sql[str(
        SQLTokenType.table)]) == set(identify_labels_from_align[str(SQLTokenType.table)]))
    identify_labels_equal['column'] += int(set(identify_labels_from_sql[str(
        SQLTokenType.column)]) == set(identify_labels_from_align[str(SQLTokenType.column)]))
    return cmp_results


def compare_identify_labels(examples, saved_path: str):
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for example in examples:
            fw.writelines(_compare_identify_labels(example))
    print('Compare over!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f'{data_dir} does not exists. exit.')
        sys.exit(0)
    else:
        print(f'load data from {data_dir}')

    if os.path.exists(os.path.join(data_dir, 'preprocess.log')):
        os.remove(os.path.join(data_dir, 'preprocess.log'))
    logging.basicConfig(filename=os.path.join(
        data_dir, 'preprocess.log'), level=logging.DEBUG)

    schemas = load_schemas(os.path.join(data_dir, 'processed_tables.json'))
    print('load schems over, size = {}'.format(len(schemas)))
    value_matchers = load_value_matches(
        os.path.join(data_dir, 'spider_tables.txt'))

    ngram_matchers: Dict[str, NGramMatcher] = {}
    sql_logs = []

    dev_queries = json.load(
        open(os.path.join(data_dir, 'slsql_dev.json'), 'r', encoding='utf-8'))
    train_queries = json.load(
        open(os.path.join(data_dir, 'slsql_train.json'), 'r', encoding='utf-8'))
    print('load SLSQL dev & train queries over, size = {}/{}'.format(len(dev_queries), len(train_queries)))
    out = process_squall_query(dev_queries[111])
    dev_processed = []
    statistics['value_unresolved'] = 0
    for query in tqdm(dev_queries):
        dev_processed += [process_squall_query(query)]
    save_json_objects(dev_processed, os.path.join(
        data_dir, 'dev.{}.json'.format(bert_version)))
    print('process dev over, value_unresolved: {}'.format(
        statistics['value_unresolved']))

    open(os.path.join(data_dir, 'dev.parsed_sqls.log'),
         'w', encoding='utf-8').write('\n'.join(sql_logs))
    print('save parsed sqls ...')
    identify_labels_equal = defaultdict(int)

    compare_identify_labels(dev_processed, os.path.join(
        data_dir, 'dev.identify_labels.diff.txt'))
    print('Identify labesl generated from SQL accuracy: table = {:.4f} ({}/{}), column = {:.4f} ({}/{})'.format(
        identify_labels_equal['table'] / len(dev_processed),
        identify_labels_equal['table'],
        len(dev_processed),
        identify_labels_equal['column'] / len(dev_processed),
        identify_labels_equal['column'],
        len(dev_processed),
    ))
    train_processed = []
    statistics['value_unresolved'] = 0
    for query in tqdm(train_queries):
        train_processed += [process_squall_query(query)]
    save_json_objects(train_processed, os.path.join(
        data_dir, 'train.{}.json'.format(bert_version)))
    print('process train over, value_unresolved: {}'.format(
        statistics['value_unresolved']))
    dev_iter = load_spider_data_iterator(os.path.join(data_dir, 'dev.{}.json'.format(
        bert_version)), tokenizer, 16, torch.device('cpu'), False, False, 512)
    total_size, num_examples = 0, 0
    input_tokens = []
    for batch_input in dev_iter:
        bs, length = batch_input['input_token_ids'].size(
            0), batch_input['input_token_ids'].size(1)
        total_size += bs * length
        num_examples += bs
        for i in range(bs):
            input_tokens.append(
                " ".join(batch_input['input_tokens'][i]) + '\n')
    print(total_size, num_examples, total_size / num_examples)
    open(os.path.join(data_dir, 'dev.input_tokens.txt'),
         'w', encoding='utf-8').writelines(input_tokens)
    train_iter = load_spider_data_iterator(os.path.join(data_dir, 'train.{}.json'.format(
        bert_version)), tokenizer, 16, torch.device('cpu'), True, True, 512)
    total_size, num_examples = 0, 0
    for batch_input in train_iter:
        bs, length = batch_input['input_token_ids'].size(
            0), batch_input['input_token_ids'].size(1)
        total_size += bs * length
        num_examples += bs
        # print(batch_input['input_token_ids'].size())
    print(total_size, num_examples, total_size / num_examples)

    train_iter2 = load_spider_data_iterator(os.path.join(data_dir, 'train.{}.json'.format(
        bert_version)), tokenizer, 16, torch.device('cpu'), False, True, 400)
    total_size, num_examples = 0, 0
    for batch_input in train_iter2:
        bs, length = batch_input['input_token_ids'].size(
            0), batch_input['input_token_ids'].size(1)
        total_size += bs * length
        num_examples += bs
        # print(batch_input['input_token_ids'].size())
    print(total_size, num_examples, total_size / num_examples)

    dev_examples = json.load(open(os.path.join(
        data_dir, 'train.{}.json'.format(bert_version)), 'r', encoding='utf-8'))
    threshold = 0.81
    count1, count2, count3 = 0, 0, 0
    for example in dev_examples:
        values: List[ValueMatch] = [
            ValueMatch.from_json(x) for x in example['values']]
        for value in values:
            if value.score < threshold and value.score > 0.5:
                if value.label and len(value.value) <= 4:
                    print(value)
                    count1 += 1
                if len(value.value) > 4:
                    count3 += 1
                continue
            count2 += int(value.label)

    print(count1, count2, count3)
