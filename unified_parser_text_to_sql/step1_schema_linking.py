"""
Based on https://github.com/ryanzhumich/editsql/blob/master/preprocess.py
"""
import argparse
import json
import os
import re
import stanza
import sqlparse
from tqdm import tqdm

from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.sql.spider_utils import disambiguate_items, fix_number_value
from semparse.sql.spider_utils import read_dataset_schema

keyword = ['select', 'distinct', 'from', 'join', 'on', 'where', 'group', 'by', 'order', 'asc', 'desc', 'limit',
           'having',
           'and', 'not', 'or', 'like', 'between', 'in',
           'sum', 'count', 'max', 'min', 'avg',
           '(', ')', ',', '>', '<', '=', '==', '>=', '!=', '<=',
           'union', 'except', 'intersect',
           '\'value\'']


stanza.download('en')
stanza_model = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
# stanza_model=None

def write_interaction(interaction_list, split, output_dir):
    interaction = []
    for db_id in interaction_list:
        interaction += interaction_list[db_id]

    json_split = os.path.join(output_dir, split + '.json')
    with open(json_split, 'w', encoding="utf-8") as outfile:
        json.dump(interaction, outfile, indent=2, ensure_ascii=False)
    return


def read_database_schema(table_path):
    schema_tokens = {}
    column_names = {}
    database_schemas_dict = {}

    with open(table_path, 'r', encoding='UTF-8') as f:
        database_schemas = json.load(f)

    def get_schema_tokens(table_schema):
        column_names_surface_form = []
        column_names = []
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(table_name, column_name)
            else:
                # this is just *
                column_name_surface_form = column_name
            column_names_surface_form.append(column_name_surface_form.lower())
            column_names.append(column_name.lower())

        # also add table_name.*
        for table_name in table_names_original:
            column_names_surface_form.append('{}.*'.format(table_name.lower()))

        return column_names_surface_form, column_names

    for table_schema in database_schemas:
        database_id = table_schema['db_id']
        if 'column_names_original' not in table_schema:
            table_schema["column_names_original"] = table_schema["column_names"]
            table_schema["table_names_original"] = table_schema["table_names"]

        table_schema['table_names_original'] = [t.lower() for t in table_schema['table_names_original']]
        table_schema['foreign_keys_col'] = [i[0] for i in table_schema['foreign_keys']]

        structure_schema = []
        for t in table_schema['foreign_keys']:
            primary_col, foreign_col = t
            primary_col = table_schema['column_names_original'][primary_col]
            primary_col_tab = table_schema['table_names_original'][primary_col[0]].lower()
            foreign_col = table_schema['column_names_original'][foreign_col]
            foreign_col_tab = table_schema['table_names_original'][foreign_col[0]].lower()
            structure_schema.append(f"( {primary_col_tab} , {foreign_col_tab} )")
        structure_schema = list(sorted(set(structure_schema)))

        table_schema['permutations'] = [structure_schema]
        database_schemas_dict[database_id] = table_schema
        schema_tokens[database_id], column_names[database_id] = get_schema_tokens(table_schema)

        if 'column_rewrite_names' in table_schema:
            for i in range(len(table_schema['column_rewrite_names'])):
                table_schema['column_rewrite_names'][i] = [table_schema['column_names'][i][-1]] + \
                                                          table_schema['column_rewrite_names'][i][-1]
            table_schema['column_rewrite_names'] = [list(set(map(lambda x: x.lower().replace(' ', ''), i))) for i in
                                                    table_schema['column_rewrite_names']]

            for i in range(len(table_schema['table_rewrite_names'])):
                table_schema['table_rewrite_names'][i] = [table_schema['table_names'][i]] + \
                                                         table_schema['table_rewrite_names'][i]
            table_schema['table_rewrite_names'] = [list(set(map(lambda x: x.lower().replace(' ', ''), i))) for i in
                                                   table_schema['table_rewrite_names']]

    return schema_tokens, column_names, database_schemas_dict


def remove_from_with_join(format_sql_2):
    used_tables_list = []
    format_sql_3 = []
    table_to_name = {}
    table_list = []
    old_table_to_name = {}
    old_table_list = []
    for sub_sql in format_sql_2.split('\n'):
        if 'select ' in sub_sql:
            # only replace alias: t1 -> table_name, t2 -> table_name, etc...
            if len(table_list) > 0:
                for i in range(len(format_sql_3)):
                    for table, name in table_to_name.items():
                        format_sql_3[i] = format_sql_3[i].replace(table, name)

            old_table_list = table_list
            old_table_to_name = table_to_name
            table_to_name = {}
            table_list = []
            format_sql_3.append(sub_sql)
        elif sub_sql.startswith('from'):
            new_sub_sql = None
            sub_sql_tokens = sub_sql.split()
            for t_i, t in enumerate(sub_sql_tokens):
                if t == 'as':
                    table_to_name[sub_sql_tokens[t_i + 1]] = sub_sql_tokens[t_i - 1]
                    table_list.append(sub_sql_tokens[t_i - 1])
                elif t == ')' and new_sub_sql is None:
                    # new_sub_sql keeps some trailing parts after ')'
                    new_sub_sql = ' '.join(sub_sql_tokens[t_i:])
            if len(table_list) > 0:
                # if it's a from clause with join
                if new_sub_sql is not None:
                    format_sql_3.append(new_sub_sql)

                used_tables_list.append(table_list)
            else:
                # if it's a from clause without join
                table_list = old_table_list
                table_to_name = old_table_to_name
                assert 'join' not in sub_sql
                if new_sub_sql is not None:
                    sub_sub_sql = sub_sql[:-len(new_sub_sql)].strip()
                    assert len(sub_sub_sql.split()) == 2
                    used_tables_list.append([sub_sub_sql.split()[1]])
                    format_sql_3.append(sub_sub_sql)
                    format_sql_3.append(new_sub_sql)
                elif 'join' not in sub_sql:
                    assert len(sub_sql.split()) == 2 or len(sub_sql.split()) == 1
                    if len(sub_sql.split()) == 2:
                        used_tables_list.append([sub_sql.split()[1]])

                    format_sql_3.append(sub_sql)
                else:
                    print('bad from clause in remove_from_with_join')
                    exit()
        else:
            format_sql_3.append(sub_sql)

    if len(table_list) > 0:
        for i in range(len(format_sql_3)):
            for table, name in table_to_name.items():
                format_sql_3[i] = format_sql_3[i].replace(table, name)

    used_tables = []
    for t in used_tables_list:
        for tt in t:
            used_tables.append(tt)
    used_tables = list(set(used_tables))

    return format_sql_3, used_tables, used_tables_list


def remove_from_without_join(format_sql_3, column_names, schema_tokens):
    format_sql_4 = []
    table_name = None
    for sub_sql in format_sql_3.split('\n'):
        if 'select ' in sub_sql:
            if table_name:
                for i in range(len(format_sql_4)):
                    tokens = format_sql_4[i].split()
                    for ii, token in enumerate(tokens):
                        if token in column_names and tokens[ii - 1] != '.':
                            if (ii + 1 < len(tokens) and tokens[ii + 1] != '.' and tokens[
                                ii + 1] != '(') or ii + 1 == len(tokens):
                                if '{}.{}'.format(table_name, token) in schema_tokens:
                                    tokens[ii] = '{} . {}'.format(table_name, token)
                    format_sql_4[i] = ' '.join(tokens)

            format_sql_4.append(sub_sql)
        elif sub_sql.startswith('from'):
            sub_sql_tokens = sub_sql.split()
            if len(sub_sql_tokens) == 1:
                table_name = None
            elif len(sub_sql_tokens) == 2:
                table_name = sub_sql_tokens[1]
            else:
                print('bad from clause in remove_from_without_join')
                print(format_sql_3)
                exit()
        else:
            format_sql_4.append(sub_sql)

    if table_name:
        for i in range(len(format_sql_4)):
            tokens = format_sql_4[i].split()
            for ii, token in enumerate(tokens):
                if token in column_names and tokens[ii - 1] != '.':
                    if (ii + 1 < len(tokens) and tokens[ii + 1] != '.' and tokens[ii + 1] != '(') or ii + 1 == len(
                            tokens):
                        if '{}.{}'.format(table_name, token) in schema_tokens:
                            tokens[ii] = '{} . {}'.format(table_name, token)
            format_sql_4[i] = ' '.join(tokens)

    return format_sql_4


def add_table_name(format_sql_3, used_tables, column_names, schema_tokens):
    # If just one table used, easy case, replace all column_name -> table_name.column_name
    if len(used_tables) == 1:
        table_name = used_tables[0]
        format_sql_4 = []
        for sub_sql in format_sql_3.split('\n'):
            if sub_sql.startswith('from'):
                format_sql_4.append(sub_sql)
                continue

            tokens = sub_sql.split()
            for ii, token in enumerate(tokens):
                if token in column_names and tokens[ii - 1] != '.':
                    if (ii + 1 < len(tokens) and tokens[ii + 1] != '.' and tokens[ii + 1] != '(') or ii + 1 == len(
                            tokens):
                        if '{}.{}'.format(table_name, token) in schema_tokens:
                            tokens[ii] = '{} . {}'.format(table_name, token)
            format_sql_4.append(' '.join(tokens))
        return format_sql_4

    def get_table_name_for(token):
        table_names = []
        for table_name in used_tables:
            if '{}.{}'.format(table_name, token) in schema_tokens:
                table_names.append(table_name)
        if len(table_names) == 0:
            return 'table'
        if len(table_names) > 1:
            return None
        else:
            return table_names[0]

    format_sql_4 = []
    for sub_sql in format_sql_3.split('\n'):
        if sub_sql.startswith('from'):
            format_sql_4.append(sub_sql)
            continue

        tokens = sub_sql.split()
        for ii, token in enumerate(tokens):
            # skip *
            if token == '*':
                continue
            if token in column_names and tokens[ii - 1] != '.':
                if (ii + 1 < len(tokens) and tokens[ii + 1] != '.' and tokens[ii + 1] != '(') or ii + 1 == len(tokens):
                    table_name = get_table_name_for(token)
                    if table_name:
                        tokens[ii] = '{} . {}'.format(table_name, token)
        format_sql_4.append(' '.join(tokens))

    return format_sql_4


def normalize_space(format_sql):
    format_sql_1 = [' '.join(
        sub_sql.strip().replace(',', ' , ').replace('.', ' . ').replace('(', ' ( ').replace(')', ' ) ').split()) for
                    sub_sql in format_sql.split('\n')]
    format_sql_1 = '\n'.join(format_sql_1)

    format_sql_2 = format_sql_1.replace('\njoin', ' join').replace(',\n', ', ').replace(' where', '\nwhere').replace(
        ' intersect', '\nintersect').replace('\nand', ' and').replace('order by t2 .\nstart desc',
                                                                      'order by t2 . start desc')

    format_sql_2 = format_sql_2.replace('select\noperator', 'select operator').replace('select\nconstructor',
                                                                                       'select constructor').replace(
        'select\nstart', 'select start').replace('select\ndrop', 'select drop').replace('select\nwork',
                                                                                        'select work').replace(
        'select\ngroup', 'select group').replace('select\nwhere_built', 'select where_built').replace('select\norder',
                                                                                                      'select order').replace(
        'from\noperator', 'from operator').replace('from\nforward', 'from forward').replace('from\nfor',
                                                                                            'from for').replace(
        'from\ndrop', 'from drop').replace('from\norder', 'from order').replace('.\nstart', '. start').replace(
        '.\norder', '. order').replace('.\noperator', '. operator').replace('.\nsets', '. sets').replace(
        '.\nwhere_built', '. where_built').replace('.\nwork', '. work').replace('.\nconstructor',
                                                                                '. constructor').replace('.\ngroup',
                                                                                                         '. group').replace(
        '.\nfor', '. for').replace('.\ndrop', '. drop').replace('.\nwhere', '. where')

    format_sql_2 = format_sql_2.replace('group by', 'group_by').replace('order by', 'order_by').replace('! =',
                                                                                                        '!=').replace(
        'limit value', 'limit_value')
    return format_sql_2


def normalize_final_sql(format_sql_5):
    format_sql_final = format_sql_5.replace('\n', ' ').replace(' . ', '.').replace('group by', 'group_by').replace(
        'order by', 'order_by').replace('! =', '!=').replace('limit value', 'limit_value')

    # normalize two bad sqls
    if 't1' in format_sql_final or 't2' in format_sql_final or 't3' in format_sql_final or 't4' in format_sql_final:
        format_sql_final = format_sql_final.replace('t2.dormid', 'dorm.dormid')

    # This is the failure case of remove_from_without_join()
    format_sql_final = format_sql_final.replace(
        'select city.city_name where city.state_name in ( select state.state_name where state.state_name in ( select river.traverse where river.river_name = value ) and state.area = ( select min ( state.area ) where state.state_name in ( select river.traverse where river.river_name = value ) ) ) order_by population desc limit_value',
        'select city.city_name where city.state_name in ( select state.state_name where state.state_name in ( select river.traverse where river.river_name = value ) and state.area = ( select min ( state.area ) where state.state_name in ( select river.traverse where river.river_name = value ) ) ) order_by city.population desc limit_value')

    return format_sql_final


def normalize_original_sql(sql):
    sql = [i.lower() for i in sql]
    sql = ' '.join(sql).strip(';').replace("``", "'").replace("\"", "'").replace("''", "'")
    sql = sql.replace(')from', ') from')
    sql = sql.replace('(', ' ( ')
    sql = sql.replace(')', ' ) ')
    sql = re.sub('\s+', ' ', sql)

    sql = re.sub(r"(')(\S+)", r"\1 \2", sql)
    sql = re.sub(r"(\S+)(')", r"\1 \2", sql).split(' ')

    sql = ' '.join(sql)
    sql = sql.strip(' ;').replace('> =', '>=').replace('! =', '!=')
    return sql.split(' ')


def parse_sql(sql_string, db_id, column_names, schema_tokens, schema):
    format_sql = sqlparse.format(sql_string, reindent=True)
    format_sql_2 = normalize_space(format_sql)

    format_sql_3, used_tables, used_tables_list = remove_from_with_join(format_sql_2)

    format_sql_3 = '\n'.join(format_sql_3)
    format_sql_4 = add_table_name(format_sql_3, used_tables, column_names, schema_tokens)

    format_sql_4 = '\n'.join(format_sql_4)
    format_sql_5 = remove_from_without_join(format_sql_4, column_names, schema_tokens)

    format_sql_5 = '\n'.join(format_sql_5)
    format_sql_final = normalize_final_sql(format_sql_5)

    return format_sql_final


def read_spider_split(dataset_path, table_path, database_path):
    with open(dataset_path) as f:
        split_data = json.load(f)
    print('read_spider_split', dataset_path, len(split_data))

    schemas = read_dataset_schema(table_path, stanza_model)

    interaction_list = {}
    for i, ex in enumerate(tqdm(split_data)):
        db_id = ex['db_id']

        ex['query_toks_no_value'] = normalize_original_sql(ex['query_toks_no_value'])
        turn_sql = ' '.join(ex['query_toks_no_value'])
        turn_sql = turn_sql.replace('select count ( * ) from follows group by value',
                                    'select count ( * ) from follows group by f1')
        ex['query_toks_no_value'] = turn_sql.split(' ')

        ex = fix_number_value(ex)
        try:
            ex['query_toks_no_value'] = disambiguate_items(db_id, ex['query_toks_no_value'],
                                                           tables_file=table_path, allow_aliases=False)
        except:
            print(ex['query_toks'])
            continue

        final_sql_parse = ' '.join(ex['query_toks_no_value'])
        final_utterance = ' '.join(ex['question_toks']).lower()

        if stanza_model is not None:
            lemma_utterance_stanza = stanza_model(final_utterance)
            lemma_utterance = [word.lemma for sent in lemma_utterance_stanza.sentences for word in sent.words]
            original_utterance = final_utterance
        else:
            original_utterance = lemma_utterance = final_utterance.split(' ')

        # using db content
        db_context = SpiderDBContext(db_id,
                                     lemma_utterance,
                                     tables_file=table_path,
                                     dataset_path=database_path,
                                     stanza_model=stanza_model,
                                     schemas=schemas,
                                     original_utterance=original_utterance)

        value_match, value_alignment, exact_match, partial_match = db_context.get_db_knowledge_graph(db_id)

        if value_match != []:
            print(value_match, value_alignment)

        if db_id not in interaction_list:
            interaction_list[db_id] = []

        interaction = {}
        interaction['id'] = i
        interaction['database_id'] = db_id
        interaction['interaction'] = [{'utterance': final_utterance,
                                       'db_id': db_id,
                                       'query': ex['query'],
                                       'question': ex['question'],
                                       'sql': final_sql_parse,
                                       'value_match': value_match,
                                       'value_alignment': value_alignment,
                                       'exact_match': exact_match,
                                       'partial_match': partial_match,
                                       }]
        interaction_list[db_id].append(interaction)

    return interaction_list



def preprocess_dataset(dataset, dataset_dir, output_dir, table_path, database_path):
    # for session in ['train', 'dev']:
    for session in ['dev']:
        dataset_path = os.path.join(dataset_dir, f'{session}.json')
        interaction_list = read_spider_split(dataset_path, table_path, database_path)
        write_interaction(interaction_list, session, output_dir)

    return interaction_list


def preprocess(dataset, dataset_dir, table_path, database_path, output_dir):
    # directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # read schema
    print('Reading spider database schema file')
    schema_tokens, column_names, database_schemas = read_database_schema(table_path)
    print('total number of schema_tokens / databases:', len(schema_tokens))
    output_table_path = os.path.join(output_dir, 'tables.json')
    with open(output_table_path, 'w') as outfile:
        json.dump([v for k, v in database_schemas.items()], outfile, indent=4)

    # process (SQL, Query) pair in train/dev
    preprocess_dataset(dataset, dataset_dir, output_dir, table_path, database_path)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=('spider', 'sparc', 'cosql'), default='spider')
    args = parser.parse_args()

    dataset = args.dataset
    dataset_dir = f'./data/{dataset}/'
    table_path = f'./data/{dataset}/tables.json'
    database_path = f'./data/{dataset}/database'
    output_dir = f'./data/{dataset}_schema_linking_tag'

    preprocess(dataset, dataset_dir, table_path, database_path, output_dir)
