"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
import json
import os
import sqlite3
from collections import defaultdict
from typing import List, Dict, Optional, Any
from semparse.sql.process_sql import get_tables_with_alias, parse_sql


class TableColumn:
    def __init__(self,
                 name: str,
                 text: str,
                 column_type: str,
                 is_primary_key: bool,
                 foreign_key: Optional[str],
                 lemma: Optional[str]):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key
        self.lemma = lemma


class Table:
    def __init__(self,
                 name: str,
                 text: str,
                 columns: List[TableColumn],
                 lemma: Optional[str]):
        self.name = name
        self.text = text
        self.columns = columns
        self.lemma = lemma


def read_dataset_schema(schema_path: str, stanza_model=None) -> Dict[str, List[Table]]:
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    dbs_json_blob = json.load(open(schema_path, "r", encoding='utf-8'))
    for db in dbs_json_blob:
        db_id = db['db_id']

        column_id_to_table = {}
        column_id_to_column = {}

        concate_columns = [c[-1] for c in db['column_names']]
        concate_tables = [c for c in db['table_names']]

        #load stanza model
        if stanza_model is not None:
            lemma_columns = stanza_model('\n\n'.join(concate_columns).replace(' ','none'))
            lemma_columns_collect = []
            for sent in lemma_columns.sentences:
                tmp = []
                for word in sent.words:
                    if word.lemma != None:
                        tmp.append(word.lemma)
                    elif word.text==' ':
                        tmp.append('none')
                    else:
                        tmp.append(word.text)
                lemma_columns_collect.append(' '.join(tmp))

            lemma_tables = stanza_model('\n\n'.join(concate_tables).replace(' ','none'))
            lemma_tables_collect = {}
            for t,sent in zip(concate_tables, lemma_tables.sentences):
                tmp = []
                for word in sent.words:
                    if word.lemma != None:
                        tmp.append(word.lemma)
                    elif word.text == ' ':
                        tmp.append('none')
                    else:
                        tmp.append(word.text)
                lemma_tables_collect[t]=' '.join(tmp)
        else:
            lemma_columns_collect = concate_columns
            lemma_tables_collect = {t:t for t in concate_tables}

        for i, (column, text, column_type) in enumerate(zip(db['column_names_original'], db['column_names'], db['column_types'])):
            table_id, column_name = column
            _, column_text = text

            table_name = db['table_names_original'][table_id]

            if table_name not in schemas[db_id]:
                table_text = db['table_names'][table_id]
                table_lemma = lemma_tables_collect[table_text]
                schemas[db_id][table_name] = Table(table_name, table_text, [], table_lemma)

            if column_name == "*":
                continue

            is_primary_key = i in db['primary_keys']
            table_column = TableColumn(column_name.lower(), column_text, column_type, is_primary_key, None, lemma_columns_collect[i])
            schemas[db_id][table_name].columns.append(table_column)
            column_id_to_table[i] = table_name
            column_id_to_column[i] = table_column

        for (c1, c2) in db['foreign_keys']:
            foreign_key = column_id_to_table[c2] + ':' + column_id_to_column[c2].name
            column_id_to_column[c1].foreign_key = foreign_key

    return {**schemas}


def read_dataset_values(db_id: str, dataset_path: str, tables: List[str]):
    db = os.path.join(dataset_path, db_id, db_id + ".sqlite")
    try:
        conn = sqlite3.connect(db)
    except Exception as e:
        raise Exception(f"Can't connect to SQL: {e} in path {db}")
    conn.text_factory = str
    cursor = conn.cursor()

    values = {}

    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table.name} LIMIT 5000")
            values[table] = cursor.fetchall()
        except:
            conn.text_factory = lambda x: str(x, 'latin1')
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table.name} LIMIT 5000")
            values[table] = cursor.fetchall()

    return values


def ent_key_to_name(key):
    parts = key.split(':')
    if parts[0] == 'table':
        return parts[1]
    elif parts[0] == 'column':
        _, _, table_name, column_name = parts
        return f'{table_name}@{column_name}'
    else:
        return parts[1]


def fix_number_value(ex):
    """
    There is something weird in the dataset files - the `query_toks_no_value` field anonymizes all values,
    which is good since the evaluator doesn't check for the values. But it also anonymizes numbers that
    should not be anonymized: e.g. LIMIT 3 becomes LIMIT 'value', while the evaluator fails if it is not a number.
    """

    def split_and_keep(s, sep):
        if not s: return ['']  # consistent with string.split()

        # Find replacement character that is not used in string
        # i.e. just use the highest available character plus one
        # Note: This fails if ord(max(s)) = 0x10FFFF (ValueError)
        p = chr(ord(max(s)) + 1)

        return s.replace(sep, p + sep + p).split(p)

    # input is tokenized in different ways... so first try to make splits equal
    query_toks = ex['query_toks']
    ex['query_toks'] = []
    for q in query_toks:
        ex['query_toks'] += split_and_keep(q, '.')

    i_val, i_no_val = 0, 0
    while i_val < len(ex['query_toks']) and i_no_val < len(ex['query_toks_no_value']):
        if ex['query_toks_no_value'][i_no_val] != 'value':
            i_val += 1
            i_no_val += 1
            continue

        i_val_end = i_val
        while i_val + 1 < len(ex['query_toks']) and \
                i_no_val + 1 < len(ex['query_toks_no_value']) and \
                ex['query_toks'][i_val_end + 1].lower() != ex['query_toks_no_value'][i_no_val + 1].lower():
            i_val_end += 1

        if i_val == i_val_end and ex['query_toks'][i_val] in ["1", "2", "3", "4", "5"] and ex['query_toks'][i_val - 1].lower() == "limit":
            ex['query_toks_no_value'][i_no_val] = ex['query_toks'][i_val]
        i_val = i_val_end

        i_val += 1
        i_no_val += 1

    return ex


_schemas_cache = None


def disambiguate_items(db_id: str, query_toks: List[str], tables_file: str, allow_aliases: bool) -> List[str]:
    """
    we want the query tokens to be non-ambiguous - so we can change each column name to explicitly
    tell which table it belongs to

    parsed sql to sql clause is based on supermodel.gensql from syntaxsql
    """

    class Schema:
        """
        Simple schema which maps table&column to a unique identifier
        """

        def __init__(self, schema, table):
            self._schema = schema
            self._table = table
            self._idMap = self._map(self._schema, self._table)

        @property
        def schema(self):
            return self._schema

        @property
        def idMap(self):
            return self._idMap

        def _map(self, schema, table):
            column_names_original = table['column_names_original']
            table_names_original = table['table_names_original']
            # print 'column_names_original: ', column_names_original
            # print 'table_names_original: ', table_names_original
            for i, (tab_id, col) in enumerate(column_names_original):
                if tab_id == -1:
                    idMap = {'*': i}
                else:
                    key = table_names_original[tab_id].lower()
                    val = col.lower().replace(' ','_')
                    idMap[key + "." + val] = i

            for i, tab in enumerate(table_names_original):
                key = tab.lower()
                idMap[key] = i

            return idMap

    def get_schemas_from_json(fpath):
        global _schemas_cache

        if _schemas_cache is not None:
            return _schemas_cache

        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        db_names = [db['db_id'] for db in data]

        tables = {}
        schemas = {}
        for db in data:
            db_id = db['db_id']
            schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
            column_names_original = db['column_names_original'] if 'column_names_original' in db else db['column_names']
            table_names_original = db['table_names_original'] if 'table_names_original' in db else db['table_names']
            tables[db_id] = {'column_names_original': column_names_original,
                             'table_names_original': table_names_original}
            for i, tabn in enumerate(table_names_original):
                table = str(tabn.lower())
                cols = [str(col.lower().replace(' ','_')) for td, col in column_names_original if td == i]
                schema[table] = cols
            schemas[db_id] = schema

        _schemas_cache = schemas, db_names, tables
        return _schemas_cache

    schemas, db_names, tables = get_schemas_from_json(tables_file)
    schema = Schema(schemas[db_id], tables[db_id])

    fixed_toks = []
    i = 0
    while i < len(query_toks):
        tok = query_toks[i]
        if tok == 'value' or tok == "'value'":
            # TODO: value should alawys be between '/" (remove first if clause)
            new_tok = f'"{tok}"'
        elif tok in ['!','<','>'] and query_toks[i+1] == '=':
            new_tok = tok + '='
            i += 1
        # elif i+1 < len(query_toks) and query_toks[i+1] == '.' and query_toks[i] in schema.schema.keys():
        elif i + 1 < len(query_toks) and query_toks[i + 1] == '.':
            new_tok = ''.join(query_toks[i:i+3])
            i += 2
        else:
            new_tok = tok
        fixed_toks.append(new_tok)
        i += 1

    toks = fixed_toks

    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql, mapped_entities = parse_sql(toks, 0, tables_with_alias, schema, mapped_entities_fn=lambda: [])

    for i, new_name in mapped_entities:
        curr_tok = toks[i]
        if '.' in curr_tok and allow_aliases:
            parts = curr_tok.split('.')
            assert(len(parts) == 2)
            toks[i] = parts[0] + '.' + new_name
        else:
            toks[i] = new_name

    if not allow_aliases:
        toks = [tok for tok in toks if tok not in ['as', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']]

    toks = [f'\'value\'' if tok == '"value"' else tok for tok in toks]

    return toks


def remove_on(query):
    query_tok = query.split()
    sql_words = []
    t = 0
    while t < len(query_tok):
        if query_tok[t] != 'on':
            sql_words.append(query_tok[t])
            t += 1
        else:
            t += 4
    return ' '.join(sql_words)

def read_dataset_values_from_json(db_id: str, db_content_dict: Dict[str, Any], tables: List[str]):

    values = {}
    item = db_content_dict[db_id]
    for table in tables:
        values[table] = item['tables'][table.name]['cell']
    return values

def extract_tree_style(sent):
    """
    sent: List
    """
    rnt = []

if __name__ == '__main__':
    import stanza
    stanza_model = stanza.Pipeline('en')
    doc = stanza_model("what is the name of the breed with the most dogs ?")
    word=[word.lemma for sent in doc.sentences for word in sent.words]
    rnt = []