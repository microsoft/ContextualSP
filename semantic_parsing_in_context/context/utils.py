# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
import json
import os
import sqlite3
from collections import defaultdict
from typing import List, Dict, Optional


class TableColumn:
    """
    Representing the column of table
    """

    def __init__(self,
                 name: str,
                 text: str,
                 column_type: str,
                 is_primary_key: bool,
                 refer_table,
                 foreign_key: Optional[List[str]]):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key
        self.refer_table = refer_table

    def __str__(self):
        return f'{self.name}'


class Table:
    """
    Representing the table
    """

    def __init__(self,
                 name: str,
                 text: str,
                 columns: List[TableColumn]):
        self.name = name
        self.text = text
        self.columns = columns


def read_dataset_schema(schema_path: str):
    """
    Reading all table from `schema_path`.
    :param schema_path: default from `tables.json` of sparc data folder.
    :return:
    """
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    schema_id_to_table: Dict[str, Dict[int, Table]] = defaultdict(dict)
    schema_id_to_col: Dict[str, Dict[int, TableColumn]] = defaultdict(dict)

    dbs_json_blob = json.load(open(schema_path, "r"))
    for db in dbs_json_blob:
        db_id = db['db_id']

        column_id_to_table = {}
        column_id_to_column = {}

        for i, (column, text, column_type) in enumerate(zip(db['column_names_original'],
                                                            db['column_names'],
                                                            db['column_types'])):
            table_id, column_name = column
            _, column_text = text

            table_name = db['table_names_original'][table_id]

            if table_name not in schemas[db_id]:
                table_text = db['table_names'][table_id]
                table_obj = Table(table_name, table_text, [])
                schemas[db_id][table_name] = table_obj

            table_obj = schemas[db_id][table_name]

            if column_name == "*":
                # TODO: we cannot add an extra command to handle * problem.
                #  we now use a special embedding for linking * and predicting action
                is_primary_key = False
            else:
                is_primary_key = i in db['primary_keys']
            # allocate new column object
            column_obj = TableColumn(column_name.lower(), column_text, column_type,
                                     is_primary_key, table_obj, None)
            schemas[db_id][table_name].columns.append(column_obj)
            column_id_to_column[i] = column_obj

        for (c1, c2) in db['foreign_keys']:
            foreign_key = column_id_to_column[c2].refer_table.name + ':' + column_id_to_column[c2].name

            # TODO: we able multiple foreign keys existing to allow the shortcut join
            if column_id_to_column[c1].foreign_key is None:
                column_id_to_column[c1].foreign_key = []

            column_id_to_column[c1].foreign_key.append(foreign_key)

        for i, table_name in enumerate(db['table_names_original']):
            column_id_to_table[i] = schemas[db_id][table_name]

        # assign id to column and id to table
        schema_id_to_table[db_id] = column_id_to_table
        schema_id_to_col[db_id] = column_id_to_column
    return {**schemas}, {**schema_id_to_col}, {**schema_id_to_table}


def read_dataset_values(db_id: str, database_path: str, tables: List) -> Dict:
    db = os.path.join(database_path, db_id, db_id + ".sqlite")
    values = {}

    if not os.path.exists(db):
        # try to read it using ".json" suffix
        # assume the table is constrained on single-table
        table = tables[0]
        db = os.path.join(database_path, db_id, db_id + ".json")
        values[table] = json.load(open(db, "r", encoding="utf8"))["rows"]
    else:
        try:
            conn = sqlite3.connect(db)
        except Exception as e:
            raise Exception(f"Can't connect to SQL: {e} in path {db}")
        conn.text_factory = str
        cursor = conn.cursor()

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
