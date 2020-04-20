# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from scripts.eval.evaluation_sqa import Evaluator, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, \
    build_foreign_key_map_from_json
from scripts.eval.process_sql import Schema, get_schema, get_sql

_schemas = {}
kmaps = None


def evaluate(gold, predict, db_name, db_dir, table) -> int:
    global kmaps

    # try:
    evaluator = Evaluator()

    if kmaps is None:
        kmaps = build_foreign_key_map_from_json(table)

    if db_name in _schemas:
        schema = _schemas[db_name]
    else:
        db = os.path.join(db_dir, db_name, db_name + ".sqlite")
        schema = _schemas[db_name] = Schema(get_schema(db))

    g_sql = get_sql(schema, gold)

    try:
        p_sql = get_sql(schema, predict)
    except Exception as e:
        return 0

    # rebuild sql for value evaluation
    kmap = kmaps[db_name]
    g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

    exact_score = evaluator.eval_exact_match(p_sql, g_sql)
    return exact_score

