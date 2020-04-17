import os
import sqlite3

from scripts.eval.evaluation_sqa import Evaluator, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, \
    build_foreign_key_map_from_json
from scripts.eval.process_sql import Schema, get_schema, get_sql

_schemas = {}
kmaps = None


def evaluate(gold, predict, db_name, db_dir, table, check_valid: bool=True) -> int:
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

    if not check_valid:
        return exact_score
    else:
        return exact_score and check_valid_sql(predict, db_name, db_dir)
    # except Exception as e:
    #     return 0


_conns = {}


def check_valid_sql(sql, db_name, db_dir, return_error=False):
    db = os.path.join(db_dir, db_name, db_name + ".sqlite")

    if db_name == 'wta_1':
        # TODO: seems like there is a problem with this dataset - slow response - add limit 1
        return True if not return_error else (True, None)

    if db_name not in _conns:
        _conns[db_name] = sqlite3.connect(db)

        # fixes an encoding bug
        _conns[db_name].text_factory = bytes

    conn = _conns[db_name]
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        cursor.fetchall()
        return 1 if not return_error else (1, None)
    except Exception as e:
        return 0 if not return_error else (0, e.args[0])


def evaluate_hardness(gold, db_name, db_dir, table) -> str:
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
    hardness = evaluator.eval_hardness(g_sql)
    return hardness
