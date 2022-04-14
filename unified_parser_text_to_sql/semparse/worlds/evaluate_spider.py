import os
import sqlite3

from semparse.worlds.evaluate import Evaluator, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, \
    build_foreign_key_map_from_json
from semparse.sql.process_sql import Schema, get_schema, get_sql

_schemas = {}
kmaps = None


def evaluate(gold, predict, db_name, db_dir, table, check_valid: bool=True, db_schema=None) -> bool:
    global kmaps

    gold=gold.replace("t1 . ","").replace("t2 . ",'').replace("t3 . ",'').replace("t4 . ",'').replace(" as t1",'').replace(" as t2",'').replace(" as t3",'').replace(" as t4",'')
    predict=predict.replace("t1 . ","").replace("t2 . ",'').replace("t3 . ",'').replace("t4 . ",'').replace(" as t1",'').replace(" as t2",'').replace(" as t3",'').replace(" as t4",'')

    # sgrammar = SpiderGrammar(
    #     output_from=True,
    #     use_table_pointer=True,
    #     include_literals=True,
    #     include_columns=True,
    # )
    # try:
    evaluator = Evaluator()

    if kmaps is None:
        kmaps = build_foreign_key_map_from_json(table)


    if 'chase' in db_dir:
        schema = _schemas[db_name] = Schema(db_schema)
    elif db_name in _schemas:
        schema = _schemas[db_name]
    else:
        db = os.path.join(db_dir, db_name, db_name + ".sqlite")
        schema = _schemas[db_name] = Schema(get_schema(db))

    g_sql = get_sql(schema, gold)
    # try:
    p_sql = get_sql(schema, predict)
    # except Exception as e:
    #     print('evaluate_spider.py L39')
    #     return False

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
    return True
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
        return True if not return_error else (True, None)
    except Exception as e:
        return False if not return_error else (False, e.args[0])
