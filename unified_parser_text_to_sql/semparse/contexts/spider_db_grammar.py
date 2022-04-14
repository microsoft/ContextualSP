# pylint: disable=anomalous-backslash-in-string
"""
A ``Text2SqlTableContext`` represents the SQL context in which an utterance appears
for the any of the text2sql datasets, with the grammar and the valid actions.
"""
from typing import List, Dict

from dataset_readers.dataset_util.spider_utils import Table


GRAMMAR_DICTIONARY = {}
GRAMMAR_DICTIONARY["statement"] = ['(query ws iue ws query)', '(query ws)']
GRAMMAR_DICTIONARY["iue"] = ['"intersect"', '"except"', '"union"']
GRAMMAR_DICTIONARY["query"] = ['(ws select_core ws groupby_clause ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause ws orderby_clause)',
                               '(ws select_core ws groupby_clause ws limit)',
                               '(ws select_core ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause)',
                               '(ws select_core ws orderby_clause)',
                               '(ws select_core)']

GRAMMAR_DICTIONARY["select_core"] = ['(select_with_distinct ws select_results ws from_clause ws where_clause)',
                                     '(select_with_distinct ws select_results ws from_clause)',
                                     '(select_with_distinct ws select_results ws where_clause)',
                                     '(select_with_distinct ws select_results)']
GRAMMAR_DICTIONARY["select_with_distinct"] = ['(ws "select" ws "distinct")', '(ws "select")']
GRAMMAR_DICTIONARY["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY["select_result"] = ['"*"', '(table_source ws ".*")',
                                       'expr', 'col_ref']

GRAMMAR_DICTIONARY["from_clause"] = ['(ws "from" ws table_source ws join_clauses)',
                                     '(ws "from" ws source)']
GRAMMAR_DICTIONARY["join_clauses"] = ['(join_clause ws join_clauses)', 'join_clause']
GRAMMAR_DICTIONARY["join_clause"] = ['"join" ws table_source ws "on" ws join_condition_clause']
GRAMMAR_DICTIONARY["join_condition_clause"] = ['(join_condition ws "and" ws join_condition_clause)', 'join_condition']
GRAMMAR_DICTIONARY["join_condition"] = ['ws col_ref ws "=" ws col_ref']
GRAMMAR_DICTIONARY["source"] = ['(ws single_source ws "," ws source)', '(ws single_source)']
GRAMMAR_DICTIONARY["single_source"] = ['table_source', 'source_subq']
GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")")']
# GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")" ws "as" ws name)', '("(" ws query ws ")")']
GRAMMAR_DICTIONARY["limit"] = ['("limit" ws non_literal_number)']

GRAMMAR_DICTIONARY["where_clause"] = ['(ws "where" wsp expr ws where_conj)', '(ws "where" wsp expr)']
GRAMMAR_DICTIONARY["where_conj"] = ['(ws "and" wsp expr ws where_conj)', '(ws "and" wsp expr)']

GRAMMAR_DICTIONARY["groupby_clause"] = ['(ws "group" ws "by" ws group_clause ws "having" ws expr)',
                                        '(ws "group" ws "by" ws group_clause)']
GRAMMAR_DICTIONARY["group_clause"] = ['(ws expr ws "," ws group_clause)', '(ws expr)']

GRAMMAR_DICTIONARY["orderby_clause"] = ['ws "order" ws "by" ws order_clause']
GRAMMAR_DICTIONARY["order_clause"] = ['(ordering_term ws "," ws order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY["ordering_term"] = ['(ws expr ws ordering)', '(ws expr)']
GRAMMAR_DICTIONARY["ordering"] = ['(ws "asc")', '(ws "desc")']

GRAMMAR_DICTIONARY["col_ref"] = ['(table_name ws "." ws column_name)', 'column_name']
GRAMMAR_DICTIONARY["table_source"] = ['(table_name ws "as" ws table_alias)', 'table_name']
GRAMMAR_DICTIONARY["table_name"] = ["table_alias"]
GRAMMAR_DICTIONARY["table_alias"] = ['"t1"', '"t2"', '"t3"', '"t4"']
GRAMMAR_DICTIONARY["column_name"] = []

GRAMMAR_DICTIONARY["ws"] = ['~"\s*"i']
GRAMMAR_DICTIONARY['wsp'] = ['~"\s+"i']

GRAMMAR_DICTIONARY["expr"] = ['in_expr',
                              # Like expressions.
                              '(value wsp "like" wsp string)',
                              # Between expressions.
                              '(value ws "between" wsp value ws "and" wsp value)',
                              # Binary expressions.
                              '(value ws binaryop wsp expr)',
                              # Unary expressions.
                              '(unaryop ws expr)',
                              'source_subq',
                              'value']
GRAMMAR_DICTIONARY["in_expr"] = ['(value wsp "not" wsp "in" wsp string_set)',
                                 '(value wsp "in" wsp string_set)',
                                 '(value wsp "not" wsp "in" wsp expr)',
                                 '(value wsp "in" wsp expr)']

GRAMMAR_DICTIONARY["value"] = ['parenval', '"YEAR(CURDATE())"', 'number', 'boolean',
                               'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY["function"] = ['(fname ws "(" ws "distinct" ws arg_list_or_star ws ")")',
                                  '(fname ws "(" ws arg_list_or_star ws ")")']

GRAMMAR_DICTIONARY["arg_list_or_star"] = ['arg_list', '"*"']
GRAMMAR_DICTIONARY["arg_list"] = ['(expr ws "," ws arg_list)', 'expr']
 # TODO(MARK): Massive hack, remove and modify the grammar accordingly
# GRAMMAR_DICTIONARY["number"] = ['~"\d*\.?\d+"i', "'3'", "'4'"]
GRAMMAR_DICTIONARY["non_literal_number"] = ['"1"', '"2"', '"3"', '"4"']
GRAMMAR_DICTIONARY["number"] = ['ws "value" ws']
GRAMMAR_DICTIONARY["string_set"] = ['ws "(" ws string_set_vals ws ")"']
GRAMMAR_DICTIONARY["string_set_vals"] = ['(string ws "," ws string_set_vals)', 'string']
# GRAMMAR_DICTIONARY["string"] = ['~"\'.*?\'"i']
GRAMMAR_DICTIONARY["string"] = ['"\'" ws "value" ws "\'"']
GRAMMAR_DICTIONARY["fname"] = ['"count"', '"sum"', '"max"', '"min"', '"avg"', '"all"']
GRAMMAR_DICTIONARY["boolean"] = ['"true"', '"false"']

# TODO(MARK): This is not tight enough. AND/OR are strictly boolean value operators.
GRAMMAR_DICTIONARY["binaryop"] = ['"+"', '"-"', '"*"', '"/"', '"="', '"!="', '"<>"',
                                  '">="', '"<="', '">"', '"<"', '"and"', '"or"', '"like"']
GRAMMAR_DICTIONARY["unaryop"] = ['"+"', '"-"', '"not"', '"not"']


def update_grammar_with_tables(grammar_dictionary: Dict[str, List[str]],
                               schema: Dict[str, Table]) -> None:
    table_names = sorted([f'"{table.lower()}"' for table in
                          list(schema.keys())], reverse=True)
    grammar_dictionary['table_name'] += table_names

    all_columns = set()
    for table in schema.values():
        all_columns.update([f'"{table.name.lower()}@{column.name.lower()}"' for column in table.columns if column.name != '*'])
    sorted_columns = sorted([column for column in all_columns], reverse=True)
    grammar_dictionary['column_name'] += sorted_columns


def update_grammar_to_be_table_names_free(grammar_dictionary: Dict[str, List[str]]):
    """
    Remove table names from column names, remove aliases
    """

    grammar_dictionary["column_name"] = []
    grammar_dictionary["table_name"] = []
    grammar_dictionary["col_ref"] = ['column_name']
    grammar_dictionary["table_source"] = ['table_name']

    del grammar_dictionary["table_alias"]


def update_grammar_flip_joins(grammar_dictionary: Dict[str, List[str]]):
    """
    Remove table names from column names, remove aliases
    """

    # using a simple rule such as join_clauses-> [(join_clauses ws join_clause), join_clause]
    # resulted in a max recursion error, so for now just using a predefined max
    # number of joins
    grammar_dictionary["join_clauses"] = ['(join_clauses_1 ws join_clause)', 'join_clause']
    grammar_dictionary["join_clauses_1"] = ['(join_clauses_2 ws join_clause)', 'join_clause']
    grammar_dictionary["join_clauses_2"] = ['(join_clause ws join_clause)', 'join_clause']