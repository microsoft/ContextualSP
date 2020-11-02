# coding: utf-8

from enum import Enum
import json

from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from spacy.symbols import ORTH, LEMMA

from src.context.converter import SQLConverter
from src.context.db_context import SparcDBContext
from src.utils.spider_evaluate.convert_sql_to_detail import convert_sql_to_detail
from src.utils.wikisql_lib.query import Query as WikiSQLQuery
from src.utils.semql_tree_util import parse_sql_tree


class SpecialSymbol(Enum):
    history_start = '@@BEG@@'
    history_end = '@@END@@'


class SemQLConverter(object):
    def __init__(self):
        spacy_tokenizer = SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        for token in SpecialSymbol.__members__.values():
            token_uni = u'{}'.format(token.value)
            spacy_tokenizer.spacy.tokenizer.add_special_case(token_uni, [{ORTH: token_uni, LEMMA: token_uni}])

        self._tokenizer = WordTokenizer(spacy_tokenizer)

        self.log = {}

    def convert_example(self, example):
        raise NotImplementedError


class SpiderSemQLConverter(SemQLConverter):
    def __init__(self):
        super().__init__()

    def convert_example(self, example):
        db_id = example['db_id']
        utterance = example['question']
        sql_converter = self._get_converter(db_id, utterance)
        semql_statements = sql_converter.translate_to_intermediate(example['sql'])
        return semql_statements

    def convert_sql_to_semql(self, db_info, utterance, sql):
        db_id = db_info['db_id']
        sql_converter = self._get_converter(db_id, utterance)
        db_info['column_index'] = {name.lower(): idx for idx, (_, name) in enumerate(db_info['column_names_original'])}
        db_info['table_index'] = {name.lower(): idx for idx, name in enumerate(db_info['table_names_original'])}
        sql_detail = convert_sql_to_detail(sql, db_id, db_info, db_dir='data/spider/database')
        sql_detail['select'] = list(sql_detail['select'])
        semql_statements = sql_converter.translate_to_intermediate(sql_detail)
        return semql_statements

    def _get_converter(self, db_id, utterance, table_file='data/spider/tables.json', database_path='data/spider/database'):
        db_context = SparcDBContext(db_id=db_id, utterance=utterance, tokenizer=self._tokenizer,
                                    tables_file=table_file, database_path=database_path)
        sql_converter = SQLConverter(db_context=db_context)
        return sql_converter


class WikiSQLConverter(SemQLConverter):
    def __init__(self, table_file):
        super().__init__()
        table_infos = [json.loads(line) for line in open(table_file, 'r', encoding='utf-8')]
        self.table_infos = {table_info['id']: table_info for table_info in table_infos}

    def convert_example(self, example):
        table_id = example['table_id']
        table_info = self.table_infos[table_id]
        table_col_names = table_info['header']
        table_name = 'table'
        utterance = example['question']
        sql_detail = example['sql']
        cond_statements_raw = []
        conds = sql_detail['conds']
        for (col_idx, operator_idx, condition_value) in conds:
            col_name = '_'.join(table_col_names[col_idx].split()).lower()
            operator = WikiSQLQuery.cond_ops[operator_idx]
            cond_statements_raw.append([f'Filter -> {operator} A', f'A -> none C T', f'C -> {col_name}', f'T -> {table_name}'])
        self.log['n_conditions'] = self.log.get('n_conditions', {})
        self.log['n_conditions'][len(cond_statements_raw)] = self.log['n_conditions'].get(len(cond_statements_raw), 0) + 1
        cond_statements = []
        if len(cond_statements_raw) == 1:
            cond_statements = cond_statements_raw[0]
        elif len(cond_statements_raw) >= 2:  # add subfilter statement
            for i in range(len(cond_statements_raw) - 1):
                cond_statements.append(f'Filter -> Filter and Filter')
                cond_statements += cond_statements_raw[i]
            cond_statements += cond_statements_raw[-1]
        assert isinstance(sql_detail['sel'], int), f'Selected column index is not int, which actually is {sql_detail["sel"]}'
        sel_col_name = '_'.join(table_col_names[sql_detail['sel']].split()).lower()
        agg_op = WikiSQLQuery.agg_ops[sql_detail['agg']].lower()  # ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        if not agg_op:
            agg_op = 'none'
        sel_statements = ['Select -> A', f'A -> {agg_op} C T', f'C -> {sel_col_name}', f'T -> {table_name}']
        semql_statements = ['Statement -> Root', 'Root -> Select Filter' if cond_statements else 'Root -> Select'] \
                           + sel_statements + cond_statements
        return semql_statements


if __name__ == '__main__':
    # wikisql_converter = WikiSQLConverter('data/wikisql/data/test.tables.jsonl')
    # examples = [json.loads(line) for line in open('data/wikisql/data/test.jsonl', 'r', encoding='utf-8')]
    # semql_statements = wikisql_converter.convert_example(examples[0])
    # print(semql_statements)
    # sql_tree, depth = parse_sql_tree(semql_statements)
    # print(sql_tree.restatement(with_table=False))

    spider_converter = SpiderSemQLConverter()
    examples = json.load(open('data/spider/dev.json', 'r'))
    semql_states = spider_converter.convert_example(examples[0])
    print(semql_states)
