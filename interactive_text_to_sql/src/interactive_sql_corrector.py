# coding: utf-8

import os

import json
import logging
import pickle as pkl

import numpy as np

from parsers.parser import Parser, IRNetSpiderParser
from src.utils.algo_utils import BipartiteGraphSolver
from src.utils.visualize_utils import draw_attention_hotmap
from src.components.human_simulator import HumanSimulator
from src.utils.semql_tree_util import Node as SemQLTree
from src.aligner_model import BertAligner
from src.data import SpiderSemQLConverter, BertUtil
from src.components import NLModifier, QuestionGenerator
from src.utils.utils import STOP_WORD_LIST, TEMPLATE_KEYWORDS


coterms = [x.strip() for x in open('data/spider/coterms.txt', 'r').readlines()]
stopwords = [x.strip() for x in open('data/common/stop_words.txt', 'r').readlines()]
coterms += stopwords


class SchemaValue:
    COLUMN_NAME_TYPE = 'column_name'
    TABLE_NAME_TYPE = 'table_name'
    AGGR_TYPE = 'aggr'
    COLUMN_VALUE_TYPE = 'value'
    NONE_TYPE = 'none'

    def __init__(self, nl_type, value):
        self.type = nl_type
        self.value = value


class InteractiveSqlCorrector:
    EVALUATE_MODE = 0
    SIMULATE_MODE = 1

    def __init__(self,
                 aligner: BertAligner = None,
                 mode: str = 'evaluate',
                 human_simulator: HumanSimulator = None,
                 align_threshold=0.628):
        assert mode.lower() in ('evaluate', 'interact'), "Mode must be evaluate or interact"
        if aligner:
            self.aligner = aligner
        else:
            self.aligner = BertAligner()
        if mode.lower() == 'evaluate':
            self.mode = InteractiveSqlCorrector.EVALUATE_MODE
        else:
            self.mode = InteractiveSqlCorrector.SIMULATE_MODE
        if self.mode == InteractiveSqlCorrector.SIMULATE_MODE:
            if human_simulator is None:
                raise ValueError('Should pass HumanSimulation object')
            self.human_simulator = human_simulator

        logger.info(f'Alignment threshold is set to be {align_threshold}')
        self.align_threshold = align_threshold

        self.semql_converter = SpiderSemQLConverter()
        self.bert_util = BertUtil()
        self.bipartite_graph_solver = BipartiteGraphSolver()
        self.question_generator = QuestionGenerator(n_options=3)
        self.nl_modifier = NLModifier()

        table_path = 'data/spider/tables.json'
        self.all_database_info = {_['db_id']: _ for _ in json.load(open(table_path, 'r', encoding='utf-8'))}
        # conceptnet_path = 'data/concept_net/conceptnet-assertions-5.6.0.csv'
        # self.schema_linker = SchemaLinker(self.table_path, conceptnet_path)

        self.nl_to_schema = dict()

    def start_session(self, example, predict_sql, label='', question_field='question', query_field='query'):
        # generate input for alignment model
        question = example[question_field]
        query = example[query_field]
        db_id = example['db_id']

        # Use ground-truth as parsed result
        # semql = self.semql_converter.convert_example(example)
        # semql_tree = SemQLTree.from_statements([str(_) for _ in semql])
        # restatement = semql_tree.restatement()

        db_schemas = self.all_database_info[db_id]
        table_names = db_schemas['table_names']
        column_names = [x[1] for x in db_schemas['column_names'] if x[1] != '*']
        schema_names = table_names + column_names + ['average', 'maximum', 'minimum', 'sum']
        schema_name_vocab = {token: schema_names.count(token) for token in schema_names}

        semql_statements = self.semql_converter.convert_sql_to_semql(self.all_database_info[db_id], question, predict_sql)
        predict_semql: SemQLTree = SemQLTree.from_statements([str(_) for _ in semql_statements])
        restatement = predict_semql.restatement()

        # 1. run aligner model to get alignment scores
        alignment_matrix, ids, tokens, lengths = self.aligner.calculate_alignment(question.lower(), restatement.lower())
        tokens1, tokens2 = self.aligner.split_tokens(tokens, lengths)
        # if len(tokens1) > len(tokens2):
        #     logger.error("NL longer than restate SQL")
        #     return
        alignment_matrix = alignment_matrix.squeeze(0).detach().cpu().numpy()
        # alignment_matrix = self.post_process_alignment(alignment_matrix=alignment_matrix,
        #                                                col_tokens=tokens1, row_tokens=tokens2,
        #                                                col_stopwords=STOP_WORD_LIST, row_stopwords=TEMPLATE_KEYWORDS)
        for i, token in enumerate(tokens2):
            alignment_matrix[:, i] /= schema_name_vocab.get(token, 1)
        max_assignment_score, assignment = self.bipartite_graph_solver.find_max(alignment_matrix)
        # assert (assignment[0] == list(range(lengths[0]))).all()
        src_aggr_alignment_score = alignment_matrix[assignment[0], assignment[1]]
        # draw_attention_hotmap(alignment_matrix, tokens1, tokens2)
        # pkl.dump((alignment_matrix, tokens1, tokens2), open('matrix2.pkl', 'wb'))

        self.question_generator.refresh(database=db_id, utterance_tokens=tokens1)
        self.nl_modifier.refresh(database=db_id, utterance=question)

        # 2. find knowledge span, ask for explanation
        question_list = []
        # exact match replacement
        long_schema_names = sorted([x for x in schema_names if len(x.split()) >= 2], key=lambda x: -len(x.split()))
        exact_match_onehot = [0 for _ in range(len(tokens))]
        for name in long_schema_names:
            st_char_position = question.lower().find(name)
            if name != -1:
                st_position = len(question[:st_char_position].split())
                ed_position = st_position + len(name.split())
                for _ in range(st_position, ed_position):
                    exact_match_onehot[_] = 1
        asked_tokens = []
        for position, score in enumerate(src_aggr_alignment_score):
            asked_token = tokens1[position]
            if asked_token in coterms or score > self.align_threshold:
                continue
            asked_tokens.append(asked_token)
            if exact_match_onehot[position] == 0:
                score = score * self._compute_score_rate(asked_token, schema_names)
            if score < self.align_threshold:
                if asked_token not in STOP_WORD_LIST:
                    token_question, options = self.get_question(token_idx=position)
                    schema_item = self.get_response(asked_token, options)
                    self.nl_modifier.modify(asked_token, schema_item)
                    # self.nl_to_schema[asked_token] = (schema_value.type, schema_value.value)

        # 3. reparse the modified NL
        new_question = self.nl_modifier.get_utterance()
        return new_question

    @staticmethod
    def _compute_score_rate(token, schema_names):
        cnt = 0
        for schema_name in schema_names:
            if token in schema_name:
                cnt += 1
        return 1.0 / max(cnt, 1)

    @staticmethod
    def restatement(node: SemQLTree, with_tag=False):
        if with_tag:
            return node.restatement_with_tag()
        else:
            return node.restatement()

    def get_question(self, token_idx):
        d = self.question_generator.generate_question(token_idx)
        question = d['question']
        options = d['options']
        return question, options

    def get_response(self, token, options) -> SchemaValue:
        if self.mode == InteractiveSqlCorrector.SIMULATE_MODE:
            schema_value = SchemaValue(options[0][2], options[0][0])  # todo: delete here
        else:
            print(token)
            print('0: None')
            print('1: It is a value')
            for idx, option in enumerate(options):
                print(f'{idx + 2}: {option[0]}')
            # response = int(input())
            response = 2
            if response == 0:
                schema_value = None
            elif response == 1:
                schema_name = ''
                schema_type = SchemaValue.COLUMN_VALUE_TYPE
                schema_value = SchemaValue(schema_type, schema_name)
            else:
                selected_option = options[response]
                schema_name = selected_option[0]
                if selected_option[2] is 'column':
                    schema_type = SchemaValue.COLUMN_NAME_TYPE
                else:
                    schema_type = SchemaValue.TABLE_NAME_TYPE
                schema_value = SchemaValue(schema_type, schema_name)
        return schema_value

    @staticmethod
    def post_process_alignment(alignment_matrix, col_tokens, row_tokens, col_stopwords, row_stopwords):
        weight_matrix = np.ones(alignment_matrix.shape)
        for idx, col_token in enumerate(col_tokens):
            if col_token in col_stopwords:
                weight_matrix[idx, :] = 0.5
        for idx, row_token in enumerate(row_tokens):
            if row_token in row_stopwords:
                weight_matrix[:, idx] = 0.5
        alignment_matrix *= weight_matrix
        return alignment_matrix

    def generate_options_for_token(self, db_id, token, top=5):
        def calculate_span_similarity(span1, span2):
            set_span1 = set(span1.split())
            set_span2 = set(span2.split())
            return len(set(set_span1) & set(set_span2)) / len(set_span1 | set_span2)
        # extract table names and column names from database info
        database_info = self.all_database_info[db_id]
        table_names = database_info['table_names']
        column_names = [_[1] for _ in database_info['column_names'] if _[1] if not '*']
        # compare token with table names and column names
        names = [(name, 'table') for name in table_names] + [(name, 'column') for name in column_names]
        names_with_score = []
        for name, source in names:
            score = calculate_span_similarity(token, name)
            names_with_score.append((name, source, score))
        names_with_score = sorted(names_with_score, key=lambda x: x[2], reverse=True)
        if top != -1:
            names_with_score = names_with_score[:top]
        return [(name, source) for name, source, score in names_with_score]


def main():
    interactive_sql_corrector = InteractiveSqlCorrector(aligner=None, mode='interact', human_simulator=HumanSimulator())
    examples = json.load(open('data/spider/dev.json', 'r'))
    predictions = open('data/parsers/irnet/output_origin.txt', 'r', encoding='utf-8').readlines()
    assert len(examples) == len(predictions)
    for i, (example, predict_sql) in enumerate(zip(examples, predictions)):
        print(i)
        interactive_sql_corrector.start_session(example, predict_sql)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()
