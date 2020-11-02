# coding: utf-8

import os
import json
import re
import pickle as pkl

import numpy as np

from src.utils.utils import lemma_token
from src.utils.link_util import find_alignment_by_rule, find_keyword_alignment_by_rule
from src.utils.utils import STOP_WORD_LIST


def jaccard_distance(word_list1, word_list2):
    word_set1 = set(word_list1)
    word_set2 = set(word_list2)
    return len(word_set1 & word_set2) / len(word_set1 | word_set2)


class QuestionGenerator(object):
    AGGRs = ['minimum', 'maximum', 'average', 'sum']

    def __init__(self, dataset_path='data/spider/', n_options=3):
        # common
        self.dataset_path = dataset_path
        self.database_path = os.path.join(dataset_path, 'database')
        table_file_path = os.path.join(self.dataset_path, 'tables.json')
        self.dbs_schemas = {_['db_id']: _ for _ in json.load(open(table_file_path, 'r', encoding='utf-8'))}
        self.n_options = n_options
        # example
        self.database = ''
        self.table_names, self.column_names = [], []
        self.utterance_tokens = []  # NOTICE: bert tokenized tokens
        self.utterance_tokens_no_stopwords = []
        # glove
        self.glove_dict = {}
        self.glove_vectors = None
        self._load_glove_vectors()
        self.glove_unk = self.glove_dict.get('unk', 0)

    def _load_glove_vectors(self):
        glove_path = 'data/common/glove_tune.42B.300d.txt'
        glove_dict = {}
        vectors = []
        with open(glove_path, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                line = line.strip()
                token, vec = line.split(' ', 1)
                vec = [float(_) for _ in vec.split()]
                assert len(vec) == 300, 'Glove vector not in dimension 300'
                glove_dict[token] = idx
                vectors.append(vec)
        vectors = np.array(vectors)
        self.glove_dict = glove_dict
        self.glove_vectors = vectors

    def refresh(self, database, utterance_tokens):
        self.database = database
        db_schemas = self.dbs_schemas[self.database]
        self.column_names = [_[1] for _ in db_schemas['column_names'] if _[1] is not '*']
        self.table_names = db_schemas['table_names']
        self.utterance_tokens = utterance_tokens
        self.utterance_tokens_no_stopwords = []
        for token_idx, token in enumerate(self.utterance_tokens):
            if token not in STOP_WORD_LIST:
                self.utterance_tokens_no_stopwords.append((token_idx, token))

    def generate_question(self, token_idx):
        asked_token = self.utterance_tokens[token_idx]
        column_similarities = self.get_keyword_similarities(asked_token, self.column_names)
        table_similarities = self.get_keyword_similarities(asked_token, self.table_names)
        aggr_similarities = self.get_keyword_similarities(asked_token, self.AGGRs)
        similarities = [(span, score, 'column_name') for (span, score) in column_similarities] + \
                       [(span, score, 'table_name') for (span, score) in table_similarities] + \
                       [(span, score, 'aggr') for (span, score) in aggr_similarities]
        similarities.sort(key=lambda x: -x[1])
        options = similarities[:self.n_options]
        options.append(('none', 0, 'none'))
        options.append(('value', 0, 'value'))
        return {'question': asked_token, 'options': options}

    def get_keyword_similarities(self, token, spans):
        scores = [(span, self.calculate_token_span_similarity(token, span)) for span in spans]  # todo: warning: error here
        scores.sort(key=lambda x: -x[1])
        return scores

    def calculate_token_span_similarity(self, token, span):
        assert len(re.split(' |_', token)) == 1
        # part1: surface similarity
        surface_similarity, status = 0.0, ''
        span_tokens = re.split(' |_', span)
        token_lemma = lemma_token(token)
        span_tokens_lemma = [lemma_token(_) for _ in span_tokens]
        if token == ''.join(span_tokens):
            surface_similarity = 5.0
            status = 'exact'
        elif token_lemma == ''.join(span_tokens_lemma):
            surface_similarity = 3.0
            status = 'exact lemma'
        elif token in span:
            surface_similarity = 1.5
            status = 'partial'
        elif token_lemma in span_tokens_lemma:
            surface_similarity = 1.0
            status = 'partial lemma'
        surface_similarity += jaccard_distance([token], span_tokens) + \
                              jaccard_distance([token_lemma], span_tokens_lemma)
        # part2: embedding similarity
        token_vector = self.glove_vectors[self.glove_dict.get(token, self.glove_unk)]
        span_vector = [self.glove_vectors[self.glove_dict.get(t, self.glove_unk)] for t in span_tokens]
        if len(span_vector) > 1:
            span_vector = np.mean(span_vector)
        embedding_similarity = np.mean(token_vector * span_vector)
        return surface_similarity + embedding_similarity


if __name__ == '__main__':
    question_generator = QuestionGenerator()
    def similarity_test(key_token, value_tokens):
        ret = []
        for value_token in value_tokens:
            score = question_generator.calculate_token_span_similarity(key_token, value_token)
            ret.append((value_token, score))
        return ret
    key_token = ''
    value_token = ''
    while True:
        s = input().lower()
        if s.startswith('key:'):
            key_token = s[4:].strip()
        elif s.startswith('value:'):
            value_token = s[6:].strip()
            print(question_generator.calculate_token_span_similarity(key_token, value_token))
        elif s.startswith('values:'):
            value_tokens = eval(s[7:].strip())
            print(similarity_test(key_token, value_tokens))
