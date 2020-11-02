# coding: utf-8

import re

import nltk

from src.utils.utils import STOP_WORD_LIST
from src.utils.external import complex_rephrase


class NLModifier(object):
    def __init__(self, mode='simple'):
        self.database = ''
        self.utterance = ''
        self.utterance_tokens = []
        self.utterance_tokens_no_stopwords = []
        self.utterance_pos = []
        assert mode.lower() in ('simple', 'rule', 'complex')
        self.mode = mode.lower()

    def refresh(self, database, utterance):
        self.database = database
        self.utterance = utterance
        self.utterance_tokens = nltk.word_tokenize(utterance)
        self.utterance_pos = [_[1] for _ in nltk.pos_tag(self.utterance_tokens)]
        self.utterance_tokens_no_stopwords = []
        for token_idx, token in enumerate(self.utterance_tokens):
            if token not in STOP_WORD_LIST:
                self.utterance_tokens_no_stopwords.append((token_idx, token))

    def modify(self, token, schema_item):
        if self.mode == 'simple':
            self.utterance_tokens = [ori_token if ori_token != token else schema_item.value
                                     for ori_token in self.utterance_tokens]
        elif self.mode == 'rule':
            if schema_item is None:
                return
            column_name, token_type = schema_item.value, schema_item.type
            assert token_type in ('column_name', 'value')
            # find spans
            column_name_tokens = re.split(' |_', column_name)
            labels = [True for _ in range(len(self.utterance_tokens_no_stopwords))]
            origin_span_idxs = []
            for list_idx, (token_idx, utt_token) in enumerate(self.utterance_tokens_no_stopwords):
                if utt_token == token:
                    labels[list_idx] = True
                    st, ed = token_idx, token_idx
                    for i in range(list_idx - 1, -1, -1):
                        if self.utterance_tokens_no_stopwords[i][1] in column_name_tokens:
                            labels[i] = True
                            st = self.utterance_tokens_no_stopwords[i][0]
                        else:
                            break
                    for i in range(list_idx + 1, len(self.utterance_tokens_no_stopwords)):
                        if self.utterance_tokens_no_stopwords[i][i] in column_name_tokens:
                            labels[i] = True
                            ed = self.utterance_tokens_no_stopwords[i][0]
                        else:
                            break
                    origin_span_idxs.append((token_idx, st, ed))
            assert len(self.utterance_tokens_no_stopwords) == len(labels)
            self.utterance_tokens_no_stopwords, self.utterance_pos_no_stopwords = \
                [self.utterance_tokens_no_stopwords[i] for i in range(len(labels)) if labels[i] is False], \
                [self.utterance_pos_no_stopwords[i] for i in range(len(labels)) if labels[i] is False]
            # adopt replacing rules
            for token_idx, span_st, span_ed in origin_span_idxs:
                token_pos = self.utterance_pos[token_idx]
                if token_pos in ('NN', 'NNS', 'NNP', 'NNPS'):  # noun
                    if token_type is 'column_name':
                        self.utterance_tokens = self.utterance_tokens[:span_st] \
                                                + column_name.lower() \
                                                + self.utterance_tokens[span_ed + 1:]
                    elif token_type is 'value':
                        self.utterance_tokens[token_idx] = self.utterance_tokens[token_idx].capitalize()
                elif token_pos in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):  # verb, not supported
                    pass
                    if token_type is 'column_name':
                        self.utterance_tokens.insert(token_idx + 1, schema_item.value)
                    elif token_type is 'value':
                        self.utterance_tokens.insert(token_idx + 1, schema_item.value)
                elif token_pos in ('JJ', 'JJR', 'JJS'):  # adjective, adjective comparative, adjective superlative
                    assert token_type is 'value'
                    # adj + n -> adj + COLUMN + n
                    self.utterance_tokens.insert(token_idx, column_name)
                elif token_pos in ('RB', 'RBR', 'RBS'):
                    pass  # cannot handle
                else:
                    raise ValueError('Cannot modify words not in n, v, adj')

            # remove doubled words
            new_utterance = []
            last_word = ''
            for word in self.utterance_tokens:
                if word != last_word:
                    new_utterance.append(word)
                last_word = word
            self.utterance_tokens = new_utterance
        elif self.mode == 'complex':
            utterance = complex_rephrase(' '.join(self.utterance_tokens), token, schema_item.value)
            self.utterance_tokens = utterance.split()

    def get_utterance(self):
        return ' '.join(self.utterance_tokens)


