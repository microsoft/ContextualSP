# coding: utf-8

import json
from typing import List

from src.utils.utils import lemma_token


STOP_WORD_LIST = [_.strip() for _ in open('data/common/stop_words.txt', 'r', encoding='utf-8').readlines()]


def align_two_sentences_in_token_level(token_list1, token_list2, stop_word_list=[]):
    token_list1 = [(word, idx) for idx, word in enumerate(token_list1) if word not in stop_word_list]
    token_list2 = [(word, idx) for idx, word in enumerate(token_list2) if word not in stop_word_list]

    def find_exact_match_pairs_from_two_sentences(word_list1, word_list2):
        pairs = []
        for word1, idx1 in word_list1:
            for word2, idx2 in word_list2:
                if word1 == word2:
                    word_list2.remove((word2, idx2))
                    pairs.append((word1, idx1, word2, idx2))
        return pairs

    exact_match = find_exact_match_pairs_from_two_sentences(token_list1, token_list2)
    exact_match_tokens_idx1 = [_[1] for _ in exact_match]
    exact_match_tokens_idx2 = [_[3] for _ in exact_match]
    sentence1_lemma = [(lemma_token(word), idx) for word, idx in token_list1 if idx not in exact_match_tokens_idx1]
    sentence2_lemma = [(lemma_token(word), idx) for word, idx in token_list2 if idx not in exact_match_tokens_idx2]
    lemma_match = find_exact_match_pairs_from_two_sentences(sentence1_lemma, sentence2_lemma)
    lemma_match_tokens_idx1 = [_[1] for _ in lemma_match]
    lemma_match_tokens_idx2 = [_[3] for _ in lemma_match]
    return exact_match + lemma_match


def find_keyword_alignment_by_rule(nl_tokens: List, keyword: str, stop_word_list: List = STOP_WORD_LIST,
                                   only_one_match: bool = False, aligned_mark: List[bool] = None):
    aligned_results = []
    position_pairs = set()

    # step0: eliminate stop words, but keep position info
    keyword = keyword.split()
    for stop_word in stop_word_list:
        if stop_word in keyword:
            keyword = keyword.remove(stop_word)
    keyword_lemma = [lemma_token(_) for _ in keyword]
    informative_token_pairs = []
    informative_token_lemma_pairs = []
    for pos, token in enumerate(nl_tokens):
        if token not in stop_word_list:
            informative_token_pairs.append((token, pos))
            informative_token_lemma_pairs.append((lemma_token(token), pos))
    if not aligned_mark:
        aligned_mark = [False for _ in range(len(informative_token_pairs))]

    # step1: exact match
    for i in range(len(informative_token_pairs) - len(keyword) + 1):
        if only_one_match and True in aligned_mark[i: i + len(keyword)]:
            continue
        st_position = informative_token_pairs[i][1]
        ed_position = informative_token_pairs[i + len(keyword) - 1][1]
        if [_[0] for _ in informative_token_pairs[i: i + len(keyword)]] == keyword \
                and (st_position, ed_position) not in position_pairs:
            aligned_results.append((st_position, ed_position, 'exact', keyword))
            position_pairs.add((st_position, ed_position))
            if only_one_match:
                for j in range(i, i + len(keyword)):
                    aligned_mark[j] = True

    # step2: lemma exactly match
    for i in range(len(informative_token_lemma_pairs) - len(keyword_lemma) + 1):
        if only_one_match and True in aligned_mark[i: i + len(keyword_lemma)]:
            continue
        st_position = informative_token_lemma_pairs[i][1]
        ed_position = informative_token_lemma_pairs[i + len(keyword) - 1][1]
        if [_[0] for _ in informative_token_lemma_pairs[i: i + len(keyword_lemma)]] == keyword_lemma \
                and (st_position, ed_position) not in position_pairs:
            aligned_results.append((st_position, ed_position, 'exact lemma', keyword))
            position_pairs.add((st_position, ed_position))
            if only_one_match:
                for j in range(i, i + len(keyword_lemma)):
                    aligned_mark[j] = True

    def check_in(utterance_span, keyword_tokens):
        return len(set(utterance_span) & set(keyword_tokens)) == len(utterance_span) and len(keyword_tokens) <= 3

    # step3: partial match
    for i in range(len(informative_token_pairs) - len(keyword) + 1):
        st_position = informative_token_pairs[i][1]
        for end_idx in reversed(range(i + 1, len(informative_token_pairs))):
            if only_one_match and True in aligned_mark[i: end_idx]:
                continue
            sub_tokens = [_[0] for _ in informative_token_pairs[i:end_idx]]
            if not sub_tokens:
                continue
            else:
                ed_position = informative_token_pairs[end_idx - 1][1]
                if check_in(sub_tokens, keyword):
                    aligned_results.append((st_position, ed_position, 'partial', keyword))
                    if only_one_match:
                        for j in range(i, end_idx):
                            aligned_mark[j] = True

    # step4: lemma partial match
    for i in range(len(informative_token_lemma_pairs) - len(keyword) + 1):
        for end_idx in reversed(range(i + 1, len(informative_token_lemma_pairs))):
            if only_one_match and True in aligned_mark[i: end_idx]:
                continue
            sub_tokens = [_[0] for _ in informative_token_lemma_pairs[i:end_idx]]
            if not sub_tokens:
                continue
            else:
                if check_in(sub_tokens, keyword):
                    aligned_results.append((informative_token_lemma_pairs[i][1],
                                            informative_token_lemma_pairs[end_idx - 1][1],
                                            'partial lemma', keyword))
                    if only_one_match:
                        for j in range(i, end_idx):
                            aligned_mark[j] = True
    return aligned_results, aligned_mark


def find_alignment_by_rule(nl_tokens: List, table_names: List, column_names: List, values: List, only_one_match=False):
    aligned_mark = None
    # step1: find value match
    value_matches = []
    for value in values:
        value_match, aligned_mark = \
            find_keyword_alignment_by_rule(nl_tokens, value, STOP_WORD_LIST,
                                           only_one_match=only_one_match, aligned_mark=aligned_mark)
        value_matches += value_match

    # step2: find table match
    table_matches = []
    for table_name in table_names:
        table_match, aligned_mark = \
            find_keyword_alignment_by_rule(nl_tokens, table_name, STOP_WORD_LIST,
                                           only_one_match=only_one_match, aligned_mark=aligned_mark)
        table_matches += table_match

    # step3 find column match
    column_matches = []
    for column_name in column_names:
        column_match, aligned_mark = \
            find_keyword_alignment_by_rule(nl_tokens, column_name, STOP_WORD_LIST,
                                           only_one_match=only_one_match, aligned_mark=aligned_mark)
        column_matches += column_match
    alignment_results = {'value': value_matches, 'table': table_matches, 'column': column_matches}
    return alignment_results


def test():
    nl_tokens = 'show me the name of all English songs and their singers'.split()
    table_names = ['singer', 'song']
    column_names = ['singer name', 'song name', 'age', 'year']
    values = ['English', 'Show time']
    ret = find_alignment_by_rule(nl_tokens, table_names, column_names, values, only_one_match=False)
    print(json.dumps(ret, indent=4))


if __name__ == '__main__':
    test()
