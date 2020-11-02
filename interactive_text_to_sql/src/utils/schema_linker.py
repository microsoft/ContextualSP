# coding: utf-8

import json
import os
import pickle as pkl

import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from src.utils.utils import lemma_token

wordnet_lemmatizer = WordNetLemmatizer()

VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'alse', 'when']
AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']


def load_dataset(table_path, data_path):
    table_list = json.load(open(table_path, 'r', encoding='utf-8'))
    spider_examples = json.load(open(data_path, 'r', encoding='utf-8'))
    tables = {}

    for table in table_list:
        db_id = table['db_id']
        table['col_table'] = [_[0] for _ in table['column_names']]
        table['schema_content'] = [_[1] for _ in table['column_names']]
        table['col_set'] = list(set(table['schema_content']))
        tables[db_id] = table

    for example in spider_examples:
        db_id = example['db_id']
        example['column_names'] = tables[db_id]['schema_content']
        example['table_names'] = tables[db_id]['table_names']
        example['col_set'] = tables[db_id]['col_set']
        example['col_table'] = tables[db_id]['col_table']
        keys = {}
        for kv in tables[db_id]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[db_id]['primary_keys']:
            keys[id_k] = id_k
        example['keys'] = keys

    return tables, spider_examples


def group_header(toks, idx, num_toks, header_toks):
    # a substring of toks[idx:] belongs to list header_toks
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        sub_toks = " ".join(sub_toks)
        if sub_toks in header_toks:
            return endIdx, sub_toks
    return idx, None


def fully_part_header(toks, idx, num_toks, header_toks):
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        if len(sub_toks) > 1:
            sub_toks = " ".join(sub_toks)
            if sub_toks in header_toks:
                return endIdx, sub_toks
    return idx, None


def partial_header(toks, idx, num_toks, header_toks):
    # a substring of tokens is a subset of a header's tokens
    def check_in(list_one, list_two):
        # print(set(list_one) & set(list_two),  len(list_one))
        if len(set(list_one) & set(list_two)) == len(list_one) and len(list_two) <= 3:
            # print(len(list_two), list_two, list_one)
            return True
    for endIdx in reversed(range(idx + 1, len(toks))):
        sub_toks = toks[idx: min(endIdx, len(toks))]
        if len(sub_toks) > 1:
            flag_count = 0
            tmp_heads = None
            for heads in header_toks:
                if check_in(sub_toks, heads):
                    flag_count += 1
                    tmp_heads = heads
            if flag_count == 1:
                return endIdx, tmp_heads
    return idx, None


def symbol_filter(questions):
    question_tmp_q = []
    for q_id, q_val in enumerate(questions):
        if len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�', '鈥�'] and q_val[-1] in ["'", '"', '`', '鈥�']:
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:-1])]
            question_tmp_q.append("'")
        elif len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�'] :
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:])]
        elif len(q_val) > 2 and q_val[-1] in ["'", '"', '`', '鈥�']:
            question_tmp_q += ["".join(q_val[0:-1])]
            question_tmp_q.append("'")
        elif q_val in ["'", '"', '`', '鈥�', '鈥�', '``', "''"]:
            question_tmp_q += ["'"]
        # elif q_val in [","]:
        #     question_tmp_q += ['銆�']
        else:
            question_tmp_q += [q_val]
    return question_tmp_q


def re_lemma(string):
    lema = lemma_token(string.lower())  # get base form of a verb
    if len(lema) > 0:
        return lema
    else:
        return string.lower()


def group_values(toks, idx, num_toks):
    # longest token sequence with upper capital letter
    def check_isupper(tok_lists):
        for tok_one in tok_lists:
            if tok_one[0].isupper() is False:
                return False
        return True

    for endIdx in reversed(range(idx + 1, num_toks + 1)):
        sub_toks = toks[idx: endIdx]

        if len(sub_toks) > 1 and check_isupper(sub_toks) is True:
            return endIdx, sub_toks
        if len(sub_toks) == 1:
            if sub_toks[0][0].isupper() and sub_toks[0].lower() not in VALUE_FILTER and \
                            sub_toks[0].lower().isalnum() is True:
                return endIdx, sub_toks
    return idx, None


def group_digital(toks, idx):
    test = toks[idx].replace(':', '')
    test = test.replace('.', '')
    if test.isdigit():
        return True
    else:
        return False


def group_symbol(toks, idx, num_toks):
    if toks[idx-1] == "'":
        for i in range(0, min(3, num_toks-idx)):
            if toks[i + idx] == "'":
                return i + idx, toks[idx:i+idx]
    return idx, None


def is_year_num(tok):
    if len(str(tok)) == 4 and str(tok).isdigit() and 15 < int(str(tok)[:2]) < 22:
        return True
    return False


class SchemaLinker(object):
    def __init__(self, table_path=None, conceptnet_path=None):
        self.tables = {}
        if table_path:
            if not os.path.exists(table_path):
                raise FileNotFoundError(f'{table_path} not found')
            table_list = json.load(open(table_path, 'r', encoding='utf-8'))
            for table in table_list:
                db_id = table['db_id']
                table['col_table'] = [_[0] for _ in table['column_names']]
                table['schema_content'] = [_[1] for _ in table['column_names']]
                table['col_set'] = list(set(table['schema_content']))
                self.tables[db_id] = table

        self.is_a_dict = []
        self.related_to_dict = []
        if conceptnet_path:
            if os.path.exists('cache/conceptnet/is_a.pkl') and os.path.exists('cache/conceptnet/relation_to.pkl'):
                self.is_a_dict = pkl.load(open('cache/conceptnet/is_a.pkl', 'rb'))
                self.related_to_dict = pkl.load(open('cache/conceptnet/relation_to.pkl', 'rb'))
            else:
                if not os.path.exists(conceptnet_path):
                    raise FileNotFoundError(f'{conceptnet_path} not found')
                is_a_dict = {}
                related_to_dict = {}
                with open(conceptnet_path, 'r', encoding='utf-8') as fr:
                    for line in tqdm(fr):
                        uri, relation, head, tail, detail = line.strip().split('\t')
                        head_split = head.split('/')[1:]
                        tail_split = tail.split('/')[1:]
                        if head_split[1] != 'en' or tail_split[1] != 'en':
                            continue
                        if relation == '/r/IsA':
                            is_a_dict[head_split[2]] = tail_split[2]
                        elif relation == '/r/RelatedTo':
                            related_to_dict[head_split[2]] = tail_split[2]
                # with open('data/concept_net/IsA.csv', 'r', encoding='utf-8') as fr1, \
                #     open('data/concept_net/RelatedTo.csv', 'r', encoding='utf-8') as fr2:
                #     for line in tqdm(fr1.readlines() + fr2.readlines()):
                #         uri, relation, head, tail, detail = line.strip().split('\t')
                #         head_split = head.split('/')[1:]
                #         tail_split = tail.split('/')[1:]
                #         if head_split[1] != 'en' or tail_split[1] != 'en':
                #             continue
                #         if relation == '/r/IsA':
                #             is_a_dict[head_split[2]] = tail_split[2]
                #         elif relation == '/r/RelatedTo':
                #             related_to_dict[head_split[2]] = tail_split[2]
                # pkl.dump(is_a_dict, open('cache/conceptnet/is_a.pkl', 'wb'))
                # pkl.dump(related_to_dict, open('cache/conceptnet/relation_to.pkl', 'wb'))
                self.is_a_dict = is_a_dict
                self.related_to_dict = related_to_dict

    def link_example(self, example, table_info=None):
        '''
        Add linking info for example
        example must contain: db_id, question_toks
        :param example:
        :param table_info:
        :return:
        '''
        db_id = example['db_id']
        if table_info is not None:
            assert table_info['db_id'] == db_id
        else:
            assert db_id in self.tables
            table_info = self.tables[db_id]
        example['column_names'] = table_info['schema_content']
        example['table_names'] = table_info['table_names']
        example['col_set'] = table_info['col_set']
        example['col_table'] = table_info['col_table']
        keys = {}
        for kv in table_info['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in table_info['primary_keys']:
            keys[id_k] = id_k
        example['keys'] = keys
        if 'origin_question_toks' not in example:
            example['origin_question_toks'] = example['question_toks']

        example['question_toks'] = symbol_filter(example['question_toks'])
        origin_question_toks = symbol_filter([x for x in example['origin_question_toks']])
        question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in example['question_toks']]

        example['question_toks'] = question_toks

        # This way for table_names lemmatizer
        table_names = []
        table_names_pattern = []

        for y in example['table_names']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            #         x = [lemma(x).lower() for x in y.split(' ')]
            table_names.append(" ".join(x))

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        # This is for header_toks lemmatizer
        header_toks = []
        header_toks_list = []

        header_toks_pattern = []
        header_toks_list_pattern = []

        for y in example['col_set']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        num_toks = len(question_toks)
        idx = 0
        tok_concol = []
        type_concol = []
        nltk_result = nltk.pos_tag(question_toks)

        while idx < num_toks:

            ############ fully header
            end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)  # length should > 1
            if header:
                tok_concol.append(origin_question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            ############ check for table

            end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(origin_question_toks[idx: end_idx])
                type_concol.append(["table"])
                idx = end_idx
                continue

            ########### check for col
            end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(origin_question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            end_idx, tname = partial_header(question_toks, idx, num_toks, header_toks_list)
            if tname:
                # tok_concol.append(tname)
                tok_concol.append(origin_question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
            if agg:
                tok_concol.append(origin_question_toks[idx: end_idx])
                type_concol.append(["agg"])
                idx = end_idx
                continue

            if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                tok_concol.append([origin_question_toks[idx]])
                type_concol.append(['MORE'])
                idx += 1
                continue

            if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                tok_concol.append([origin_question_toks[idx]])
                type_concol.append(['MOST'])
                idx += 1
                continue

            if is_year_num(question_toks[idx]):
                question_toks[idx] = 'year'
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(origin_question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

            pro_result = "NONE"

            def get_concept_result(toks, graph):
                find_col = False
                for begin_id in range(0, len(toks)):
                    for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                        tmp_query = "_".join(toks[begin_id:r_ind])
                        if tmp_query in graph:
                            mi = graph[tmp_query]
                            for col in example['col_set']:
                                if col in mi:
                                    return col

            end_idx, symbol = group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                origin_tmp_toks = [x for x in origin_question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                pro_result = get_concept_result(tmp_toks, self.is_a_dict)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, self.related_to_dict)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in origin_tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]
                origin_tmp_toks = [x for x in origin_question_toks[idx: end_idx] if x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = get_concept_result(tmp_toks, self.is_a_dict)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, self.related_to_dict)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in origin_tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            result = group_digital(question_toks, idx)
            if result is True:
                tok_concol.append(origin_question_toks[idx: idx + 1])
                type_concol.append(["value"])
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            tok_concol.append([origin_question_toks[idx]])
            type_concol.append(['NONE'])
            idx += 1
            continue

        example['question_arg'] = tok_concol
        example['question_arg_type'] = type_concol
        example['nltk_pos'] = nltk_result

        return example

    def link_nl(self, db_id, nl):
        if not isinstance(nl, list):
            nl = nl.split()
        ret = self.link_example({
            'db_id': db_id,
            'question_toks': nl,
        })
        return ret


if __name__ == '__main__':
    table_path = 'data/datasets/spider/tables.json'
    data_path = 'data/datasets/spider/dev.json'
    conceptnet_path = 'data/concept_net/conceptnet-assertions-5.6.0.csv'
    linker = SchemaLinker(table_path, conceptnet_path)
    examples = json.load(open(data_path, 'r', encoding='utf-8'))
    example = examples[0]
    import copy
    new_example = linker.link(copy.copy(example))
    pass
