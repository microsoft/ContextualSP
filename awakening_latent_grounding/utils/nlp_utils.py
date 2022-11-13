import os
from collections import defaultdict
from contracts.base_types import *
from contracts.schemas import *
from typing import Tuple, Dict, List
from multiprocessing import Pool
from Levenshtein import ratio

Proj_Abs_Dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except:
        return False


def parse_number_value(value: str):
    try:
        i = int(value)
        return i
    except:
        pass
    try:
        f = float(value)
        return f
    except:
        pass

    raise ValueError("{} can't be parsed to number".format(value))


def permutate_ngrams(tokens: List[str], sep: str = ' ', max_N: int = None) -> List[Tuple[int, int, str]]:
    ngrams = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            ngram = sep.join(tokens[i:j + 1]).lower()
            ngrams.append((i, j, ngram))

            if max_N is not None and j - i + 1 >= max_N:
                break

    return ngrams


def _merge_tokens_with_index_mappings(tokens: List[str]):
    tokens_str = ''
    start_map, end_map = {}, {}
    for i, token in enumerate(tokens):
        if i > 0:
            tokens_str += ' '
        start_map[len(tokens_str)] = i
        tokens_str += token
        end_map[len(tokens_str) - 1] = i

    return tokens_str, (start_map, end_map)


def _normalize_value(value: str) -> str:
    return value.replace(" ", "")


def _match_ngram_values(query: Dict) -> List[CellValue]:
    matched_values = []
    column: str = query['column']
    threshold: float = query['threshold']
    question: Utterance = query['question']
    for value in query['values']:
        for i, j, ngram in permutate_ngrams(question.text_tokens, sep=' ', max_N=5):
            score = ratio(_normalize_value(value), _normalize_value(ngram))
            if score < threshold:
                continue
            matched_values += [
                CellValue(column=column, name=value, tokens=question.tokens[i:j + 1],
                          span=Span(start=i, end=j),
                          score=score)]

    return list(sorted(matched_values, key=lambda x: x.score, reverse=True))


class ValueMatcher:
    processes: int
    column_with_values: List[Tuple[Column, List[str]]]  # column with distinct values

    def __init__(self, columns: List[Tuple[Column, List[object]]], processes: int = 16) -> None:
        self.processes = processes
        column_with_values = []
        for column, values in columns:
            # if column.data_type == DataType.Number or column.data_type == DataType.DateTime:
            #     column_with_values.append((column, []))
            #     continue
            distinct_values = list(set([str(x).lower().strip() for x in values]))
            column_with_values.append((column, distinct_values))

        self.column_with_values = column_with_values

    def match_text_values(self, question: Utterance, threshold: float, top_k: int = 3, **kwargs) -> List[CellValue]:
        # Init matching pool
        pool = Pool(self.processes)
        queries = []
        for column, values in self.column_with_values:
            # if len(values) == 0:
            #     continue
            queries.append(
                {'column': column.identifier, 'values': values, 'question': question, 'threshold': threshold})

        all_ngram_matches = pool.map(_match_ngram_values, queries)
        ngram_matches = []
        for matches in all_ngram_matches:
            ngram_matches += matches[:top_k]

        non_overlap_matches: List[CellValue] = []
        for match in sorted(ngram_matches, key=lambda x: x.score * 1000 + x.span.length, reverse=True):
            is_overlap = False
            for match2 in non_overlap_matches:
                if not (match.span.start > match2.span.end or match.span.end < match2.span.start or (
                        match2.score - match.score) < 1e-2):
                    is_overlap = True
                    break
                if match2.span.start <= match.span.start and match2.span.end >= match.span.end and match2.span.length == match.span.length:
                    is_overlap = True
                    break
            if not is_overlap:
                non_overlap_matches.append(match)

        return non_overlap_matches

    def match_values(self, question: Utterance, threshold: float, top_k: int = 3, **kwargs) -> List[CellValue]:
        all_matches = []
        text_matches = self.match_text_values(question, threshold, top_k)
        all_matches = text_matches

        number_matches = self.match_number_values(question)
        for num_match in number_matches:
            is_overlap = False
            for match in text_matches:
                if match.span.start <= num_match.span.start and match.span.end >= num_match.span.end and (
                        match.span.length > num_match.span.length):
                    is_overlap = True
                    break
            if not is_overlap:
                all_matches.append(num_match)

        return all_matches


class NGramMatcher(object):
    def __init__(self, ngram_tokens: Tuple[str, List[str]], sep: str = ' ') -> None:
        self.sep = sep
        self.ngrams_dict = self._initialize_ngrams(ngram_tokens)

    def _initialize_ngrams(self, ngram_tokens: Tuple[str, List[str]]) -> Dict[str, List[Tuple[str, int, int]]]:
        ngrams_dict = defaultdict(list)
        for key, tokens in ngram_tokens:
            for i, j, ngram in permutate_ngrams(tokens):
                ngrams_dict[ngram].append((key, i, j))
        return ngrams_dict

    def match(self, query_tokens: List[str]):
        all_matches = []
        for i, j, ngram in permutate_ngrams(query_tokens):
            if ngram not in self.ngrams_dict:
                continue
            for key, k_i, k_j in self.ngrams_dict[ngram]:
                all_matches.append((i, j, key, k_i, k_j))

        non_overlaps = []
        for q_i, q_j, key, k_i, k_j in sorted(all_matches, key=lambda x: x[1] - x[0], reverse=True):
            is_overlap = False
            for q_i2, q_j2, key2, k_i2, k_j2 in non_overlaps:
                if key == key2 and q_i2 <= q_i and q_j2 >= q_j:
                    is_overlap = True
                    break

            if not is_overlap:
                non_overlaps.append((q_i, q_j, key, k_i, k_j))

        return non_overlaps


class PhraseMatcher(object):
    def __init__(self, path: str, sep: str) -> None:
        self.sep = sep
        self.prefix_dict = self.load_prefix_dict(path)

    def load_prefix_dict(self, path: str):
        prefix_dict = {}
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                items = line.split('\t')
                assert len(items) > 0
                assert len(items[0]) > 0

                tokens = items[0].split(self.sep) if len(self.sep) > 0 else list(items[0])
                prefix = ""
                for token in tokens:
                    if len(token) == 0:
                        continue
                    if len(prefix) == 0:
                        prefix += self.sep

                    prefix += token
                    if prefix in prefix_dict:
                        continue
                    prefix_dict[prefix] = False  # a prefix
                if len(prefix) > 0:
                    prefix_dict[prefix] = True  # a phrase
        return prefix_dict

    def prefix_match(self, tokens):
        query = self.sep.join(tokens)
        return query in self.prefix_dict

    def extract_match(self, tokens):
        query = self.sep.join(tokens)
        return query in self.prefix_dict and self.prefix_dict[query] == True

