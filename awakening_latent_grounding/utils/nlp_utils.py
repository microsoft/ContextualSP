import os
from multiprocessing import Pool

import recognizers_suite as Recognizers
from Levenshtein import ratio
from recognizers_suite import Culture

from utils.data_types import *


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except:
        return False


def is_adjective(value: str) -> bool:
    return value in ['old', 'older', 'oldest', 'young', 'youngest', 'younger', 'heavy', 'heavier', 'heaviest']


def permutate_ngrams(tokens: List[str], sep: str = ' ') -> List[Tuple[int, int, str]]:
    ngrams = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            ngram = sep.join(tokens[i:j + 1]).lower()
            ngrams.append((i, j, ngram))
    return ngrams


class Vocab:
    id2tokens: Dict[int, str]
    token2ids: Dict[str, int]

    def __init__(self, tokens: List[str], special_tokens: List[str]) -> None:
        token2ids = {}
        for token in special_tokens:
            assert token not in token2ids
            token2ids[token] = len(token2ids)

        for token in tokens:
            assert token not in token2ids
            token2ids[token] = len(token2ids)

        self.token2ids = token2ids
        self.id2tokens = {idx: token for token, idx in token2ids.items()}

    def __len__(self) -> int:
        return len(self.token2ids)

    def lookup_id(self, token: str, default_token: str = UNK_Token) -> int:
        if token in self.token2ids:
            return self.token2ids[token]

        if default_token in self.token2ids:
            return self.token2ids[default_token]

        raise ValueError("Token {} not found in vocab".format(token))

    def lookup_token(self, idx: int) -> str:
        return self.id2tokens[idx]

    @classmethod
    def from_file(cls, path: str, special_tokens: List[str] = [SOS_Token, EOS_Token], min_freq: int = 5):
        tokens = []
        assert os.path.exists(path), '{} not found'.format(path)
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                items = line.split('\t')
                if len(items) != 2:
                    raise ValueError()

                token = items[0]
                freq = int(items[1])
                if freq >= min_freq:
                    tokens.append(token)

        return Vocab(tokens, special_tokens)


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


def merge_tokens_with_index_mappings(tokens: List[str]):
    tokens_str = ''
    start_map, end_map = {}, {}
    for i, token in enumerate(tokens):
        if i > 0:
            tokens_str += ' '
        start_map[len(tokens_str)] = i
        tokens_str += token
        end_map[len(tokens_str) - 1] = i

    return tokens_str, (start_map, end_map)


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


def recognize_numbers(tokens: List[str], enable_ordinal: bool = True, culture: Culture = Culture.English):
    numbers = []
    tokens_str, (start_map, end_map) = merge_tokens_with_index_mappings(tokens)

    results = Recognizers.recognize_number(tokens_str, culture)
    if enable_ordinal:
        results += Recognizers.recognize_ordinal(tokens_str, culture)

    for result in results:
        if result.start not in start_map or result.end not in end_map:
            continue
        start, end = start_map[result.start], end_map[result.end]
        if result.resolution is not None and 'value' in result.resolution:
            value = parse_number_value(result.resolution['value'])
            if result.type_name == 'ordinal' and value == '0':
                continue
            numbers.append((start, end, value, result.type_name))

    return numbers


@dataclass
class ValueMatch:
    column: str
    value: str
    start: int
    end: int
    score: float
    label: bool = False

    def __str__(self) -> str:
        return f'{self.value}[{self.start}:{self.end}]/{self.column}/{self.score:.3f}/{self.label}'

    def to_json(self):
        return self.__dict__

    @classmethod
    def from_json(cls, obj: Dict):
        return ValueMatch(**obj)


def lookup_values(input) -> List[ValueMatch]:
    query_tokens, column, values, threshold = input
    column_matches = []
    for value in values:
        for i, j, ngram in permutate_ngrams(query_tokens):
            if j - i > 6:
                continue
            score = ratio(value.replace(" ", ""), ngram.replace(" ", "").lower())
            if score < threshold:
                continue
            column_matches.append(ValueMatch(column=column, value='{}'.format(value), start=i, end=j, score=score))

    return list(sorted(column_matches, key=lambda x: x.score, reverse=True))


class ValueMatcher:
    def __init__(self, columns: List[Tuple[str, str, List[object]]]):
        self.columns = []
        for column_name, column_type, values in columns:
            distinct_values = list(set([str(x).lower().strip() for x in values]))  # [:200]
            self.columns.append((column_name, column_type, distinct_values))

    def match_text_values(self, query_tokens: List[str], threshold: float, top_k: int = 3) -> List[ValueMatch]:
        ngram_matches = []
        pool = Pool(16)
        inputs = [(query_tokens, column, values, threshold) for column, data_type, values in self.columns if
                  data_type == 'text']
        column_ngram_matches = pool.map(lookup_values, inputs)
        pool.close()
        pool.join()
        for matches in column_ngram_matches:
            ngram_matches += matches[:top_k]

        non_overlap_matches: List[ValueMatch] = []
        for match in sorted(ngram_matches, key=lambda x: x.score * 1000 + x.end - x.start, reverse=True):
            is_overlap = False
            for match2 in non_overlap_matches:
                if not (match.start > match2.end or match.end < match2.start or (match2.score - match.score) < 1e-2):
                    is_overlap = True
                    break
                if match2.start <= match.start and match2.end >= match.end and (
                        (match2.end - match2.start) > (match.end - match.start)):
                    is_overlap = True
                    break
            if not is_overlap:
                non_overlap_matches.append(match)

        # match substring/
        for i, token in enumerate(query_tokens):
            is_string = False
            if i - 1 >= 0 and query_tokens[i - 1] == '\'' \
                    and i + 1 < len(query_tokens) and query_tokens[i + 1] == '\'':
                is_string = True

            if not is_string:
                continue

            for column, data_type, _ in self.columns:
                if data_type == 'text':
                    non_overlap_matches.append(
                        ValueMatch(column=column, value=token.lower(), start=i, end=i, score=0.5))

        return non_overlap_matches

    def match_number_values(self, query_tokens: List[str]) -> List[ValueMatch]:
        numbers = recognize_numbers(query_tokens, False, Culture.English)
        matches = []
        for (start, end, value, _) in numbers:
            matches.append(ValueMatch(column='*', value=value, start=start, end=end, score=1.0))
            for column, data_type, _ in self.columns:
                if data_type not in ['number', 'int', 'real']:
                    continue
                matches.append(ValueMatch(column=column, value=value, start=start, end=end, score=1.0))
        return matches

    def match(self, query_tokens: List[str], threshold: float, top_k: int = 3) -> List[ValueMatch]:
        all_matches = []
        text_matches = self.match_text_values(query_tokens, threshold, top_k)
        all_matches = text_matches

        number_matches = self.match_number_values(query_tokens)
        for num_match in number_matches:
            is_overlap = False
            for match in text_matches:
                if match.start <= num_match.start and match.end >= num_match.end and (
                        (match.end - match.start) > (num_match.end - num_match.start)):
                    is_overlap = True
                    break
            if not is_overlap:
                all_matches.append(num_match)

        return all_matches
