from enum import Enum
from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from contracts import Utterance, Token, DataType, Span, DBSchema, CellValue, LanguageCode

SPM_UNDERLINE = '▁'

class TableSpecialToken(str, Enum):
    Unknown = '<unk>'
    Table = '<tab>'
    Column = '<col>'
    Value = '<val>'
    Empty = '<empty>'

class PrefixMatcher(object):
    def __init__(self, path: str, sep: str="") -> None:
        self.sep = sep
        self.prefix_dict = self.load_prefix_dict(path)

    def load_prefix_dict(self, path: str):
        prefix_dict = {}
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                phrase = line.rstrip()
                tokens = list(phrase)
                if len(tokens) == 0:
                    continue
                prefix = ""
                for token in tokens:
                    prefix += token
                    if prefix in prefix_dict:
                        continue
                    prefix_dict[prefix] = False # a prefix

                if len(prefix) > 0:
                    prefix_dict[prefix] = True # a phrase

        return prefix_dict

    def prefix_match(self, tokens):
        query = self.sep.join(tokens)
        return query in self.prefix_dict

    def extract_match(self, tokens):
        query = self.sep.join(tokens)
        return query in self.prefix_dict and self.prefix_dict[query] == True

def preprocess_phrase_tokens(phrase_matcher: PrefixMatcher, tokens: List[str]) -> List[str]:
    assert len(tokens) > 0
    processed_tokens, idx_mappings = [], []
    idx = 0
    while idx < len(tokens):
        phrase_tokens = [tokens[idx]]

        next_idx, max_idx = idx, idx
        while phrase_matcher.prefix_match(phrase_tokens):
            if phrase_matcher.extract_match(phrase_tokens):
                max_idx = next_idx

            next_idx += 1
            if next_idx < len(tokens):
                phrase_tokens.append(tokens[next_idx])
            else:
                break

        for _ in range(idx, max_idx + 1):
            idx_mappings.append(len(processed_tokens))

        phrase_token = "".join(tokens[idx:max_idx + 1])
        processed_tokens.append(phrase_token)
        idx = max_idx + 1

    assert len(idx_mappings) == len(tokens), "Invalid index mappings for preporcessing."
    return processed_tokens, idx_mappings

def preprocess_phrase_nl_tokens(phrase_matcher: PrefixMatcher, tokens: List[Token]) -> Utterance:
    _, idx_mappings = preprocess_phrase_tokens(phrase_matcher, [x.token for x in tokens])
    processed_tokens: List[Token] = []
    for i, token in enumerate(tokens):
        if i > 0 and idx_mappings[i-1] == idx_mappings[i]:
            processed_tokens[-1].token += token.token
            processed_tokens[-1].lemma += token.lemma
            continue

        processed_tokens.append(token)

    return processed_tokens, idx_mappings

def is_char_based(lang: LanguageCode):
    return lang in [LanguageCode.zh]

@dataclass
class DataEncoder:
    tokenizer: PreTrainedTokenizer
    do_lower_case: bool = True
    @property
    def special_tokens(self) -> Dict[object, str]:
        tokenizer_class_name = self.tokenizer.__class__.__name__.lower()
        if tokenizer_class_name.startswith('xlm') or tokenizer_class_name.startswith('roberta'):
            return {
                DataType.Text : '<text>',
                DataType.Number : '<number>',
                DataType.DateTime : '<date>',
                DataType.Boolean : '<bool>',
                TableSpecialToken.Column: '<col>',
                TableSpecialToken.Value: '<val>',
                TableSpecialToken.Empty: '<empty>',
            }
        else:
            raise NotImplementedError(tokenizer_class_name)

    def _tokenize(self, tokens: List[str], remove_spm_prefix: bool=True):
        all_spm_tokens, spm_indices = [], []
        for idx, token in enumerate(tokens):
            start = len(all_spm_tokens)
            if self.do_lower_case:
                token = token.lower()

            spm_tokens = self.tokenizer.tokenize(token)
            if remove_spm_prefix and idx > 0:
                spm_tokens = [tok.lstrip(SPM_UNDERLINE) for tok in spm_tokens]

            all_spm_tokens += [x for x in spm_tokens if len(x) > 0]
            end = len(all_spm_tokens) - 1
            spm_indices.append(Span(start, end))

        return all_spm_tokens, spm_indices

    def encode(self, query: Utterance, schema: DBSchema, matched_values: List[CellValue], lang: LanguageCode):
        # Encode query
        remove_spm_prefix = is_char_based(lang)
        input_tokens = [self.tokenizer.cls_token]
        input_tokens, query_indices = self._tokenize(query.text_tokens, remove_spm_prefix)

        query_indices.append(Span(start=len(input_tokens), end=len(input_tokens)))
        input_tokens += [self.tokenizer.sep_token]

        # Encode schema & value
        column2values, column_indices, value_indices = defaultdict(list), [], []
        for idx, value in enumerate(matched_values):
            column2values[value.column].append(idx)
            value_indices.append(None)

        entity_blocks = []
        for column in schema.columns:
            block_start = len(input_tokens)
            input_tokens += [self.special_tokens[column.data_type]] + self._tokenize([x.token for x in column.tokens], remove_spm_prefix)[0]
            block_median = len(input_tokens) - 1
            column_indices += [Span(block_start, block_median)]

            for v_idx in column2values.get(column.identifier, []):
                matched_value = matched_values[v_idx]
                val_start = len(input_tokens)
                input_tokens += [self.special_tokens[TableSpecialToken.Value]] + self._tokenize([x.token for x in matched_value.tokens], remove_spm_prefix)[0]
                val_end = len(input_tokens) - 1
                value_indices[v_idx] = Span(val_start, val_end)

            input_tokens += [self.tokenizer.sep_token]
            block_end = len(input_tokens) - 1
            entity_blocks.append((block_start, block_median, block_end))

        assert all(value_indices), "Invalid value indices: {}".format(query.text)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_tokens, input_ids, { 'question': query_indices, 'column': column_indices, 'value': value_indices }, entity_blocks
