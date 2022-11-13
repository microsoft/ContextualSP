"""
Encode Text-Table pair into one sequence as input of pretrained language model
"""
from collections import defaultdict
import enum
from typing import List, Tuple, Dict

from transformers import PreTrainedTokenizer

from .bind_types import (
    LanguageCode,
    NLToken,
    NLColumn,
    NLMatchedValue,
    NLDataType
)

Data_Type_Tokens = {
    NLDataType.String: '[text]',
    NLDataType.Integer: '[number]',
    NLDataType.Double: '[number]',
    NLDataType.DateTime: '[datetime]',
    NLDataType.Boolean: '[boolean]'
}

class TableSpecialToken(str, enum.Enum):
    Unknown = '<unk>'
    Table = '<tab>'
    Column = '<col>'
    Value = '<val>'
    Empty = '<empty>'

SPM_UNDERLINE = '▁'

class SpmPreprocessor:
    def __init__(self, path: str, sep: str="") -> None:
        self.sep = sep
        self.prefix_dict = self.load_prefix_dict(path)

    @staticmethod
    def load_prefix_dict(path: str):
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
        return query in self.prefix_dict and self.prefix_dict[query]

    def _preprocess_tokens(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        processed_tokens, idx_mappings = [], []
        idx = 0
        while idx < len(tokens):
            phrase_tokens = [tokens[idx]]

            next_idx, max_idx = idx, idx
            while self.prefix_match(phrase_tokens):
                if self.extract_match(phrase_tokens):
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

        return processed_tokens, idx_mappings

    def preprocess_nl_tokens(self, nl_tokens: List[NLToken]) -> Tuple[List[NLToken], List[int]]:
        _, idx_mappings = self._preprocess_tokens([x.token for x in nl_tokens])
        processed_tokens: List[NLToken] = []
        for i, token in enumerate(nl_tokens):
            if i > 0 and idx_mappings[i-1] == idx_mappings[i]:
                processed_tokens[-1].token += token.token
                processed_tokens[-1].lemma += token.lemma
                continue

            processed_tokens.append(token.clone())

        return processed_tokens, idx_mappings

class NLDataEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizer, spm_phrase_path: str, do_lower_case: bool = True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.spm_processor = SpmPreprocessor(path=spm_phrase_path)
        self.do_lower_case = do_lower_case

    @property
    def special_tokens(self) -> Dict[object, str]:
        tokenizer_class_name = self.tokenizer.__class__.__name__.lower()
        if tokenizer_class_name.startswith('xlm') or tokenizer_class_name.startswith('roberta'):
            return {
                NLDataType.String : '<text>',
                NLDataType.Integer : '<number>',
                NLDataType.Double : '<number>',
                NLDataType.DateTime : '<date>',
                NLDataType.Boolean : '<bool>',
                TableSpecialToken.Column: '<col>',
                TableSpecialToken.Value: '<val>',
                TableSpecialToken.Empty: '<empty>',
            }

        raise NotImplementedError(tokenizer_class_name)

    def _tokenize(self, token: NLToken):
        input_text = token.token if isinstance(token, NLToken) else str(token)
        if self.do_lower_case:
            input_text = input_text.lower()
        return self.tokenizer.tokenize(input_text)

    def tokenize_tokens(self, tokens: List[NLToken], lang: LanguageCode):
        is_char_based = lang.is_char_based()
        if is_char_based:
            norm_tokens, idx_mappings = self.spm_processor.preprocess_nl_tokens(tokens)
        else:
            norm_tokens, idx_mappings = tokens, list(range(len(tokens)))

        all_spm_tokens, spm_indices = [], []
        for idx, token in enumerate(norm_tokens):
            start = len(all_spm_tokens)
            spm_tokens = self._tokenize(token)
            if is_char_based and idx > 0:
                spm_tokens = [tok.lstrip(SPM_UNDERLINE) for tok in spm_tokens]

            all_spm_tokens += [x for x in spm_tokens if len(x) > 0]
            end = len(all_spm_tokens) - 1
            spm_indices.append((start, end))

        return all_spm_tokens, spm_indices, idx_mappings

    def encode(self, query_tokens: List[NLToken], columns: List[NLColumn], matched_values: List[NLMatchedValue], lang: LanguageCode):
        # Encode query
        query_spm_tokens, query_indices, spm_idx_mappings = self.tokenize_tokens(query_tokens, lang)

        input_tokens = [self.tokenizer.cls_token] + query_spm_tokens
        query_indices = [(s+1, e+1) for (s, e) in query_indices]

        # Add a pad token for question sequence
        query_indices.append((len(input_tokens), len(input_tokens)))
        input_tokens += [self.tokenizer.sep_token]

        # Encode schema & value
        column2values, column_indices, value_indices = defaultdict(list), [], []
        for idx, value in enumerate(matched_values):
            column2values[value.column].append(idx)
            value_indices.append(None)

        for column in columns:
            column_start = len(input_tokens)
            input_tokens += [self.special_tokens[column.data_type]] + self.tokenize_tokens(column.tokens, lang)[0]
            column_indices += [(column_start, len(input_tokens)-1)]

            for v_idx in column2values.get(column.name, []):
                matched_value = matched_values[v_idx]
                value_start = len(input_tokens)
                input_tokens += [self.special_tokens[TableSpecialToken.Value]] + self.tokenize_tokens(matched_value.tokens, lang)[0]
                value_indices[v_idx] = (value_start, len(input_tokens) - 1)

            input_tokens += [self.tokenizer.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_tokens, input_ids, spm_idx_mappings, { 'question': query_indices, 'column': column_indices, 'value': value_indices }
