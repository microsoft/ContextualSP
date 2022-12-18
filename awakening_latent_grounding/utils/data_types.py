from enum import Enum
import re
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from transformers import BertTokenizer

"""
Constant values
"""
SOS_Token = '<sos>'
EOS_Token = '<eos>'
UNK_Token = '<unk>'
TBL_Token = '<tbl>'
VAL_Token = '<val>'

Tbl_Col_Sep = '[TC_SEP]'
Col_Val_Sep = '[CV_SEP]'

DB_Col_Keys = ['[Null_Key]', '[Primary_Key]', '[Foreign_Key]', '[PF_Key]']

Bert_Special_Tokens = {
    TBL_Token: '[unused10]',
    '*': '[unused15]',
    'text': '[unused21]',
    'number': '[unused22]',
    'time': '[unused23]',
    'boolean': '[unused24]',
    'real': '[unused25]',
    'integer': '[unused26]',
    Tbl_Col_Sep: '[unused30]',
    DB_Col_Keys[0]: '[unused40]',
    DB_Col_Keys[1]: '[unused41]',
    DB_Col_Keys[2]: '[unused42]',
    DB_Col_Keys[3]: '[unused43]',
}

Max_Decoding_Steps = 100


@dataclass(order=False, frozen=True)
class Token:
    index: int  # token index in utterance
    token: str  # token original value
    lemma: str  # lemmatise value
    pieces: List[str]  # bert pieces

    def to_json(self) -> Dict:
        return self.__dict__

    @staticmethod
    def from_json(obj: Dict):
        return Token(**obj)

    def __str__(self):
        return self.token


@dataclass
class Utterance:
    text: str
    tokens: List[Token]

    pieces: List[str] = field(init=False)
    token2pieces: List[Tuple[int, int]] = field(init=False)
    piece2token: List[int] = field(init=False)

    def __post_init__(self):
        pieces, token2pieces, piece2token = [], [], []
        for i, token in enumerate(self.tokens):
            n_pieces = len(token.pieces)
            token2pieces += [(len(pieces), len(pieces) + n_pieces - 1)]
            pieces += token.pieces
            piece2token += [i] * n_pieces

        self.pieces = pieces
        self.token2pieces = token2pieces
        self.piece2token = piece2token

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def text_tokens(self) -> List[str]:
        return [token.token for token in self.tokens]

    def to_json(self) -> Dict:
        return {
            'text': self.text,
            'tokens': [x.to_json() for x in self.tokens]
        }

    @classmethod
    def from_json(cls, obj: Dict):
        return Utterance(
            text=obj['text'],
            tokens=[Token.from_json(x) for x in obj['tokens']] if obj['tokens'] is not None else None
        )

    def get_token2pieces(self):
        token2pieces = []
        count = 0
        for i, token in enumerate(self.tokens):
            token2pieces += [(count, count + len(token.pieces) - 1)]
            count += len(token.pieces)
        return token2pieces

    def get_piece2token(self):
        piece2token = []
        for tok_idx, token in enumerate(self.tokens):
            for piece in token.pieces:
                piece2token.append(tok_idx)
        return piece2token

    def get_pieces(self):
        pieces = []
        for token in self.tokens:
            pieces += token.pieces
        return pieces


@dataclass
class STSchema:
    """
    Single-table schema
    """
    table_id: str
    column_names: List[str]
    column_types: List[str]

    id_map: Dict[str, int] = field(init=False)  # column names to index

    def __post_init__(self):
        assert len(self.column_names) == len(self.column_types)

        id_map = {}
        for name in self.column_names:
            assert name not in id_map
            id_map[name] = len(id_map)
        self.id_map = id_map

    @property
    def num_columns(self):
        return len(self.column_names)

    def to_json(self) -> Dict:
        return {
            'table_id': self.table_id,
            'column_names': self.column_names,
            'column_types': self.column_types
        }

    @classmethod
    def from_json(cls, obj: Dict):
        return STSchema(**obj)

    def to_string(self):
        column_with_types = ["{}/{}".format(c, t) for c, t in zip(self.column_names, self.column_types)]
        return "{}:\t{}".format(self.table_id, " || ".join(column_with_types))


@dataclass
class WTQSchema:
    table_id: str
    column_headers: List[str]
    column_names_internal: List[str]  # Internal column name used in WikiTableQuestion to generate SQL
    column_types_internal: List[str]
    internal_to_header: List[int]

    header_to_internals: Dict[int, List[int]] = field(init=False)
    column_header_to_id: Dict[str, int] = field(init=False)
    internal_name_to_id: Dict[str, int] = field(init=False)
    column_id_to_suffix_types: Dict[int, List[str]] = field(init=False)

    def __post_init__(self):
        header_to_internals = defaultdict(list)
        for internal_id, header_id in enumerate(self.internal_to_header):
            header_to_internals[header_id].append(internal_id)
        assert len(header_to_internals) == len(self.column_headers)
        self.header_to_internals = header_to_internals

        column_header_to_id = {}
        for idx, name in enumerate(self.column_headers):
            if name in column_header_to_id:
                continue
            column_header_to_id[name] = idx
        self.column_header_to_id = column_header_to_id

        internal_name_to_id = {}
        for idx, name in enumerate(self.column_names_internal):
            if name in internal_name_to_id:
                continue
            internal_name_to_id[name] = idx
        self.internal_name_to_id = internal_name_to_id

        column_id_to_suffix_types = {}
        for idx in range(len(self.column_headers)):
            suffix_types = set([])
            for internal_id in self.header_to_internals[idx]:
                suffix_types.add(self.get_suffix_type(self.column_names_internal[internal_id]))
            column_id_to_suffix_types[idx] = list(suffix_types)

        self.column_id_to_suffix_types = column_id_to_suffix_types

    @staticmethod
    def get_suffix_type(internal_name: str) -> str:
        if not re.match('^c\d', internal_name):
            return ''
        return re.sub('^c\d+', '', internal_name).strip()

    def lookup_header_id(self, column_name: str):
        header_id = self.column_header_to_id[column_name]
        return header_id

    def lookup_header_id_from_internal(self, internal_name: str):
        internal_id = self.internal_name_to_id[internal_name]
        header_id = self.internal_to_header[internal_id]
        return header_id

    def lookup_header_and_suffix(self, internal_name: str):
        header_id = self.lookup_header_id_from_internal(internal_name)
        return self.column_headers[header_id], self.get_suffix_type(internal_name)

    def to_json(self):
        return {
            'table_id': self.table_id,
            'column_headers': self.column_headers,
            'column_names_internal': self.column_names_internal,
            'column_types_internal': self.column_types_internal,
            'internal_to_header': self.internal_to_header,
        }

    @classmethod
    def from_json(cls, obj: Dict):
        return WTQSchema(**obj)

    def to_string(self):
        out_strs = []
        for _, header in enumerate(self.column_headers):
            out_strs.append(header)
        return "{}: {}".format(self.table_id, " || ".join(out_strs))


@dataclass
class SpiderSchema:
    db_id: str
    column_names: List[str]
    column_types: List[str]
    column_names_lemma: List[str]
    column_names_original: List[str]

    table_names: List[str]
    table_names_lemma: List[str]
    table_names_original: List[str]

    table_to_columns: Dict[int, List[int]]
    column_to_table: Dict[int, int]

    primary_keys: List[int]
    foreign_keys: List[Tuple[int, int]]

    id_map: Dict[str, int] = field(init=False)  # column full name & table name to index

    def __post_init__(self):
        self.id_map = self._build()

    def _build(self):
        idMap = {}
        for i, _ in enumerate(self.column_names_original):
            idMap[self.get_column_full_name(i)] = i
        for i, tab in enumerate(self.table_names_original):
            key = tab.lower()
            idMap[key] = i
        return idMap

    @property
    def num_tables(self) -> int:
        return len(self.table_names_original)

    @property
    def num_columns(self) -> int:
        return len(self.column_names_original)

    def build_column2ids(self) -> Dict[str, int]:
        col2ids = defaultdict(list)
        for c_idx, c_name in enumerate(self.column_names):
            col2ids[c_name.lower()].append(c_idx)
        return col2ids

    @classmethod
    def from_json(cls, obj: Dict):
        table_to_columns = {}
        for i, ids in obj['table_to_columns'].items():
            table_to_columns[int(i)] = ids
        obj['table_to_columns'] = table_to_columns

        column_to_table = {}
        for c_idx, t_idx in obj['column_to_table'].items():
            column_to_table[int(c_idx)] = int(t_idx)
        obj['column_to_table'] = column_to_table
        obj.pop('id_map', None)
        return SpiderSchema(**obj)

    def to_json(self) -> Dict:
        return self.__dict__

    @property
    def schema(self):
        tables = defaultdict(list)
        for tbl_idx, tbl_name in enumerate(self.table_names_original):
            for col_idx in self.table_to_columns[tbl_idx]:
                tables[tbl_name.lower()].append(self.column_names_original[col_idx].lower())
        return tables

    @property
    def idMap(self):
        return self.id_map

    def get_column_full_name(self, column_idx: int) -> str:
        if self.column_names_original[column_idx] == '*':
            return '*'
        table_name = self.table_names_original[self.column_to_table[column_idx]]
        return '{}.{}'.format(table_name, self.column_names_original[column_idx]).lower()

    def get_col_identifier_name(self, index: int) -> str:
        if self.column_names_original[index] == '*':
            return '*'
        table_name = self.table_names_original[self.column_to_table[index]]
        return '{}.{}'.format(table_name, self.column_names_original[index]).lower()

    def get_tbl_identifier_name(self, index: int) -> str:
        return self.table_names_original[index].lower()

    def get_identifier_name(self, type: str, index: int) -> str:
        if type in ['tbl', 'table']:
            return self.get_tbl_identifier_name(index)
        elif type in ['col', 'column']:
            return self.get_col_identifier_name(index)
        else:
            raise NotImplementedError()

    def get_column_key_code(self, column_idx: int) -> int:
        key = 0
        if column_idx in self.primary_keys:
            key |= 1

        for c1, c2 in self.foreign_keys:
            if column_idx in [c1, c2]:
                key |= 2

        return key

    def to_string(self, sep: str = '\n'):
        schema_strs = ["db id: {}".format(self.db_id)]
        for tbl_id, col_ids in self.table_to_columns.items():
            if tbl_id == -1:
                continue
            tbl_name = self.table_names[tbl_id]
            col_strs = []
            for col_id in col_ids:
                col_str = '{}/{}'.format(self.column_names[col_id], self.column_types[col_id])
                if col_id in self.primary_keys:
                    col_str += '(PK)'
                col_strs += [col_str]
            schema_strs.append("{}: {}".format(tbl_name, " || ".join(col_strs)))

        fk_strs = []
        for c_idx1, c_idx2 in self.foreign_keys:
            fk_strs.append(
                '{}::{} - {}::{}'.format(self.table_names[self.column_to_table[c_idx1]], self.column_names[c_idx1],
                                         self.table_names[self.column_to_table[c_idx2]], self.column_names[c_idx2]))
        schema_strs.append("FKs: {}".format(" || ".join(fk_strs)))
        return sep.join(schema_strs)

    def get_name(self, e_type: str, e_id: int):
        if e_type == 'tbl':
            return self.table_names[e_id]
        elif e_type == 'col':
            tbl_id = self.column_to_table[e_id]
            return '{}[{}]'.format(self.column_names[e_id], self.table_names[tbl_id])
        else:
            col_name = self.get_name('col', e_id)
            return 'val_{}'.format(col_name)

    def get_table_with_columns(self) -> Dict[str, List[str]]:
        tables = defaultdict(list)
        for tbl_idx, tbl_name in enumerate(self.table_names_original):
            for col_idx in self.table_to_columns[tbl_idx]:
                tables[tbl_name].append(self.column_names_original[col_idx])
        return tables


class SQLTokenType(int, Enum):
    null = 0
    keyword = 1
    table = 2
    column = 3
    value = 4

    def __str__(self):
        return self.name

    @property
    def abbr(self):
        return ['null', 'keyword', 'tbl', 'col', 'val'][int(self)]


class SQLFieldType(int, Enum):
    Select = 0
    From = 1
    GroupBy = 2
    Where = 3
    Having = 4
    Sort = 5


class SQLToken:
    token_type: SQLTokenType
    value: str

    def __init__(self, token_type: SQLTokenType, value: str) -> None:
        self.token_type = token_type
        self.value = value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SQLToken):
            return False

        return other.token_type == self.token_type and other.value == self.value

    def to_json(self) -> Dict:
        return {
            'token_type': self.token_type,
            'value': self.value,
        }

    @classmethod
    def from_json(cls, obj: Dict):
        token_type = SQLTokenType(obj['token_type'])
        if token_type == SQLTokenType.keyword:
            return KeywordToken.from_json(obj)
        elif token_type == SQLTokenType.table:
            return TableToken.from_json(obj)
        elif token_type == SQLTokenType.column:
            return ColumnToken.from_json(obj)
        elif token_type == SQLTokenType.value:
            return ValueToken.from_json(obj)
        elif token_type == SQLTokenType.null:
            return SQLToken(SQLTokenType.null, None)
        else:
            raise NotImplementedError("Not supported type: {}".format(token_type))


class KeywordToken(SQLToken):
    def __init__(self, keyword: str) -> None:
        super().__init__(SQLTokenType.keyword, keyword)

    @property
    def keyword(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KeywordToken) and self.keyword == other.keyword

    @classmethod
    def from_json(cls, obj: Dict):
        return KeywordToken(keyword=obj['value'])


class TableToken(SQLToken):
    def __init__(self, table_name: str = None) -> None:
        super().__init__(SQLTokenType.table, table_name)

    def __str__(self):
        if not self.table_name:
            return 'T'
        return self.table_name

    @property
    def table_name(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TableToken) and self.table_name == other.table_name

    @classmethod
    def from_json(cls, obj: Dict):
        return TableToken(table_name=obj['value'])


class ColumnToken(SQLToken):
    suffix_type: str  # Used for WTQ

    def __init__(self, column_header: str, suffix_type: str) -> None:
        super().__init__(SQLTokenType.column, column_header)
        self.suffix_type = suffix_type

    @property
    def column_name(self):
        return self.value

    def __str__(self):
        return self.column_name + self.suffix_type

    def __eq__(self, other: object) -> bool:
        return isinstance(other,
                          ColumnToken) and other.column_name == self.column_name and other.suffix_type == self.suffix_type

    def to_json(self) -> Dict:
        return {
            'token_type': self.token_type,
            'value': self.value,
            'suffix_type': self.suffix_type
        }

    @classmethod
    def from_json(cls, obj: Dict):
        return ColumnToken(column_header=obj['value'], suffix_type=obj['suffix_type'])


class ValueToken(SQLToken):
    columns: List[str]
    span: Tuple[int, int]

    def __init__(self, value: str, span: Tuple[int, int] = None, columns: List[str] = None) -> None:
        assert value is not None or span is not None
        super().__init__(SQLTokenType.value, value)
        self.span = span
        self.columns = columns

    def __str__(self):
        if self.span is None:
            if isinstance(self.value, str):
                return self.value
            return str(self.value)

        return "{}[{}:{}]".format(self.value, self.span[0], self.span[1])

    def to_json(self):
        return {
            'token_type': self.token_type,
            'value': self.value,
            'columns': self.columns,
            'span': self.span
        }

    def __eq__(self, other: object) -> bool:
        if self.span is not None:
            return isinstance(other, ValueToken) and self.span == other.span
        return isinstance(other, ValueToken) and self.value == other.value

    def __ne__(self, other: object) -> bool:
        return not other == self

    @property
    def start(self) -> int:
        return self.span[0]

    @property
    def end(self) -> int:
        return self.span[1]

    @classmethod
    def from_json(cls, obj: Dict):
        return ValueToken(value=obj['value'], span=obj['span'], columns=obj['columns'])


@dataclass(frozen=True)
class SQLExpression:
    tokens: List[SQLToken]
    db_id: str = field(default=None)

    @property
    def sql(self):
        return " ".join([str(term) for term in self.tokens]).replace('\n', '\\n')

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return self.sql

    def __repr__(self) -> str:
        return self.sql

    def to_json(self) -> Dict:
        return {'tokens': [x.to_json() for x in self.tokens]}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SQLExpression) or len(self.tokens) != len(other.tokens):
            return False

        for i in range(len(self.tokens)):
            if not self.tokens[i] == other.tokens[i]:
                return False

        return True

    @classmethod
    def from_json(cls, obj: Dict):
        return SQLExpression(tokens=[SQLToken.from_json(x) for x in obj['tokens']])


@dataclass
class SQLTokenHypothesis:
    tokens: List[SQLToken]
    scores: List[float]
    total_score: float

    def is_finished(self):
        last_token = self.tokens[-1]
        return last_token.token_type == SQLTokenType.keyword and last_token.value == EOS_Token

    def __len__(self):
        return len(self.tokens)

    @property
    def num_steps(self):
        return len(self.tokens)

    def update(self, token: SQLToken, score: float):
        return SQLTokenHypothesis(tokens=self.tokens + [token], scores=self.scores + [score],
                                  total_score=self.total_score + score)

    def to_sql(self):
        if self.is_finished:
            return SQLExpression(tokens=self.tokens[1:-1])
        else:
            return SQLExpression(tokens=self.tokens[1:])

    @classmethod
    def from_token(cls, token: SQLToken):
        return SQLTokenHypothesis(tokens=[token], scores=[0.0], total_score=0.0)

    @classmethod
    def from_sos(cls):
        return SQLTokenHypothesis(tokens=[KeywordToken(SOS_Token)], scores=[0], total_score=0)


class AlignmentLabel:
    token: Token
    align_type: SQLTokenType
    align_value: str
    confidence: float = 1.0

    def __init__(self, token: Token, align_type: SQLTokenType, align_value: str, confidence: float = 1.0):
        self.token = token
        self.align_type = align_type
        self.align_value = align_value
        self.confidence = confidence

    def to_json(self) -> Dict:
        return {'token': self.token.to_json(), 'align_type': self.align_type, 'align_value': self.align_value,
                'confidence': self.confidence}

    @classmethod
    def from_jon(cls, obj: Dict):
        return AlignmentLabel(
            token=Token.from_json(obj['token']),
            align_type=SQLTokenType(obj['align_type']),
            align_value=obj['align_value'],
            confidence=obj['confidence']
        )

    def to_slsql(self, schema: SpiderSchema) -> Dict:
        if self.align_type == SQLTokenType.null:
            return None
        elif self.align_type == SQLTokenType.table:
            tbl_id = schema.id_map[self.align_value]
            return {'type': 'tbl', 'id': tbl_id, 'token': self.token.token, 'value': self.align_value}
        elif self.align_type == SQLTokenType.column:
            col_id = schema.id_map[self.align_value]
            return {'type': 'col', 'id': col_id, 'token': self.token.token, 'value': self.align_value}
        elif self.align_type == SQLTokenType.value:
            column_name = self.align_value.replace("VAL_", "")
            col_id = schema.id_map[column_name]
            return {'type': 'val', 'id': col_id, 'token': self.token.token, 'value': self.align_value}
        else:
            raise NotImplementedError()

    def __str__(self):
        if self.align_type == SQLTokenType.null:
            return self.token.token
        return "{}/{}/{:.3f}".format(self.token.token, self.align_value, self.confidence)

    def __eq__(self, value):
        assert isinstance(value, AlignmentLabel)
        return self.token.index == value.token.index and self.align_type == value.align_type and self.align_value == value.align_value


class SchemaRelation(int, Enum):
    null = 0

    table_table = 1
    table_column = 2
    column_table = 3
    column_column = 4

    column_value = 5
    value_column = 6

    table_column_pk = 7
    column_table_pk = 8
    column_column_fk_fw = 9
    column_column_fk_bw = 10


def save_json_objects(objects: List, path: str):
    with open(path, 'w', encoding='utf-8') as fw:
        fw.write('[\n')
        for idx, obj in enumerate(objects):
            if idx == len(objects) - 1:
                fw.write(json.dumps(obj) + "\n")
            else:
                fw.write(json.dumps(obj) + ",\n")
        fw.write(']\n')


def generate_utterance(tokenizer: BertTokenizer, text: str, tokens: List[str] = None,
                       lemma: List[str] = None) -> Utterance:
    assert text is not None or tokens is not None
    if text is None:
        text = " ".join(tokens)
    if tokens is None:
        tokens = text.split()
    if lemma is None:
        lemma = [x.lower() for x in tokens]

    assert len(tokens) == len(lemma)
    new_tokens = []
    for i, (tok, lem) in enumerate(zip(tokens, lemma)):
        pieces = tokenizer.tokenize(lem)
        token = Token(index=i, token=tok, lemma=lem, pieces=pieces)
        new_tokens += [token]
    return Utterance(text=text, tokens=new_tokens)
