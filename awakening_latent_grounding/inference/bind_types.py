"""
NLBinding model interface contracts
"""
from copy import Error
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Dict

class LanguageCode(str, Enum):
    en = "en"
    zh = "zh"
    es = "es"
    ja = "ja"
    de = "de"
    fr = "fr"

    def is_char_based(self) -> bool:
        return self in [LanguageCode.zh, LanguageCode.ja]

    def to_json(self) -> Any:
        return self.value

    @classmethod
    def from_json(cls, obj):
        return LanguageCode(obj)

    def __str__(self) -> str:
        return str(self.value)

class NLDataType(int, Enum):
    String = 0
    DateTime = 1
    Integer = 2
    Double = 3
    Boolean = 4

    def to_json(self) -> int:
        return self.value

    @classmethod
    def from_json(cls, obj):
        return NLDataType(obj)

    def __str__(self) -> str:
        return str(self.name)

@dataclass
class NLToken:
    token: str
    lemma: str

    def to_json(self) -> Dict:
        return {
            'token': self.token,
            'lemma': self.lemma
        }

    @classmethod
    def from_json(cls, obj: Dict):
        return NLToken(**obj)

    def __str__(self) -> str:
        return self.token

    def clone(self):
        return NLToken(
            token=self.token,
            lemma=self.lemma
        )

@dataclass
class NLColumn:
    name: str
    data_type: NLDataType
    tokens: List[NLToken]

    def to_json(self) -> Dict:
        return {
            'name': self.name,
            'data_type': self.data_type.to_json(),
            'tokens': [x.to_json() for x in self.tokens]
        }

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, obj: Dict):
        if 'header_tokens' in obj:
            obj['tokens'] = obj['header_tokens']
            obj.pop('header_tokens', None)

        obj['data_type'] = NLDataType.from_json(obj['data_type'])
        obj['tokens'] = [NLToken.from_json(x) for x in obj['tokens']]
        return NLColumn(**obj)

@dataclass
class NLMatchedValue:
    column: str
    name: str # Value Text
    tokens: List[NLToken] # Value Tokens
    start: int = -1
    end: int = -1

    def to_json(self) -> Dict:
        return {
            'column': self.column,
            'name': self.name,
            'tokens': [x.to_json() for x in self.tokens],
            'start': self.start,
            'end': self.end
        }

    @classmethod
    def from_json(cls, obj: Dict):
        if 'value' in obj:
            obj['name'] = obj['value']
            obj.pop('value', None)

        if 'value_tokens' in obj:
            obj['tokens'] = obj['value_tokens']
            obj.pop('value_tokens', None)

        obj['tokens'] = [NLToken.from_json(x) for x in obj['tokens']]
        return NLMatchedValue(**obj)

    def __str__(self) -> str:
        return "{}/{}".format(self.name, self.column)

class NLBindingType(int, Enum):
    Null = 0
    Keyword = 1

    Table = 1 << 1
    Column = 1 << 2
    Value = 1 << 3

    LiteralString = 1 << 5
    LiteralNumber = 1 << 6
    LiteralDateTime = 1 << 7

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, obj):
        return NLBindingType(obj)

    def __str__(self) -> str:
        return str(self.name)

@dataclass
class NLBindingTermResult:
    term_type: NLBindingType
    term_index: int
    term_value: str
    term_score: float # term prediction score
    grounding_scores: List[float] # term grounding scores

    def to_json(self) -> Dict:
        return {
            'term_type': self.term_type.to_json(),
            'term_index': self.term_index,
            'term_value': self.term_value,
            'term_score': self.term_score,
            'grounding_scores': self.grounding_scores
        }

    @classmethod
    def from_json(cls, obj):
        obj['term_type'] = NLBindingType(obj['term_type'])
        return NLBindingTermResult(**obj)

    def to_string(self, question_tokens: List[NLToken]) -> str:
        str_result = "{}({}): {:.3f}".format(self.term_value, str(self.term_type), self.term_score)
        str_result += "; Grounding Scores: " + " ".join(["{}/{:.3f}".format(
            token.token, score) for token, score in zip(question_tokens, self.grounding_scores)]
        )
        return str_result

    def clone(self):
        return NLBindingTermResult(
            term_type=self.term_type,
            term_index=self.term_index,
            term_value=self.term_value,
            term_score=self.term_score,
            grounding_scores=self.grounding_scores
        )

@dataclass
class NLBindingToken(NLToken):
    term_type: NLBindingType
    term_index: int
    term_value: str
    confidence: float

    @classmethod
    def from_json(cls, obj: Dict):
        obj['term_type'] = NLBindingType(obj['term_type'])
        return NLBindingToken(**obj)

    def to_json(self) -> Dict:
        obj = super().to_json()
        obj['term_type'] = self.term_type.to_json()
        obj['term_index'] = self.term_index
        obj['term_value'] = self.term_value
        obj['confidence'] = self.confidence
        return obj

    def __str__(self):
        if self.term_type == NLBindingType.Null:
            return self.token

        return '[{}/{}/{:.3f}]'.format(self.token, self.term_value, self.confidence)

@dataclass
class NLBindingRequest:
    question_tokens: List[NLToken]
    columns: List[NLColumn]
    matched_values: List[NLMatchedValue]
    language: LanguageCode = LanguageCode.en

    def to_string(self, sep='\n'):
        logs = []
        logs.append("{} Question: {}".format(str(self.language), " ".join([x.token for x in self.question_tokens])))
        logs.append("Columns: {}".format(" || ".join([x.name for x in self.columns])))
        return sep.join(logs)

    def to_json(self):
        return {
            'question_tokens': [x.to_json() for x in self.question_tokens],
            'columns': [x.to_json() for x in self.columns],
            'matched_values': [x.to_json() for x in self.matched_values],
            'language': self.language.to_json(),
        }

    @classmethod
    def from_json(cls, obj):
        obj['question_tokens'] = [NLToken.from_json(x) for x in obj['question_tokens']]
        obj['columns'] = [NLColumn.from_json(x) for x in obj['columns']]
        obj['matched_values'] = [NLMatchedValue.from_json(x) for x in obj['matched_values']]

        if 'language' in obj:
            obj['language'] = LanguageCode.from_json(obj['language'])
        return NLBindingRequest(**obj)

class StatusCode(int, Enum):
    succeed = 0
    timeout = 1
    invalid_input = 2
    invalid_output = 3
    model_error = 4

    internal_error = 10
    not_implemented = 20

@dataclass
class NLModelError(Error):
    error_code: StatusCode
    message: str = None

@dataclass
class AnnaTalkBaseResult:
    status_code: StatusCode # status code
    inference_ms: int # model inference elapsed milliseconds
    message: str # error message if status code is not 0

    def to_json(self) -> Dict:
        return {
            'status_code': self.status_code.value,
            'inference_ms': self.inference_ms,
            'message': self.message
        }

    @classmethod
    def from_json(cls, obj: Dict):
        obj['status_code'] = StatusCode(obj['status_code'])
        return AnnaTalkBaseResult(**obj)

@dataclass
class NLBindingResult(AnnaTalkBaseResult):
    term_results: List[NLBindingTermResult]
    binding_tokens: List[NLBindingToken]

    def to_json(self):
        return {
            'status_code': self.status_code.value,
            'inference_ms': self.inference_ms,
            'message': self.message,

            'term_results': [x.to_json() for x in self.term_results],
            'binding_tokens': [x.to_json() for x in self.binding_tokens] if self.binding_tokens is not None else None,
        }

    def to_string(self, question_tokens: List[NLToken], sep="\n"):
        logs = []
        for term_result in self.term_results:
            logs.append(term_result.to_string(question_tokens))

        logs.append("NL Binding Sequence: {}".format(" ".join([str(x) for x in self.binding_tokens])))

        return sep.join(logs)
    
    def export_binding_json(self):
        binding_js_list = []

        for item in self.binding_tokens:
            js = {}
            js['token'] = item.token
            js['term_type'] = item.term_type.name
            js['term_value'] = item.term_value
            js['confidence'] = item.confidence
            binding_js_list.append(js)

        return binding_js_list

    @classmethod
    def from_json(cls, obj):
        if 'status_code' not in obj:
            obj['status_code'] = 0
            obj['message'] = None

        obj['status_code'] = StatusCode(obj['status_code'])
        obj['term_results'] = [NLBindingTermResult.from_json(x) for x in obj['term_results']]
        obj['binding_tokens'] = [NLBindingToken.from_json(x) for x in obj['binding_tokens']] if obj['binding_tokens'] is not None else None
        return NLBindingResult(**obj)
