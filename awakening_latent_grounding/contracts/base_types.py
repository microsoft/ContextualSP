import json
from enum import Enum
from abc import ABCMeta
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

@dataclass
class JsonSerializable(metaclass=ABCMeta):
    def to_json(self) -> Dict:
        return self._parse_json_obj({key: val for key, val in self.__dict__.items() if not key.startswith("_")})

    @staticmethod
    def _parse_json_obj(obj: object) -> Dict:
        if isinstance(obj, list):
            return [JsonSerializable._parse_json_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {key:JsonSerializable._parse_json_obj(val) for key, val in obj.items()}

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, JsonSerializable):
            return obj.to_json()

        return obj

    @classmethod
    def from_json(cls, obj: Dict):
        if isinstance(obj, cls):
            return obj
        return cls(**obj)

def save_json_objects(objects: List, path: str, ensure_ascii=True):
    with open(path, 'w', encoding='utf-8') as fw:
        if isinstance(objects, list):
            fw.write('[\n')
            for idx, obj in enumerate(objects):
                if isinstance(obj, JsonSerializable):
                    obj = obj.to_json()
                if idx == len(objects) - 1:
                    fw.write(json.dumps(obj, ensure_ascii=ensure_ascii) + "\n")
                else:
                    fw.write(json.dumps(obj, ensure_ascii=ensure_ascii) + ",\n")
            fw.write(']\n')
        elif isinstance(objects, dict):
            objects = {key: val.to_json() if isinstance(val, JsonSerializable) else val for key, val in objects.items() }
            json.dump(objects, fw)
        else:
            raise NotImplementedError()

def load_json_objects(obj_class, path: str):
    json_objects = json.load(open(path, 'r', encoding='utf-8'))
    if isinstance(json_objects, list):
        objects = []
        for json_obj in json_objects:
            objects += [obj_class.from_json(json_obj)]
        return objects
    elif isinstance(json_objects, dict):
        objects = {}
        for key, val in json_objects.items():
            objects[key] = obj_class.from_json(val)
        return objects
    else:
        raise NotImplementedError()

class LanguageCode(str, Enum):
    en = "en"
    zh = "zh"
    es = "es"
    ja = "ja"
    de = "de"
    fr = "fr"

    def is_char_based(self) -> bool:
        return self in [LanguageCode.zh, LanguageCode.ja]

    def to_json(self) -> str:
        return self.value

    @classmethod
    def from_json(cls, obj):
        return LanguageCode(obj)

    def __str__(self) -> str:
        return self.value


@dataclass
class Token(JsonSerializable):
    token: str # original value
    lemma: str # lemma value

    def __str__(self):
        return self.token

@dataclass(order=False, frozen=True)
class Span(JsonSerializable):
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start + 1

    def add(self, offset:int):
        return Span(start=self.start+offset, end=self.end+offset)

@dataclass
class Utterance(JsonSerializable):
    text: str # utterance text
    tokens: List[Token] # tokens

    @classmethod
    def from_json(cls, obj: Dict):
        obj['tokens'] = [Token.from_json(x) for x in obj['tokens']]
        return super().from_json(obj)

    @property
    def text_tokens(self):
        return [x.token for x in self.tokens]

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

@dataclass
class Concept(JsonSerializable):
    name: str # Displayed name
    tokens: List[Token] # Tokens

    @classmethod
    def from_json(cls, obj: Dict):
        obj['tokens'] = [Token.from_json(x) for x in obj['tokens']]
        return super().from_json(obj)

    def __str__(self) -> str:
        return self.identifier

    @property
    def identifier(self) -> str: # Unique id of given Concept
        return self.name

    def __hash__(self) -> int:
        return self.identifier.__hash__()

    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        return other.identifier == self.identifier

class DependentConcept(Concept):
    base_concept: Concept
    def __init__(self, base_concept: Concept, name: str) -> None:
        self.base_concept = base_concept
        self.name = name
        self.tokens = None

    @property
    def identifier(self) -> str:
        return "{}({})".format(self.name, self.base_concept.identifier)

class Keyword(Concept):
    def __init__(self, keyword: str) -> None:
        self.name = keyword
        self.tokens = None
