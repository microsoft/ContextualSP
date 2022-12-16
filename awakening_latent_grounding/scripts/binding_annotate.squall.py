# %%
from collections import defaultdict
import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict
import pandas as pd


# %%
class NLBindingType(int, Enum):
    Null = 0
    Table = 1  # table in DB
    Column = 2  # table column header
    Value = 3  # 1) Cell value in given table 2) Number|DateTime|String, which may not exist in given table

    Function = 4  # Aggregation, such as MAX, MIN, AVG, ...
    Operator = 5  # Comparsion operator, such as ==, >, <=, starts_with, contains...

    def __str__(self) -> str:
        return ['', 'Tbl', 'Col', 'Val', 'Func', 'Op'][self.value]


@dataclass
class NLBindingToken:
    text: str  # original text of binding token
    type: NLBindingType  # binding type
    value: str  # unique value, which means we can use 'type' and 'value' to find a unique entity (table/column/value, ...)

    def __str__(self) -> str:
        if self.type == NLBindingType.Null:
            return self.text

        return "{}/[{}::{}]".format(self.text, str(self.type), self.value)


@dataclass
class NLBindingExample:
    unique_id: str
    table_id: str
    question: str
    binding_tokens: List[NLBindingToken]
    tag: str = field(default="")

    def to_json(self) -> Dict:
        pass

    @property
    def question_tokens(self) -> List[str]:
        return [x.text for x in self.binding_tokens]

    @property
    def serialized_string(self):
        items = []
        items.append(self.unique_id)
        # items.append(self.table_id)
        items.append(" ".join([str(x) for x in self.binding_tokens]))
        return '\t'.join(items)


# %%
keywords = defaultdict(set)


def parse_squall_align_token(token: str, align_label: str, align_value: object) -> NLBindingToken:
    if align_label == 'None':
        return NLBindingToken(text=token, type=NLBindingType.Null, value=None)

    if align_label == 'Column':
        assert isinstance(align_value, str), align_value
        return NLBindingToken(text=token, type=NLBindingType.Column, value=align_value)

    if align_label == 'Keyword':
        assert isinstance(align_value, list), align_value
        keywords[align_value[0]].add(align_value[1])
        return NLBindingToken(text=token, type=NLBindingType.Function, value="_".join(align_value))

    if align_label == 'Literal':
        return NLBindingToken(text=token, type=NLBindingType.Value, value=token)

    raise NotImplementedError()


def load_squall_data(path: str):
    raw_examples = json.load(open(path, 'r', encoding='utf-8'))
    print('load {} examples from {} over.'.format(len(raw_examples), path))

    binding_examples = []
    for raw_example in raw_examples:
        question_tokens = raw_example['nl']
        assert len(question_tokens) == len(raw_example['nl_ralign'])
        binding_tokens = []
        for i, (align_label, align_value) in enumerate(raw_example['nl_ralign']):
            binding_tokens += [parse_squall_align_token(question_tokens[i], align_label, align_value)]

        binding_example = NLBindingExample(
            unique_id="WTQ_Squall__{}".format(raw_example['nt']),
            table_id="WTQ_Squall_{}".format(raw_example['tbl']),
            question=" ".join(question_tokens),
            binding_tokens=binding_tokens
        )

        binding_examples += [binding_example]

    df = pd.DataFrame(
        data=[[ex.unique_id, ex.table_id, " ".join([str(x) for x in ex.binding_tokens])] for ex in binding_examples],
        columns=['id', 'table_id', 'binding_tokens'],
    )

    return binding_examples, df


dev_examples, dev_df = load_squall_data(r'../data/squall/dev-0.json')
# %%
for key, val in keywords.items():
    print(key, val)
# %%
dev_df.to_csv(r'../data/squall/dev.binding.csv', index=False, decimal='\t')
# %%
