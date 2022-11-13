from enum import Enum
from typing import Any
from contracts.base_types import *
from contracts.schemas import *


class SQLTokenType(int, Enum):
    Keyword = 0

    Table = 1
    Column = 2
    Value = 3
    Literal = 4

    Null = 5

    def __str__(self):
        return self.name

    @property
    def string(self):
        return str(self)

    @property
    def abbr(self):
        return ['keyword', 'tbl', 'col', 'val', 'literal', 'null'][int(self)]


class SQLFieldType(int, Enum):
    Select = 0
    From = 1
    GroupBy = 2
    Where = 3
    Having = 4
    Sort = 5


class QuestionLabel(int, Enum):
    Null = 0  # Others
    Keyword = 1

    Table = 2
    Column = 3
    Value = 4

    Ambiguity = 5
    Unknown = 6

    # Others = 7

    # LiteralString = 20
    # LiteralNumber = 21
    # LiteralDateTime = 22

    def __str__(self):
        return str(self.name)

    @classmethod
    def get_all_labels(cls):
        return [cls.Null, cls.Keyword, cls.Table, cls.Column, cls.Value, cls.Ambiguity, cls.Unknown]

    @classmethod
    def get_value(cls, text: str) -> int:
        for one in QuestionLabel.get_all_labels():
            if one.name == text:
                return one.value
        raise ValueError(f'unknown question label: {text}')

    @classmethod
    def get_by_value(cls, value: int):
        for one in QuestionLabel.get_all_labels():
            if one.value == value:
                return one
        raise ValueError(f'unknown question label value : {value}')

    @classmethod
    def get_abbr_by_value(cls, value: int):
        enum2abbr = {
            QuestionLabel.Null: 'O',
            QuestionLabel.Keyword: 'Key',
            QuestionLabel.Table: 'Tab',
            QuestionLabel.Column: 'Col',
            QuestionLabel.Value: 'Val',
            QuestionLabel.Ambiguity: 'Amb',
            QuestionLabel.Unknown: 'Unk',
        }
        for one in QuestionLabel.get_all_labels():
            if one.value == value:
                return enum2abbr[one]
        raise ValueError(f'unknown question label value : {value}')


class Aggregation(int, Enum):
    No = 0
    Max = 1
    Min = 2
    Sum = 3
    Avg = 4
    Count = 5

    def __str__(self):
        return str(self.name).upper()

    @classmethod
    def from_str(cls, agg: str):
        return {
            str(Aggregation.No).upper(): Aggregation.No,
            str(Aggregation.Max).upper(): Aggregation.Max,
            str(Aggregation.Min).upper(): Aggregation.Min,
            str(Aggregation.Sum).upper(): Aggregation.Sum,
            str(Aggregation.Avg).upper(): Aggregation.Avg,
            str(Aggregation.Count).upper(): Aggregation.Count
        }[agg.upper()]


class DatePartFunction(int, Enum):
    Date = 0
    Decade = 1
    Year = 2
    Quarter = 3
    Month = 4
    Day = 5
    Hour = 6
    Minute = 7
    HourAndMinute = 8
    HourMinuteAndSecond = 9
    DateAndHour = 10
    DateHourAndMinute = 11
    DateHourMinuteAndSecond = 12
    MonthAndDay = 13
    YearAndMonth = 14
    YearAndQuarter = 15

    def __str__(self):
        return str(self.name)  # .upper()

    pass


class LogicalOperator(int, Enum):
    No = 0
    And = 1
    Or = 2

    def __str__(self):
        return str(self.name).upper()


class ComparisonOperator(int, Enum):
    Eq = 0  # ==
    Ne = 1  # !=
    Gt = 2  # >
    Ge = 3  # >=
    Lt = 4  # <
    Le = 5  # <=

    def __str__(self):
        return ['=', '!=', '>', '>=', '<', '<='][int(self.value)]

    @classmethod
    def from_str(cls, op: str):
        return {
            str(ComparisonOperator.Eq): ComparisonOperator.Eq,
            str(ComparisonOperator.Ne): ComparisonOperator.Ne,
            str(ComparisonOperator.Gt): ComparisonOperator.Gt,
            str(ComparisonOperator.Ge): ComparisonOperator.Ge,
            str(ComparisonOperator.Lt): ComparisonOperator.Lt,
            str(ComparisonOperator.Le): ComparisonOperator.Le
        }[op]


class SQLOperator(int, Enum):
    Eq = 0  # ==
    Ne = 1  # !=
    Gt = 2  # >
    Ge = 3  # >=
    Lt = 4  # <
    Le = 5  # <=

    Startswith = 6  # startswith
    Endswith = 7  # Endswith
    Contains = 8  # contains

    def __str__(self):
        return ['=', '!=', '>', '>=', '<', '<=', 'startswith', 'endswith', 'contains'][int(self.value)]

    @classmethod
    def from_str(cls, op: str):
        return {
            str(ComparisonOperator.Eq): ComparisonOperator.Eq,
            str(ComparisonOperator.Ne): ComparisonOperator.Ne,
            str(ComparisonOperator.Gt): ComparisonOperator.Gt,
            str(ComparisonOperator.Ge): ComparisonOperator.Ge,
            str(ComparisonOperator.Lt): ComparisonOperator.Lt,
            str(ComparisonOperator.Le): ComparisonOperator.Le,
            str(ComparisonOperator.Startswith): ComparisonOperator.Startswith,
            str(ComparisonOperator.Endswith): ComparisonOperator.Endswith,
            str(ComparisonOperator.Contains): ComparisonOperator.Contains,
        }[op]


@dataclass
class SQLToken(JsonSerializable):
    type: SQLTokenType
    field: SQLFieldType
    value: str  # identifier of token value

    def __str__(self) -> str:
        items = self.value.split(Concept_Sep)
        if self.type == SQLTokenType.Value:
            return items[-1]

        if self.type in [SQLTokenType.Table, SQLTokenType.Column]:
            return f'[{self.value}]'

        return self.value

    @property
    def column(self) -> str:
        items = self.value.split(Concept_Sep)

        if self.type == SQLTokenType.Column:
            return self.value

        if self.type == SQLTokenType.Value:
            return Concept_Sep.join(items[:-1])

        # if self.type == SQLTokenType.Function:
        #     return Concept_Sep.join(items[1:])

        # if self.type == SQLTokenType.Operator:
        #     return Concept_Sep.join(items[1:-1])

        return None

    @property
    def cell_value(self) -> str:
        items = self.value.split(Concept_Sep)
        if self.type == SQLTokenType.Value:
            return self.value
        # if self.type == SQLTokenType.Operator:
        #     return Concept_Sep.join(items[1:])
        return None

    # @property
    # def func(self) -> Aggregation:
    #     items = self.value.split(Concept_Sep)
    #     if self.type == SQLTokenType.Function:
    #         return Aggregation.from_str(items[0])
    #     return None

    # @property
    # def op(self) -> ComparisonOperator:
    #     items = self.value.split(Concept_Sep)
    #     if self.type == SQLTokenType.Operator:
    #         return ComparisonOperator.from_str(items[0])
    #     return None

    @classmethod
    def format_func_value(column: str, func: str) -> str:
        return "{}{}{}".format(func, Concept_Sep, column)

    @classmethod
    def format_op_value(value: str, op: str) -> str:
        return "{}{}{}".format(op, Concept_Sep, value)

    @classmethod
    def from_json(cls, obj: Dict):
        obj['type'] = SQLTokenType(obj['type'])
        obj['field'] = SQLFieldType(obj['field'])
        return super().from_json(obj)


@dataclass
class SQLExpression(JsonSerializable):
    db_id: str
    tokens: List[SQLToken]
    sql_dict: Dict[str, object]

    def __str__(self) -> str:
        # query_str = ' '.join([str(x) for x in self.tokens])
        # query_str = query_str.replace("( ", "(").replace(" )", ")").replace(" ,", ",")
        # for agg in Aggregation:
        #     query_str = query_str.replace(f"{str(agg)} ", f"{str(agg)}")

        # for func in DatePartFunction:
        #     query_str = query_str.replace(f"{str(func)} ", f"{str(func)}")
        query_str_buff = []
        last_token = None
        for token in self.tokens:
            add_sep_flag = True
            if token.type == SQLTokenType.Keyword:
                if token.value in [')', ',']:
                    add_sep_flag = False

            if last_token is not None and last_token.type == SQLTokenType.Keyword:
                if last_token.value in ['(', 'eval']:
                    add_sep_flag = False

                for agg in Aggregation:
                    if str(agg).lower() == last_token.value.lower():
                        add_sep_flag = False

                for func in DatePartFunction:
                    if str(func).lower() == last_token.value.lower():
                        add_sep_flag = False

            if add_sep_flag and len(query_str_buff) > 0:
                query_str_buff.append(' ')

            query_str_buff.append(str(token))
            last_token = token

        return ''.join(query_str_buff)

    @classmethod
    def from_json(cls, obj: Dict):
        obj['tokens'] = [SQLToken.from_json(x) for x in obj['tokens']]
        return super().from_json(obj)


All_Agg_Op_Keywords = [
    str(Aggregation.Max),
    str(Aggregation.Min),
    str(Aggregation.Sum),
    str(Aggregation.Avg),
    str(Aggregation.Count),

    # str(ComparisonOperator.Eq),
    str(ComparisonOperator.Ne),
    str(ComparisonOperator.Gt),
    str(ComparisonOperator.Ge),
    str(ComparisonOperator.Lt),
    str(ComparisonOperator.Le)
]

All_Question_Labels = [
    str(QuestionLabel.Null),
    str(QuestionLabel.Keyword),
    str(QuestionLabel.Table),
    str(QuestionLabel.Column),
    str(QuestionLabel.Value),
    str(QuestionLabel.Ambiguity),
    str(QuestionLabel.Unknown)
]


def is_agg_or_op_keyword(token: SQLToken):
    if not token.type == SQLTokenType.Keyword:
        return False

    for keyword in All_Agg_Op_Keywords:
        if token.value == keyword:
            return True

    return False


@dataclass
class BindingItem:
    token: str
    term_type: str
    term_value: str
    confidence: float

    @classmethod
    def from_json(cls, obj: Dict):
        return BindingItem(**obj)

    def to_json(self) -> Dict:
        obj = super().to_json()
        obj['token'] = self.token
        obj['term_type'] = self.term_type
        obj['term_value'] = self.term_value
        obj['confidence'] = self.confidence
        return obj

    def __str__(self):
        if self.term_type == QuestionLabel.Null.name:
            return self.token

        return '[{}/{}/{:.3f}]'.format(self.token, self.term_value, self.confidence)


@dataclass
class Text2SQLExample(JsonSerializable):
    """
    Preprocessed Text-to-SQL example
    """
    dataset: str
    question: Utterance
    schema: DBSchema
    sql: SQLExpression

    matched_values: List[CellValue]  # retrieved value candidates
    erased_ngrams: List[Span]  # question ngrams for erasing
    binding_result: List[BindingItem]

    uuid: str
    value_resolved: bool

    @classmethod
    def from_json(cls, obj: Dict):
        obj['question'] = Utterance.from_json(obj['question'])
        obj['schema'] = DBSchema.from_json(obj['schema'])
        obj['sql'] = SQLExpression.from_json(obj['sql'])
        obj['matched_values'] = [CellValue.from_json(x) for x in obj['matched_values']] if obj[
                                                                                               'matched_values'] is not None else None
        obj['erased_ngrams'] = [Span.from_json(x) for x in obj['erased_ngrams']] if obj[
                                                                                        'erased_ngrams'] is not None else None
        obj['value_resolved'] = obj['value_resolved']

        # def get_question_labels(binding_result_list):
        #     labels = []
        #     for item in binding_result_list:
        #         labels.append(QuestionLabel.get_value(item['term_type']))
        #     return labels

        obj['binding_result'] = [BindingItem.from_json(x) for x in obj['binding_result']] if obj[
                                                                                                 'binding_result'] is not None else None

        return super().from_json(obj)

    def to_json(self) -> Dict:
        js = {}
        js['dataset'] = self.dataset
        js['question'] = self.question.to_json()
        js['schema'] = self.schema.to_json()
        js['sql'] = self.sql.to_json()
        js['matched_values'] = [one.to_json() for one in self.matched_values]
        js['erased_ngrams'] = [one.to_json() for one in self.erased_ngrams]
        js['language'] = str(self.language)
        js['uuid'] = self.uuid
        return js

    @property
    def language(self) -> LanguageCode:
        dataset_prefix = self.dataset.lower().split("__")[0]
        return {
            'annatalk_en': LanguageCode.en,
            'annatalk_en_label': LanguageCode.en,
            "annatalk_zh": LanguageCode.zh,
            "annatalk_es": LanguageCode.es,
            "wikitq": LanguageCode.en,
            "wikisql": LanguageCode.en,
            'zhuiyi': LanguageCode.zh,
            'cbank': LanguageCode.zh,
            'formula': LanguageCode.zh
        }[dataset_prefix]

    def ignore_unmatched_values(self):
        self.matched_values = [v for v in self.matched_values if v.is_from_match]
        return self

    @property
    def resolved(self) -> bool:
        return self.value_resolved

    def get_question_labels(self) -> List[int]:
        question_labels = []
        for item in self.binding_result:
            label_value = QuestionLabel.get_value(str(item.term_type))
            question_labels.append(label_value)
        return question_labels

    def get_concept_labels(self) -> Dict[SQLTokenType, List[int]]:
        column_labels, value_labels, keyword_labels = [], [], []

        for column in self.schema.columns:
            col_tokens = [sql_token for sql_token in self.sql.tokens if
                          sql_token.type == SQLTokenType.Column and sql_token.column == column.identifier]
            column_labels += [len(col_tokens)]

        for value in self.matched_values:
            val_tokens = [sql_token for sql_token in self.sql.tokens if
                          sql_token.type == SQLTokenType.Value and self.are_equal_values(sql_token, value)]
            value_labels += [len(val_tokens)]

        for keyword in All_Agg_Op_Keywords:
            valid_tokens = [sql_token for sql_token in self.sql.tokens if
                            sql_token.type == SQLTokenType.Keyword and keyword.lower() == sql_token.value.lower()]
            keyword_labels += [len(valid_tokens)]

        return {SQLTokenType.Column: column_labels, SQLTokenType.Value: value_labels,
                SQLTokenType.Keyword: keyword_labels}

    def are_equal_values(self, sql_value: SQLToken, matched_value: CellValue) -> bool:
        matched_value_str = matched_value.name
        return sql_value.cell_value.replace('"', '').replace(" ", "").lower() == (
            Concept_Sep.join([matched_value.column, matched_value_str])).replace(" ", "").lower()

    def get_value_indices(self, column: Any) -> List[int]:
        indices = []
        assert isinstance(column, int) or isinstance(column, str)
        column_identifier = column if isinstance(column, str) else self.schema.columns[int(column)].identifier
        for vi, cell_value in enumerate(self.matched_values):
            if cell_value.column == column_identifier:
                indices += [vi]
        return indices

    def get_grounding_concepts(self) -> List[Concept]:
        concepts = []
        for op in All_Agg_Op_Keywords:
            concepts += [Keyword(keyword=op.upper())]

        for column in self.schema.columns:
            concepts += [column]

        for value in self.matched_values:
            concepts += [value]

        return concepts


@dataclass
class SchemaLinkingItem(JsonSerializable):
    span: Span
    entity: SQLToken

    @classmethod
    def from_json(cls, obj: Dict):
        obj['span'] = Span.from_json(obj['span'])
        obj['entity'] = SQLToken.from_json(obj['entity'])
        return super().from_json(obj)


@dataclass
class ChaseExample(Text2SQLExample):
    schema_linkings: List[SchemaLinkingItem]

    @property
    def resolved(self) -> bool:
        return True

    @classmethod
    def from_json(cls, obj: Dict):
        obj['schema_linkings'] = [SchemaLinkingItem.from_json(x) for x in obj['schema_linkings']]
        return super().from_json(obj)

    def get_concept_labels(self):
        column_labels = []
        for column in self.schema.columns:
            col_tokens = [x.entity for x in self.schema_linkings if
                          x.entity.type == SQLTokenType.Column and x.entity.value == column.identifier]
            fields = set([token.field for token in col_tokens])
            column_labels += [{'fields': fields, 'funcs': []}]

        table_labels = []
        for table in self.schema.tables:
            tbl_tokens = [x.entity for x in self.schema_linkings if
                          x.entity.type == SQLTokenType.Table and x.entity.value == table.identifier]
            fields = set([token.field for token in tbl_tokens])
            table_labels += [len(fields)]

        keyword_labels = [0] * (len(Aggregation) - 1 + len(ComparisonOperator) - 1)
        for token in self.sql.tokens:
            if token.type == SQLTokenType.Function:
                agg = int(Aggregation.from_str(token.value)) - 1
                if agg >= 0:
                    keyword_labels[agg] = 1

            if token.type == SQLTokenType.Operator:
                op = int(ComparisonOperator.from_str(token.value)) - 1
                if op >= 0:
                    keyword_labels[op + len(Aggregation) - 1] = 1

        return {SQLTokenType.Column: column_labels, SQLTokenType.Value: [], SQLTokenType.Table: table_labels,
                SQLTokenType.Keyword: keyword_labels}
