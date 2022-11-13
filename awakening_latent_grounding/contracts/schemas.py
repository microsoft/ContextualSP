from enum import Enum
from contracts.base_types import *

Row_Column_Str = '*' # Special column
Concept_Sep = '::'

class DataType(int, Enum):
    Text = 0
    Number = 1
    DateTime = 2
    Boolean = 3
    def __str__(self) -> str:
        return ['text', 'number', 'datetime', 'bool'][int(self)]

@dataclass
class Column(Concept):
    data_type: DataType
    table: str = field(default=None) # table identifier

    @property
    def identifier(self) -> str:
        if self.table is not None:
            return "{}{}{}".format(self.table, Concept_Sep, self.name)
        return self.name

    def __str__(self) -> str:
        if self.table is not None:
            return "{}.{}".format(self.table, self.name)
        return self.name

    def to_json(self) -> Dict:
        obj = super().to_json()
        if self.table is None:
            obj.pop('table', None)
        return obj

    @classmethod
    def from_json(cls, obj: Dict):
        obj['data_type'] = DataType(obj['data_type'])
        return super().from_json(obj)

@dataclass
class Table(Concept):
    columns: List[str] # column identifier names

@dataclass
class CellValue(Concept):
    span: Span
    column: str # column identifier
    score: float # matched score

    @property
    def identifier(self) -> str:
        return "{}{}{}".format(self.column, Concept_Sep, self.name)

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def end(self) -> int:
        return self.span.end

    @classmethod
    def from_json(cls, obj: Dict):
        obj['span'] = Span.from_json(obj['span'])
        return super().from_json(obj)

    @property
    def is_from_match(self) -> bool:
        return self.start >= 0 and self.end >= 0 or self.name.lower() in ['true', 'false']

@dataclass
class DBSchema(JsonSerializable):
    db_id: str
    columns: List[Column]
    tables: List[Table] = field(default=None)

    primary_keys: List[str] = field(default=None)
    foreign_keys: List[Tuple[str, str]] = field(default=None)

    identifier_map: Dict[str, object] = field(init=False)

    def __post_init__(self):
        id_map = {}

        if self.tables is not None:
            for table in self.tables:
                if table.identifier in id_map:
                    # logging.info("Ignore duplicated table identifier: {}({})".format(table.identifier, self.db_id))
                    continue

        for column in self.columns:
            if column.identifier in id_map:
                # logging.info("Ignore duplicated column identifier: {}({})".format(column.identifier, self.db_id))
                continue
            id_map[column.identifier] = column

        self.identifier_map = id_map

    @property
    def num_tables(self) -> int:
        return len(self.tables) if self.tables is not None else 0

    @property
    def num_columns(self) -> int:
        return len(self.columns) if self.columns is not None else 0

    @property
    def is_single_table(self):
        return self.tables is None

    def lookup_column(self, identifier: str) -> Column:
        assert identifier in self.identifier_map
        column: Column = self.identifier_map[identifier]
        assert isinstance(column, Column)
        return column

    def lookup_table(self, identifier: str) -> Table:
        assert identifier in self.identifier_map
        table: Table = self.identifier_map[identifier]
        assert isinstance(table, Table)
        return table

    def to_json(self) -> Dict:
        obj = super().to_json()
        if self.primary_keys is None:
            obj.pop('primary_keys', None)
        if self.foreign_keys is None:
            obj.pop('foreign_keys', None)

        obj.pop('identifier_map', None)
        return obj

    def is_primary_key(self, col_identifier: str) -> bool:
        return self.primary_keys is not None and col_identifier in self.primary_keys

    def is_foreign_key(self, col_identifier: str) -> bool:
        if self.foreign_keys is None:
            return False

        for c1, c2 in self.foreign_keys:
            if col_identifier == c1 or col_identifier == c2:
                return True

        return False

    def to_string(self):
        column_strs = [f"{col.identifier}/{str(col.data_type)}" for col in self.columns]
        return "{}: ".format(self.db_id) + " || ".join(column_strs)

    @classmethod
    def from_json(cls, obj: Dict):
        obj['columns'] = [Column.from_json(x) for x in obj['columns']]
        obj['tables'] = [Table.from_json(x) for x in obj['tables']] if obj['tables'] is not None else None
        return super().from_json(obj)