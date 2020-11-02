from typing import Dict, Set
from context.db_context import SparcDBContext
from context.utils import Table, TableColumn

Keywords = ['limit', 'des', 'asc', 'and', 'or', 'sum', 'min', 'max', 'avg', 'none', '=', '!=', '<', '>', '<=', '>=',
            'between', 'like', 'not_like', 'in', 'not_in', 'intersect', 'union', 'except', 'none', 'count', 'ins']


class GrammarType:
    """
    Filter Grammar Type
    """
    FilterBetween = 1
    FilterEqual = 2
    FilterGreater = 3
    FilterLess = 4
    FilterGeq = 5
    FilterLeq = 6
    FilterNeq = 7
    FilterInNes = 8
    FilterNotInNes = 9
    FilterLike = 10
    FilterNotLike = 11
    FilterIs = 12
    FilterExist = 13

    # TODO: in and like does not have a nested version
    FilterNotNes = 14
    FilterBetweenNes = 15
    FilterEqualNes = 16
    FilterGreaterNes = 17
    FilterLessNes = 18
    FilterGeqNes = 19
    FilterLeqNes = 20
    FilterNeqNes = 21
    FilterIsNes = 22
    FilterExistNes = 23

    FilterAnd = 24
    FilterOr = 25
    # FilterNone = 26

    """
    Statement Grammar Type
    """
    StateInter = 1
    StateUnion = 2
    StateExcept = 3
    StateNone = 4

    """
    Root Grammar Type
    """
    RootSFO = 1
    RootSO = 2
    RootSF = 3
    RootS = 4

    """
    Select Grammar Type depends on the length of A
    """

    """
    A Grammar Type
    """
    ANone = 1
    AMax = 2
    AMin = 3
    ACount = 4
    ASum = 5
    AAvg = 6

    """
    Order Grammar Type
    """
    OrderNone = 1
    OrderAsc = 2
    OrderDes = 3
    OrderAscLim = 4
    OrderDesLim = 5


class Grammar(object):
    # static property, production rule to id
    productions = None

    def __init__(self, db_context: SparcDBContext):
        self._pro_counter = 0
        self._type_counter = 0

        # lazy loading, init the production
        if self.productions is None:
            # new self.productions
            self.productions = []

            # C and T only contain one rule so they do not need initialization
            self.build_production_map(Statement)
            self.build_production_map(Root)
            self.build_production_map(Select)
            self.build_production_map(A)
            self.build_production_map(Filter)
            self.build_production_map(Order)

        self.db_context = db_context
        self.local_grammar = self.build_instance_production()

    def build_production_map(self, cls):
        """
        Record the production rules of class cls into self
        :param cls: son class of Action
        """
        # (note) the values could provide a fixed order
        # only when the dictionary is built on
        prod_ids = cls.grammar_dict.keys()
        for prod_id in prod_ids:
            cls_obj = cls(prod_id)
            self.productions.append(cls_obj)

    def build_instance_production(self):
        """
        Instance all possible column and table production rules using db schema
        """
        db_schema: Dict[str, Table] = self.db_context.schema

        # fetch table name(id)
        table_names = sorted([db_schema[table_ind].name for table_ind in
                              list(db_schema.keys())], reverse=True)

        local_grammars = [T(table_name) for table_name in table_names]

        all_columns = set()
        for table in db_schema.values():
            # use name(id) as standard grammar
            all_columns.update([C(column.name) for column in table.columns])
        column_grammars = list(all_columns)
        local_grammars.extend(column_grammars)
        # convert into set and sorted
        local_grammars = set(local_grammars)
        # sorted local grammars
        local_grammars = sorted(local_grammars)
        return local_grammars

    @property
    def global_grammar(self):
        return sorted(self.productions)

    @staticmethod
    def default_sql_clause() -> Dict:
        default_sql = {
            "orderBy": [],
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        1
                    ]
                ],
                "conds": []
            },
            "union": None,
            "except": None,
            "groupBy": None,
            "limit": None,
            "intersect": None,
            "where": [],
            "having": [],
            "select": [
                False,
                [
                    [
                        3,
                        [
                            0,
                            [
                                0,
                                5,
                                False
                            ],
                            None
                        ]
                    ]
                ]
            ]
        }
        return default_sql


class Action(object):
    grammar_dict = {}

    def __init__(self):
        self.ins_id = None
        self.production = None

    def get_next_action(self, is_sketch=False):
        actions = list()
        for x in self.production.split(' ')[1:]:
            if x not in Keywords:
                rule_type = eval(x)
                if is_sketch:
                    if rule_type is not A and rule_type is not T:
                        actions.append(rule_type)
                else:
                    actions.append(rule_type)
        return actions

    def __repr__(self):
        space_ind = self.production.find(' ')
        return f'{self.production[:space_ind]} -> {self.production[space_ind + 1:]}'

    def is_global(self):
        """
        Actions are global means they fit for the whole dataset, while others only
        fit for specific instances
        :return:
        """
        if self.__class__ in [C, T]:
            return False
        else:
            return True

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    @staticmethod
    def from_str(action_repr: str):
        """
        Build an action object from string
        :param action_repr: the representation of action
        :return: Action object
        """
        lhs, rhs = action_repr.split(' -> ')
        # eval class object
        cls_obj = eval(lhs)
        if cls_obj in [C, T]:
            return cls_obj(rhs)
        else:
            # find the rule id
            rule_str = ' '.join([lhs, rhs])
            grammar_dict: Dict = cls_obj.grammar_dict
            rule_id = list(grammar_dict.keys())[list(grammar_dict.values()).index(rule_str)]
            return cls_obj(rule_id)


class Statement(Action):
    grammar_dict = {
        GrammarType.StateInter: 'Statement intersect Root Root',
        GrammarType.StateUnion: 'Statement union Root Root',
        GrammarType.StateExcept: 'Statement except Root Root',
        GrammarType.StateNone: 'Statement Root'
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Root(Action):
    grammar_dict = {
        GrammarType.RootSFO: 'Root Select Filter Order',
        GrammarType.RootSF: 'Root Select Filter',
        GrammarType.RootSO: 'Root Select Order',
        GrammarType.RootS: 'Root Select'
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Select(Action):
    grammar_dict = {
        0: 'Select A',
        1: 'Select A A',
        2: 'Select A A A',
        3: 'Select A A A A',
        4: 'Select A A A A A',
        5: 'Select A A A A A A'
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class A(Action):
    grammar_dict = {
        GrammarType.ANone: 'A none C T',
        GrammarType.AMax: 'A max C T',
        GrammarType.AMin: 'A min C T',
        GrammarType.ACount: 'A count C T',
        GrammarType.ASum: 'A sum C T',
        GrammarType.AAvg: 'A avg C T'
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Filter(Action):
    # TODO: why not directly predict the number of Filters
    grammar_dict = {
        GrammarType.FilterAnd: 'Filter Filter and Filter',
        GrammarType.FilterOr: 'Filter Filter or Filter',

        GrammarType.FilterEqual: 'Filter = A',
        GrammarType.FilterGreater: 'Filter > A',
        GrammarType.FilterLess: 'Filter < A',
        GrammarType.FilterGeq: 'Filter >= A',
        GrammarType.FilterLeq: 'Filter <= A',
        GrammarType.FilterNeq: 'Filter != A',
        GrammarType.FilterBetween: 'Filter between A',
        # TODO: like/not_like only apply to string type
        GrammarType.FilterLike: 'Filter like A',
        GrammarType.FilterNotLike: 'Filter not_like A',

        GrammarType.FilterEqualNes: 'Filter = A Root',
        GrammarType.FilterGreaterNes: 'Filter > A Root',
        GrammarType.FilterLessNes: 'Filter < A Root',
        GrammarType.FilterGeqNes: 'Filter >= A Root',
        GrammarType.FilterLeqNes: 'Filter <= A Root',
        GrammarType.FilterNeqNes: 'Filter != A Root',
        GrammarType.FilterBetweenNes: 'Filter between A Root',
        GrammarType.FilterInNes: 'Filter in A Root',
        GrammarType.FilterNotInNes: 'Filter not_in A Root',
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Order(Action):
    grammar_dict = {
        GrammarType.OrderAsc: 'Order asc A',
        GrammarType.OrderDes: 'Order des A',
        GrammarType.OrderAscLim: 'Order asc A limit',
        GrammarType.OrderDesLim: 'Order des A limit'
    }

    def __init__(self, ins_id):
        super().__init__()
        self.ins_id = ins_id
        self.production = self.grammar_dict[ins_id]


# class Ref(Action):
#
#     grammar_dict = {
#         GrammarType.RefStar: 'Ref *',
#         GrammarType.RefCol: 'Ref C',
#     }
#
#     def __init__(self, ins_id):
#         super().__init__()
#         self.ins_id = ins_id
#         self.production = self.grammar_dict[ins_id]


class C(Action):
    def __init__(self, ins_id):
        super().__init__()
        # TODO: here we lower it because the col -> id (entities_names) in SparcWorld is the lower key-value pair.
        self.ins_id = ins_id.lower()
        self.production = f'C {self.ins_id}'


class T(Action):
    def __init__(self, ins_id):
        super().__init__()
        self.ins_id = ins_id.lower()
        self.production = f'T {self.ins_id}'


# TODO: consider copy value from source sentence
# class V(Action):
#     def __init__(self, id_c):
#         super().__init__()
#         self.id_c = id_c
#         self.production = 'V ins'


if __name__ == '__main__':
    order_rule = Order(GrammarType.OrderDesLim)
    assert order_rule.production == 'Order des A limit'
    assert str(order_rule) == 'Order -> des A limit'

    sel_rule = Select(1)
    assert sel_rule.production == 'Select A A'
    assert str(sel_rule) == 'Select -> A A'

    col_rule = C('sales')
    assert col_rule.production == 'C sales'
    assert str(col_rule) == 'C -> sales'
