from strongsup.predicate import Predicate
from strongsup.tables.executor import is_unary, is_binary, ALL_BUILT_INS
from strongsup.tables.graph import ALL_GRAPH_BUILT_INS
from strongsup.utils import EOU


class WikiTablePredicate(Predicate):
    def __init__(self, name, original_string=None):
        types = self._compute_types(name)
        super(WikiTablePredicate, self).__init__(name, original_string, types=types)

    def _compute_types(self, name):
        """Get the types (and a few features) of a predicate.

        Args:
            name (unicode): name of the predicate
        Return:
            tuple[string]
        """
        types = []
        if is_unary(name):
            types.append(WikiTablePredicateType.UNARY)
        if is_binary(name):
            types.append(WikiTablePredicateType.BINARY)
        if name in FIXED_PREDICATE_NAMES:
            types.append(WikiTablePredicateType.BUILTIN)
        if name.startswith('fb:cell.') and not name.startswith('fb:cell.cell.'):
            types.append(WikiTablePredicateType.CELL)
        elif name.startswith('fb:part.'):
            types.append(WikiTablePredicateType.PART)
        elif name.startswith('fb:row.row.'):
            types.append(WikiTablePredicateType.COLUMN)
        elif name.startswith('!fb:row.row.'):
            types.append(WikiTablePredicateType.RCOLUMN)
        elif name.startswith('N'):
            types.append(WikiTablePredicateType.NUMBER)
        elif name.startswith('D'):
            types.append(WikiTablePredicateType.DATE)
        return tuple(types)

    @property
    def types_vector(self):
        """Return the types as a k-hot vector.

        Returns:
            list[boolean]
        """
        return [x in self.types for x in WikiTablePredicateType.ALL_TYPES]

    @property
    def words(self):
        """Get the words from the ID.

        Returns:
            list[unicode]
        """
        return self.name.split('.')[-1].split('_')

    @property
    def delexicalized_name(self):
        """A placeholder used in a delexicalized utterance.

        Returns:
            unicode
        """
        if WikiTablePredicateType.COLUMN in self.types:
            return 'COL'
        if WikiTablePredicateType.CELL in self.types:
            return 'ENT'
        return None


class WikiTablePredicateType(object):
    UNARY = 'unary'
    BINARY = 'binary'
    BUILTIN = 'builtin'
    CELL = 'cell'
    PART = 'part'
    COLUMN = 'column'
    RCOLUMN = '!column'
    NUMBER = 'number'
    DATE = 'date'
    ALL_TYPES = (UNARY, BINARY, BUILTIN, CELL, PART, COLUMN, RCOLUMN, NUMBER, DATE)

    @classmethod
    def is_relation(cls, pred):
        return (WikiTablePredicateType.BINARY in pred.types) and not cls.is_builtin(pred)

    @classmethod
    def is_entity(cls, pred):
        return WikiTablePredicateType.UNARY in pred.types

    @classmethod
    def is_builtin(cls, pred):
        return WikiTablePredicateType.BUILTIN in pred.types


FIXED_PREDICATE_NAMES = (EOU,) + ALL_BUILT_INS + ALL_GRAPH_BUILT_INS
FIXED_PREDICATES = [WikiTablePredicate(name) for name in FIXED_PREDICATE_NAMES]
