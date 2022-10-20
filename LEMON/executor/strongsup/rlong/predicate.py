from strongsup.predicate import Predicate


class RLongPredicate(Predicate):
    """Predicates for the RLong domain.

    Conventions:
    - colors are single characters (y, g, ...)
    - numbers are integers, positive or negative (1, -2, ...)
    - fractions start with X (X1/2, X2/3, ...)
    - properties start with P (PColor, PHatColor, ...)
    - actions start with A (ADrain, AMove, ...)
    - built-in predicates include:
        all-objects, index, argmin, argmax
    - history slots start with H (H0, H1, ...)
    """
    CACHE = {}

    def __new__(cls, name, original_string=None):
        if name not in cls.CACHE:
            types = cls._compute_types(name)
            # pred = super(RLongPredicate, cls).__new__(
            #         cls, name, original_string=original_string, types=types)
            pred = super(RLongPredicate, cls).__new__(cls)
            cls.CACHE[name] = pred
        return cls.CACHE[name]

    @classmethod
    def _compute_types(cls, name):
        assert isinstance(name, str)
        types = []
        if len(name) == 1 and name[0].isalpha():
            types.append(RLongPredicateType.COLOR)
        elif name[0] == '-' or name[0].isdigit():
            types.append(RLongPredicateType.NUMBER)
        elif name[0] == 'X':
            types.append(RLongPredicateType.FRACTION)
        elif name[0] == 'P':
            types.append(RLongPredicateType.PROPERTY)
        elif name[0] == 'D':
            types.append(RLongPredicateType.DOUBLE_PROPERTY)
        elif name[0] == 'A':
            types.append(RLongPredicateType.ACTION)
        elif name in BUILTIN_NAMES:
            types.append(RLongPredicateType.BUILTIN)
        elif name[0] == 'H':
            types.append(RLongPredicateType.HISTORY_SLOT)
        else:
            raise ValueError('Unknown predicate: {}'.format(name))
        return tuple(types)

    @property
    def types_vector(self):
        """Return the types as a k-hot vector.

        Returns:
            list[boolean]
        """
        return [x in self.types for x in RLongPredicateType.ALL_TYPES]


BUILTIN_NAMES = ['all-objects', 'index', 'argmin', 'argmax']

class RLongPredicateType(object):
    COLOR = 'color'
    NUMBER = 'number'
    FRACTION = 'fraction'
    PROPERTY = 'property'
    DOUBLE_PROPERTY = 'double_property'
    ACTION = 'action'
    BUILTIN = 'builtin'
    HISTORY_SLOT = 'history_slot'
    ALL_TYPES = (COLOR, NUMBER, FRACTION, PROPERTY, 
            DOUBLE_PROPERTY, ACTION, BUILTIN, HISTORY_SLOT)
