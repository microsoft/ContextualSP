from gtd.utils import cached_property

from strongsup.executor import Executor, Denotation
from strongsup.predicate import Predicate
from strongsup.utils import EOU
from strongsup.value import Value

from strongsup.tables.structure import (
        parse_number,
        parse_date,
        Date,
        ensure_same_type,
        InfiniteSet,
        NeqInfiniteSet,
        RangeInfiniteSet,
        GenericDateInfiniteSet,
        )
from strongsup.tables.graph import TablesKnowledgeGraph
from strongsup.tables.value import StringValue, NumberValue, DateValue


################################
# Naming Conventions

NUMBER_PREFIX = 'N'
DATE_PREFIX = 'D'
NAME_PREFIX = 'fb:'
REVERSED_NAME_PREFIX = '!fb:'
ASSERT_PREFIX = 'assert-'

TYPE_ROW = 'type-row'
SPECIAL_BINARIES = ('!=', '<', '>', '<=', '>=')
AGGREGATES = ('count', 'min', 'max', 'sum', 'avg')
MERGES = ('and', 'or', 'diff')
BEGIN_GROWS = ('x',)
END_GROWS = ('argmin', 'argmax')

ALL_BUILT_INS = ((TYPE_ROW,)
        + SPECIAL_BINARIES + AGGREGATES + MERGES + BEGIN_GROWS + END_GROWS)

def is_unary_name(x):
    return x.startswith(NAME_PREFIX) and x.count('.') == 1

def is_unary(x):
    return (x[0] in (NUMBER_PREFIX, DATE_PREFIX) or
            (x.startswith(NAME_PREFIX) and x.count('.') == 1))

def parse_unary(x):
    """Return the correct unary object if x represents a unary.
    Otherwise, return None."""
    if is_unary_name(x):
        return x
    elif x.startswith(NUMBER_PREFIX):
        return parse_number(x[len(NUMBER_PREFIX):])
    elif x.startswith(DATE_PREFIX):
        return parse_date(x[len(DATE_PREFIX):])
    return None

def is_binary_name(x):
    return x.startswith(NAME_PREFIX) and x.count('.') == 2

def is_reversed_binary_name(x):
    return x.startswith(REVERSED_NAME_PREFIX) and x.count('.') == 2

def is_binary(x):
    return (x in SPECIAL_BINARIES or
            ((x.startswith(NAME_PREFIX) or x.startswith(REVERSED_NAME_PREFIX))
                and x.count('.') == 2))


################################
# Helper Decorators

def handle_dict_1arg(fn):
    """Decorator to support a 1-argument operation on dict"""
    def wrapped_fn(self, predicate, arg):
        if isinstance(arg, dict):
            answer = {}
            for key, things in arg.items():
                answer[key] = fn(self, predicate, things)
            return answer
        else:
            return fn(self, predicate, arg)
    wrapped_fn.original_fn = fn
    return wrapped_fn


def handle_dict_2args(fn):
    """Decorator to support a 2-argument operation on dict(s)"""
    def wrapped_fn(self, predicate, arg1, arg2):
        if isinstance(arg1, dict) or isinstance(arg2, dict):
            answer = {}
            if not isinstance(arg1, dict):
                for key, things in arg2.items():
                    answer[key] = fn(self, predicate, arg1, things)
            elif not isinstance(arg2, dict):
                for key, things in arg1.items():
                    answer[key] = fn(self, predicate, things, arg2)
            else:
                # Both are dicts
                for key in set(arg1) | set(arg2):
                    answer[key] = fn(self, predicate,
                            arg1.get(key, set()), arg2.get(key, set()))
            return answer
        else:
            return fn(self, predicate, arg1, arg2)
    wrapped_fn.original_fn = fn
    return wrapped_fn


################################
# Denotation

class TablesDenotation(list, Denotation):
    """A TablesDenotation is a stack of objects.
    Each object is either a set (unary) or a dict with sets as values (binary).
    See strongsup.tables.structure docstring for more details.

    For convenience during execution, TablesDenotation is mutable.
    """
    def __init__(self, *args):
        list.__init__(self, *args)
        if len(args) == 1 and isinstance(args[0], TablesDenotation):
            self._utterance_idx = args[0]._utterance_idx
        else:
            self._utterance_idx = 0

    @property
    def utterance_idx(self):
        return self._utterance_idx

    def increment_utterance_idx(self):
        self._utterance_idx += 1


################################
# Executor

class TablesPostfixExecutor(Executor):
    """Stack-based executor for the tables domain.

    Executes a postfix-encoded logical form on the table knowledge graph.
    """
    CACHE_LIMIT = 20000

    def __init__(self, graph, debug=False, forbid_partial_empty=True):
        """Construct a new executor.

        Args:
            graph (TablesKnowledgeGraph): graph to be executed on.
            debug (bool): whether to be verbose.
            forbid_partial_empty (bool): throw an error if any step produces
                an empty denotation. (True by default)
        """
        assert isinstance(graph, TablesKnowledgeGraph), \
                'Argument graph must be a TablesKnowledgeGraph; got {}'.format(type(graph))
        self.graph = graph
        self.debug = debug
        self.forbid_partial_empty = forbid_partial_empty
        self.cache = {}

    def execute(self, y_toks, old_denotation=None):
        """Return the denotation of the formula.

        Args:
            y_toks (list[Predicate]): the formula
            old_denotation (TablesDenotation)
        Returns:
            TablesDenotation
            The denotation is not finalized.
        Throws:
            Exception if the formula is malformed.
        """
        if self.debug:
            print('Executing: {} (old deno: {})'.format(y_toks, old_denotation))
        if old_denotation:
            stack = TablesDenotation(old_denotation)  # copy
            assert stack.utterance_idx == old_denotation.utterance_idx
        else:
            stack = TablesDenotation()
            assert stack.utterance_idx == 0
        for predicate in y_toks:
            if predicate.name == EOU:
                stack.increment_utterance_idx()
            else:
                self.apply(predicate.name, stack)
            if self.debug:
                print(predicate, stack)
        return stack

    def execute_predicate(self, predicate, old_denotation=None):
        """Return the new denotation of the lf when the predicate is added.

        Args:
            predicate (Predicate)
            old_denotation (TablesDenotation)
        Returns:
            denotation (TablesDenotation)
        """
        if predicate.name == EOU:
            if old_denotation is None:
                denotation = TablesDenotation()
            else:
                denotation = TablesDenotation(old_denotation)
            denotation.increment_utterance_idx()
            return denotation
        signature = (str(old_denotation), predicate)
        if signature in self.cache:
            denotation = self.cache[signature]
        else:
            try:
                stack = (TablesDenotation(old_denotation)
                        if old_denotation else TablesDenotation())
                self.apply(predicate.name, stack)
                denotation = stack
            except Exception as e:
                denotation = e
            if len(self.cache) < TablesPostfixExecutor.CACHE_LIMIT:
                self.cache[signature] = denotation
        if isinstance(denotation, TablesDenotation):
            old_utterance_idx = (old_denotation.utterance_idx
                    if old_denotation is not None else 0)
            if denotation.utterance_idx != old_utterance_idx:
                denotation = TablesDenotation(denotation)  # Make a copy
                denotation._utterance_idx = old_utterance_idx
        return denotation

    INVALID_FINAL_DENOTATION = ValueError('Invalid final denotation')

    def finalize(self, denotation):
        """Return the finalized denotation as list[Value]."""
        if (len(denotation) != 1
                or not isinstance(denotation[0], set)
                or not denotation[0]):
            raise TablesPostfixExecutor.INVALID_FINAL_DENOTATION
        values = []
        for item in denotation[0]:
            if isinstance(item, str):
                if not self.graph.has_id(item):
                    raise TablesPostfixExecutor.INVALID_FINAL_DENOTATION
                values.append(StringValue(self.graph.original_string(item)))
            elif isinstance(item, float):
                values.append(NumberValue(item))
            elif isinstance(item, Date):
                values.append(DateValue(item.year, item.month, item.day))
            else:
                # This should not happen.
                assert False, "Unknown item type: {}".format(item)
        return values

    ################################
    # Internal methods

    def apply(self, predicate, stack):
        """Apply the predicate to the stack. The stack is modified in-place.

        Args:
            predicate (basestring): The next predicate to apply.
            stack (TablesDenotation): The current execution stack
        """
        # Predefined operations
        if predicate in AGGREGATES:
            arg = stack.pop()
            stack.append(self.apply_aggregate(predicate, arg))
        elif predicate in MERGES:
            arg2 = stack.pop()
            arg1 = stack.pop()
            stack.append(self.apply_merge_arith(predicate, arg1, arg2))
        elif predicate in BEGIN_GROWS:
            arg = stack.pop()
            stack.append(self.apply_begin_grow(predicate, arg))
        elif predicate in END_GROWS:
            arg = stack.pop()
            stack.append(self.apply_end_grow(predicate, arg))
        # Assert
        elif predicate.startswith(ASSERT_PREFIX):
            unary = predicate[len(ASSERT_PREFIX):]
            assert is_unary(unary)
            self.apply_assert(unary, stack[-1])
        # Unary or Binary
        elif predicate == TYPE_ROW:
            stack.append(self.apply_type_row(predicate))
        elif is_unary(predicate):
            stack.append(self.apply_unary(predicate))
        elif is_binary(predicate):
            arg = stack.pop()
            stack.append(self.apply_join_fast(predicate, arg))
        else:
            raise ValueError('Unknown predicate {}'.format(predicate))
        # Optional: Check if the partial denotation is empty.
        if self.forbid_partial_empty:
            if (not stack[-1] or (isinstance(stack[-1], dict) and
                all(not x for x in stack[-1].values()))):
                raise self.EMPTY_EXCEPTION

    EMPTY_EXCEPTION = ValueError('Denotation is empty!')

    ################################
    # Operators

    def apply_unary(self, predicate):
        unary = parse_unary(predicate)
        if (isinstance(unary, Date) and
                (unary.year == -1 or unary.month == -1 or unary.day == -1)):
            return GenericDateInfiniteSet(unary)
        else:
            return {unary}

    def apply_type_row(self, predicate):
        return self.graph.all_rows

    @handle_dict_1arg
    def apply_join(self, predicate, arg):
        assert isinstance(predicate, str), str(predicate)
        assert isinstance(arg, (set, InfiniteSet)), str(arg)
        if predicate in SPECIAL_BINARIES:
            if predicate == '!=':
                assert len(arg) == 1, '{} takes exactly 1 object; got {}'.format(predicate, arg)
                thing = next(iter(arg))
                return NeqInfiniteSet(thing)
            elif predicate in ('<', '<=', '>', '>='):
                if isinstance(arg, GenericDateInfiniteSet):
                    arg = [arg.min_()] if predicate in ('<', '>=') else [arg.max_()]
                assert len(arg) == 1, '{} takes exactly 1 object; got {}'.format(predicate, arg)
                thing = next(iter(arg))
                return RangeInfiniteSet(predicate, thing)
            else:
                raise NotImplementedError(predicate)
        elif is_binary_name(predicate):
            return self.graph.join(predicate, arg)
        elif is_reversed_binary_name(predicate):
            return self.graph.reversed_join(predicate[1:], arg)
        else:
            raise NotImplementedError(predicate)

    JOIN_EXCEPTION = ValueError('Join Exception!')

    def apply_join_fast(self, predicate, arg):
        if predicate == '!=':
            if isinstance(arg, dict):
                answer = {}
                for key, thing in arg.items():
                    if len(thing) != 1:
                        raise self.JOIN_EXCEPTION
                    answer[key] = NeqInfiniteSet(next(iter(thing)))
                return answer
            elif len(arg) != 1:
                raise self.JOIN_EXCEPTION
            return NeqInfiniteSet(next(iter(arg)))
        elif predicate in ('<', '<=', '>', '>='):
            if isinstance(arg, dict):
                answer = {}
                for key, thing in arg.items():
                    if isinstance(thing, GenericDateInfiniteSet):
                        thing = [thing.min_()] if predicate in ('<', '>=') else [thing.max_()]
                    if len(thing) != 1:
                        raise self.JOIN_EXCEPTION
                    answer[key] = RangeInfiniteSet(predicate, next(iter(thing)))
                return answer
            else:
                if isinstance(arg, GenericDateInfiniteSet):
                    arg = [arg.min_()] if predicate in ('<', '>=') else [arg.max_()]
                if len(arg) != 1:
                    raise self.JOIN_EXCEPTION
                return RangeInfiniteSet(predicate, next(iter(arg)))
        elif predicate[0] == '!':
            relation = predicate[1:]
            if isinstance(arg, dict):
                return {key: self.graph.reversed_join(relation, things)
                        for (key, things) in arg.items()}
            return self.graph.reversed_join(relation, arg)
        else:
            if isinstance(arg, dict):
                return {key: self.graph.join(predicate, things)
                        for (key, things) in arg.items()}
            return self.graph.join(predicate, arg)

    def apply_assert(self, unary, stack_top):
        assert isinstance(stack_top, set), 'Stack top {} is not a set'.format(stack_top)
        assert len(stack_top) == 1, 'Stack top {} has size more than 1'.format(stack_top)
        thing = next(iter(stack_top))
        assert parse_unary(unary) == thing

    @handle_dict_1arg
    def apply_aggregate(self, predicate, arg):
        if predicate == 'count':
            return {float(len(arg))}
        agreed_type = ensure_same_type(arg, ['N', 'D'])
        if predicate == 'max':
            return {max(arg)}
        if predicate == 'min':
            return {min(arg)}
        assert agreed_type == 'N', 'Cannot do {} over non-numbers'.format(predicate)
        if predicate == 'sum':
            return {sum(arg)}
        if predicate == 'avg':
            return {sum(arg) / len(arg)}
        raise NotImplementedError(predicate)

    @handle_dict_2args
    def apply_merge_arith(self, predicate, arg1, arg2):
        if predicate in ('and', 'or'):
            return (arg1 & arg2) if predicate == 'and' else (arg1 | arg2)
        elif predicate == 'diff':
            assert isinstance(arg1, set) and isinstance(arg2, set)
            assert len(arg1) == 1 or len(arg2) == 1, 'One of diff arguments must have size 1'
            if len(arg1) == 1:
                return {abs(x - next(iter(arg1))) for x in arg2}
            else:
                return {abs(x - next(iter(arg2))) for x in arg1}
        raise NotImplementedError(predicate)

    def apply_begin_grow(self, predicate, arg):
        assert isinstance(arg, set), \
                'begin_grow only operates on a finite unary; got {}'.format(arg)
        return dict((x, {x}) for x in arg)

    def apply_end_grow(self, predicate, arg):
        assert isinstance(arg, dict), \
                'end_grow only operates on a dict; got {}'.format(arg)
        agreed_type = ensure_same_type(arg, ['N', 'D'])
        best_keys = set()
        best_value = None
        for key, values in arg.items():
            for value in values:
                if (best_value is None
                        or (predicate == 'argmin' and value < best_value)
                        or (predicate == 'argmax' and value > best_value)):
                    best_value = value
                    best_keys = {key}
                elif value == best_value:
                    best_keys.add(key)
        return best_keys


################################
# For profiling

def add_decorated_methods(profiler):
    for k, v in list(TablesPostfixExecutor.__dict__.items()):
        if hasattr(v, 'original_fn'):
            print('Adding function {} to profiler'.format(k))
            profiler.add_function(v)
            profiler.add_function(v.original_fn)
