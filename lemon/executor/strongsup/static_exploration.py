import itertools
import time
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque
import logging

from strongsup.parse_case import ParseCase
from strongsup.value import check_denotation


class StaticCase(object, metaclass=ABCMeta):
    """Like a ParseCase, but only statically analyzed, never dynamically executed.

    Primarily used by StaticBatchExploration.
    """

    @abstractmethod
    def seeds(cls):
        """Return a list of seed cases to start searching from."""
        pass

    @abstractmethod
    def extend(self, predicate):
        """Return a new StaticCase which extends from this one."""
        pass

    @abstractproperty
    def choices(self):
        """Choices available from this state."""
        pass

    @abstractproperty
    def length(self):
        """Length of episode so far."""
        pass

    @abstractproperty
    def utterances_read(self):
        """Number of utterances processed so far in this episode."""
        pass

    @abstractproperty
    def stack_depth(self):
        """Depth of execution stack."""
        pass

    @abstractproperty
    def path(self):
        """Return a list of StaticCases."""
        pass


class AlchemyCase(object):
    __slots__ = ['predicate', 'prev_case', 'length', 'utterances_read', 'execution_stack', 'command_history']

    choices = [
        'r', 'y', 'g', 'o', 'p', 'b',
        '1', '2', '3', '4', '5', '6', '7',
        '-1',
        'X1/1',
        'PColor',
        'APour', 'AMix', 'ADrain',
        'all-objects', 'index',
        'H0', 'H1', 'H2',
    ]

    def __init__(self, predicate, prev_case, length, utterances_read, execution_stack, command_history):
        self.predicate = predicate
        self.prev_case = prev_case
        self.length = length
        self.utterances_read = utterances_read
        self.execution_stack = execution_stack
        self.command_history = command_history

    @classmethod
    def seeds(cls):
        seeds = []
        for p in cls.choices:
            state = cls._update_state([], [], p)
            if state is None:
                continue
            exec_stack, cmd_history = state
            case = AlchemyCase(p, None, 1, 0, exec_stack, cmd_history)
            seeds.append(case)
        return seeds

    def extend(self, predicate):
        state = self._update_state(self.execution_stack, self.command_history, predicate)
        if state is None:
            return None  # predicate leads to invalid state
        exec_stack, cmd_history = state

        utterances_read = self.utterances_read
        if predicate[0] == 'A' or predicate == 'H0':
            utterances_read += 1
        return AlchemyCase(predicate, self, self.length + 1, utterances_read, exec_stack, cmd_history)

    @property
    def stack_depth(self):
        return len(self.execution_stack)

    @property
    def path(self):
        path = []
        current = self
        while True:
            path.append(current)
            current = current.prev_case
            if current is None:
                break
        path.reverse()
        return path

    @classmethod
    def _get_args_from_stack(cls, exec_stack, predicate):
        if predicate in ('APour', 'ADrain', 'index'):
            n = 2
        elif predicate in ('AMix', 'PColor') or predicate[0] == 'H':
            n = 1
        else:
            return None

        if len(exec_stack) < n:  # not enough arguments
            return None

        return exec_stack[-n:]

    def __repr__(self):
        return self.predicate

    @classmethod
    def _update_state(cls, exec_stack, command_history, predicate):
        """

        We assume action clears stack.

        Args:
            exec_stack
            command_history
            predicate

        Returns:
            new_exec_stack, new_command_history
        """
        # TYPES
        COLOR = 'CLR'
        BEAKER = 'BKR'
        LIST = 'LST'
        is_number = lambda s: s in ('1', '2', '3', '4', '5', '6', '7', '-1')

        # SIMPLE VALUES
        if predicate in ('r', 'y', 'g', 'o', 'p', 'b'):
            # abstract to COLOR
            return exec_stack + [COLOR], list(command_history)

        if is_number(predicate):
            # preserve numbers exactly
            return exec_stack + [predicate], list(command_history)

        if predicate == 'all-objects':
            # abstract to LIST
            return exec_stack + [LIST], list(command_history)

        # FUNCTIONS
        args = cls._get_args_from_stack(exec_stack, predicate)
        if args is None:
            return None  # not enough arguments

        logging.debug('Args peeked: {}'.format(args))

        prefix = predicate[0]

        # actions
        if prefix == 'A':
            logging.debug('Processing action')
            logging.debug(exec_stack)

            if len(args) != len(exec_stack):  # action must clear stack
                return None

            # type check
            if predicate == 'APour':
                if args != [BEAKER, BEAKER]:
                    return None
            if predicate == 'ADrain':
                if args[0] != BEAKER or not is_number(args[1]):
                    return None
            if predicate == 'AMix':
                if args != [BEAKER]:
                    return None

            new_stack = []
            new_command_history = list(command_history)
            new_command_history.append([predicate] + args)
            return new_stack, new_command_history

        if predicate == 'PColor':
            if args[0] != COLOR:
                return None

            new_stack = exec_stack[:-1]
            new_stack.append(LIST)
            return new_stack, list(command_history)

        if predicate == 'index':
            if args[0] != LIST or not is_number(args[1]):
                return None

            new_stack = exec_stack[:-2]
            new_stack.append(BEAKER)
            return new_stack, list(command_history)

        # history referencing predicates
        if prefix == 'H':
            arg_pos = int(predicate[1:])
            history_idx_str = args[0]

            if not is_number(history_idx_str):
                return None
            if history_idx_str in ('X1/1', '-1'):
                return None

            history_idx = int(history_idx_str) - 1

            try:
                referenced = command_history[history_idx][arg_pos]
            except IndexError:
                return None  # failed to retrieve

            return cls._update_state(exec_stack, command_history, referenced)

        raise ValueError('Invalid predicate: {}'.format(predicate))


class StaticBatchExploration(object):
    def __init__(self, examples, case_type, max_length, max_utterances, max_stack_depth):
        """Perform BFS to find silver logical forms.

        Args:
            examples (list[Example])
            case_type: subclass of Case
            max_length (int): max # predicates in a logical form
            max_utterances (int): max # utterances processed by a logical form
            max_stack_depth (int): max depth of execution stack
        """
        # metrics for reporting
        start_time = time.time()
        visited = 0
        longest_so_far = 0
        max_queue_size = 0

        queue = deque(case_type.seeds())  # seed the queue
        complete = []

        while len(queue) != 0:
            case = queue.popleft()

            # update metrics
            visited += 1
            max_queue_size = max(max_queue_size, len(queue))
            if case.length > longest_so_far:
                now = time.time()
                print('reached length {} after visiting {} states ({} s)'.format(case.length, visited, now - start_time))
            longest_so_far = max(longest_so_far, case.length)
            if visited % 100000 == 0:
                print('visited: {}, completed: {}, peak queue size: {}'.format(visited, len(complete), max_queue_size))

            # prune
            if case.stack_depth > max_stack_depth:
                continue

            has_terminated = case.utterances_read >= max_utterances
            if has_terminated:
                complete.append(case.path)
                continue

            if case.length >= max_length:
                continue

            # extend
            for choice in case.choices:
                new_case = case.extend(choice)
                if new_case is None:
                    continue
                queue.append(new_case)

        self.complete = complete

        self.complete = complete


# Here just for comparison with StaticBatchExploration
# Performs a typical search which uses dynamic execution for pruning.
def simple_bfs(example, path_checker, max_depth):
    root = ParseCase.initial(example.context)
    queue = deque([root])
    terminated = []
    start_time = time.time()

    depth = 0
    max_queue_size = 0
    for i in itertools.count():
        if len(queue) == 0:
            break

        max_queue_size = max(max_queue_size, len(queue))

        case = queue.popleft()
        zeros = [0.] * len(case.choices)
        case.choice_logits = zeros  # not used
        case.choice_log_probs = zeros  # not used

        for choice in case.choices:
            clone = case.copy_with_decision(choice)  # set the decision

            # don't extend cases with invalid denotation
            denotation = clone.denotation
            if isinstance(denotation, Exception):
                continue

            path = clone.path

            if len(path) != depth:
                depth = len(path)
                now = time.time()
                print('reached depth {} after visiting {} states ({}s)'.format(depth, i + 1, now - start_time))
                print('peak queue size: {}'.format(max_queue_size))

            if path.terminated:  # terminates when all the utterances have been processed
                terminated.append(path)
                continue

            if len(path) >= max_depth:
                continue

            # Path is not complete. Apply pruning to see if we should continue.
            if not path_checker(path):
                continue

            # Decide to extend this path.
            new_case = path.extend()
            queue.append(new_case)

    silver_lfs = []
    for path in terminated:
        try:
            if check_denotation(example.answer, path.finalized_denotation):
                silver_lfs.append(path)
        except Exception:
            pass
    return silver_lfs