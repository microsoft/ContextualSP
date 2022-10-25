# from gtd.utils import cached_property

from strongsup.executor import Executor, Denotation
from strongsup.rlong.value import RLongStateValue
from strongsup.rlong.state import RLongObject


################################
# Denotation

class RLongDenotation(tuple, Denotation):
    """A pretty lightweight class representing the intermediate denotation."""
    __slots__ = ()

    def __new__(self, world_state, command_history, execution_stack):
        """Create a new RLongDenotation.

        Args:
            world_state (RLongState): Current states of the objects
            command_history (list[tuple]): List of actions and arguments
            execution_stack (list[object]): Used for building arguments for the next action
        """
        return tuple.__new__(RLongDenotation, (world_state, command_history, execution_stack))

    @property
    def world_state(self):
        return self[0]

    @property
    def command_history(self):
        return self[1]

    @property
    def execution_stack(self):
        return self[2]

    @property
    def utterance_idx(self):
        return len(self[1])


################################
# Executor

class RLongExecutor(Executor):
    """Stack-based executor for alchemy, scene, and tangrams domains.
    """

    def __init__(self, initial_state, debug=False):
        self.initial_state = initial_state
        self.debug = debug

    def execute(self, y_toks, old_denotation=None):
        """Return the intermediate denotation of the formula.

        Args:
            y_toks (list[Predicate]): the formula fragment to be executed
            old_denotation (Denotation): If specified, continue execution
                from this intermediate denotation.
        Returns:
            Denotation
            The denotation is not finalized.
        Throws:
            Exception if the formula is malformed.
        """
        if not old_denotation:
            denotation = RLongDenotation(self.initial_state, [], [])
        else:
            assert isinstance(old_denotation, tuple)
            denotation = RLongDenotation(
                    old_denotation.world_state,
                    old_denotation.command_history,
                    old_denotation.execution_stack[:])
        if self.debug:
            print(('Executing: {} (old deno: {})'.format(y_toks, denotation)))
        for predicate in y_toks:
            denotation = self.apply(predicate.name, denotation)
            if self.debug:
                print((predicate, denotation))
        return denotation

    def execute_predicate(self, predicate, old_denotation=None):
        if not old_denotation:
            denotation = RLongDenotation(self.initial_state, [], [])
        else:
            assert isinstance(old_denotation, tuple)
            denotation = RLongDenotation(
                    old_denotation.world_state,
                    old_denotation.command_history,
                    old_denotation.execution_stack[:])
        return self.apply(predicate.name, denotation)

    STACK_NOT_EMPTY = ValueError('Cannot finalize: Stack not empty')

    def finalize(self, denotation):
        """Return the finalized denotation as list[Value].
        Return None if the denotation cannot be finalized.

        For rlong domain, a denotation can be finalized if the stack is empty.
        The result will be a list of a single RLongValue.
        """
        if denotation.execution_stack:
            raise RLongExecutor.STACK_NOT_EMPTY
        return [RLongStateValue(denotation.world_state)]

    ################################
    # Apply

    def apply(self, name, denotation):
        """Return a new denotation.

        The execution stack can be modified directly.
        But the world state and command history cannot be modified directly;
        a new Denotation object must be created.
        This happens only when an action is performed.

        Args:
            name (str): The next predicate name
            denotation (RLongDenotation): Current denotation
        Returns:
            RLongDenotation
            can be the same object as the input argument
            if only the execution stack is modified
        """
        if len(name) == 1 and name[0].isalpha():
            # Color: Push onto the stack
            denotation.execution_stack.append(name)
            return denotation
        elif name[0] == '-' or name[0].isdigit():
            # Number: Push onto the stack
            denotation.execution_stack.append(int(name))
            return denotation
        elif name[0] == 'X':
            # Fraction: Push onto the stack
            denotation.execution_stack.append(name)
            return denotation
        elif name == 'all-objects':
            # All objects: Push onto the stack
            denotation.execution_stack.append(denotation.world_state.all_objects)
            return denotation
        elif name[0] == 'P':
            # Property: Join with the value
            value = denotation.execution_stack.pop()
            result = denotation.world_state.apply_join(value, name[1:])
            assert result, 'Empty result'
            denotation.execution_stack.append(result)
            return denotation
        elif name[0] == 'D':
            # Double-Property: Join with the values
            value2 = denotation.execution_stack.pop()
            value1 = denotation.execution_stack.pop()
            result = denotation.world_state.apply_double_join(
                    value1, value2, name[1:])
            assert result, 'Empty result'
            denotation.execution_stack.append(result)
            return denotation
        elif name[0] == 'A':
            # Perform action
            new_state, history_entry = denotation.world_state.apply_action(
                    name[1:], denotation.execution_stack)
            return RLongDenotation(new_state,
                    denotation.command_history + [history_entry],
                    denotation.execution_stack)
        elif name == 'index':
            # Perform indexing on a list of objects
            number = denotation.execution_stack.pop()
            assert isinstance(number, int)
            objects = denotation.execution_stack.pop()
            assert isinstance(objects, list)
            if number > 0:
                # Because the LF uses 1-based indexing
                denotation.execution_stack.append(objects[number - 1])
            else:
                # Negative indices: count from the right
                denotation.execution_stack.append(objects[number])
            return denotation
        elif name[0] == 'H':
            # History slot
            number = denotation.execution_stack.pop()
            assert isinstance(number, int)
            # Pull out the argument
            command = denotation.command_history[
                    number - 1 if number > 0 else number]
            if name == 'H0':
                # Get the action and execute
                argument = command[0]
                new_state, history_entry = denotation.world_state.apply_action(
                        argument, denotation.execution_stack)
                return RLongDenotation(new_state,
                        denotation.command_history + [history_entry],
                        denotation.execution_stack)
            elif name == 'HUndo':
                # Get the opposite and execute
                argument = denotation.world_state.reverse_action(command[0])
                new_state, history_entry = denotation.world_state.apply_action(
                        argument, denotation.execution_stack)
                return RLongDenotation(new_state,
                        denotation.command_history + [history_entry],
                        denotation.execution_stack)
            else:
                # Just push onto the stack
                argument = command[int(name[1:])]
                if not isinstance(argument, (int, str)):
                    assert isinstance(argument, RLongObject)
                    argument = denotation.world_state.resolve_argument(argument)
                denotation.execution_stack.append(argument)
                return denotation
        else:
            raise ValueError('Unknown predicate {}'.format(name))
