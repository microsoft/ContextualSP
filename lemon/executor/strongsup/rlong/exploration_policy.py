from abc import ABCMeta, abstractmethod
import itertools

from strongsup.parse_case import ParseCase
from strongsup.exploration_policy import Beam
from strongsup.rlong.predicate import RLongPredicate
from strongsup.rlong.state import RLongAlchemyObject
from strongsup.rlong.world import RLongAlchemyWorld


################################
# Alchemy
# TODO: Refactor things common to other domains

class AlchemyOracleExplorationPolicy(object):

    def infer_paths(self, ex):
        return AlchemyOraclePathFinder(ex).all_actual_paths


class AlchemyOraclePathFinder(object):

    def __init__(self, ex, debug=False):
        self.context = ex.context
        self.world = ex.context.world
        self.initial_state = self.world.initial_state
        self.final_state = ex.answer[0].state
        self.num_steps = len(ex.context.utterances)
        self.coarse_paths = []
        self.find_coarse_paths(self.initial_state, [])
        self.all_actual_paths = []
        for coarse_path in self.coarse_paths:
            self.actual_paths = []
            self.find_actual_paths(coarse_path, None, 0)
            if debug:
                print('-' * 10, [item[1] for item in coarse_path], '-' * 10)
                for path in self.actual_paths:
                    print(' ', path.decisions)
            self.all_actual_paths.extend(self.actual_paths)

    def find_coarse_paths(self, current_state, path_so_far):
        """Populate self.coarse_paths with coarse paths.
        A coarse path is just a list of commands (actions + arguments)
        """
        if len(path_so_far) == self.num_steps:
            if current_state == self.final_state:
                self.coarse_paths.append(path_so_far[:])
            return
        # Try Pour
        for i in range(len(current_state)):
            for j in range(len(current_state)):
                try:
                    new_state, command = current_state.apply_action(
                            'Pour', [current_state[i], current_state[j]])
                except Exception as e:
                    continue
                path_so_far.append((current_state, command, new_state))
                self.find_coarse_paths(new_state, path_so_far)
                path_so_far.pop()
        # Try Mix
        for i in range(len(current_state)):
            try:
                new_state, command = current_state.apply_action(
                        'Mix', [current_state[i]])
            except Exception as e:
                continue
            path_so_far.append((current_state, command, new_state))
            self.find_coarse_paths(new_state, path_so_far)
            path_so_far.pop()
        # Try Drain
        for i in range(len(current_state)):
            for j in range(1, current_state[i].amount + 1):
                try:
                    new_state, command = current_state.apply_action(
                            'Drain', [current_state[i], j])
                except Exception as e:
                    continue
                path_so_far.append((current_state, command, new_state))
                self.find_coarse_paths(new_state, path_so_far)
                path_so_far.pop()

    def find_actual_paths(self, coarse_path, current_parse_case, current_step):
        """Populate self.actual_paths with actual logical forms."""
        if current_step == self.num_steps:
            # Finish up the logical form
            assert current_parse_case is not None
            assert (not isinstance(current_parse_case.denotation, Exception)
                    and current_parse_case.denotation.world_state == self.final_state), \
                    repr(['BUG', current_parse_case.path.decisions, current_parse_case.denotation, self.final_state, 'FINAL', coarse_path])
            self.actual_paths.append(current_parse_case.path)
            return
        # Build LF for the current step
        current_state, command, new_state = coarse_path[current_step]
        if current_parse_case is not None:
            assert (not isinstance(current_parse_case.denotation, Exception)
                    and current_parse_case.denotation.world_state == current_state), \
                    repr([current_parse_case.path.decisions, current_parse_case.denotation, current_state, command, coarse_path])
            history = current_parse_case.denotation.command_history
        else:
            history = None
        args = []
        if command[0] == 'Pour':
            args.append(list(self.get_object_refs(command[1], current_state, history)))
            args.append(list(self.get_object_refs(command[2], current_state, history)))
            args.append(list(self.get_action_refs(command[0], current_state, history)))
        elif command[0] == 'Mix':
            args.append(list(self.get_object_refs(command[1], current_state, history)))
            args.append(list(self.get_action_refs(command[0], current_state, history)))
        elif command[0] == 'Drain':
            args.append(list(self.get_object_refs(command[1], current_state, history)))
            args.append(list(self.get_amount_refs(command[2], current_state, history, command[1])))
            args.append(list(self.get_action_refs(command[0], current_state, history)))
        else:
            raise ValueError('Unknown action: {}'.format(command[0]))
        for combination in itertools.product(*args):
            new_predicates = [y for arg in combination for y in arg]
            self.find_actual_paths(coarse_path,
                    self.extend(current_parse_case, new_predicates),
                    current_step + 1)

    def get_object_refs(self, target_object, current_state, history):
        # Pure index
        yield ['all-objects', str(target_object.position), 'index']
        # Index from the back
        if target_object.position == len(current_state):
            yield ['all-objects', '-1', 'index']
        # Color
        if target_object.color is not None:
            matched = current_state.apply_join(target_object.color, 'Color')
            if len(matched) == 1:
                yield [target_object.color, 'PColor']
            else:
                position = matched.index(target_object) + 1
                yield [target_object.color, 'PColor', str(position), 'index']
                if position == len(matched):
                    yield [target_object.color, 'PColor', '-1', 'index']
        # History
        if history:
            for hist_id, hist in enumerate(history):
                for arg_id, arg in enumerate(hist):
                    if (isinstance(arg, RLongAlchemyObject)
                            and arg.position == target_object.position):
                        yield [str(hist_id + 1), 'H{}'.format(arg_id)]
                        yield [str(hist_id - len(history)), 'H{}'.format(arg_id)]

    def get_amount_refs(self, amount, current_state, history, target_object):
        # Pure number
        yield [str(amount)]
        # Relative number
        if amount == target_object.amount:
            yield ['X1/1']
        # TODO: Other fractions
        # History
        if history:
            for hist_id, hist in enumerate(history):
                for arg_id, arg in enumerate(hist):
                    if (isinstance(arg, int) and arg == amount):
                        yield [str(hist_id + 1), 'H{}'.format(arg_id)]
                        yield [str(hist_id - len(history)), 'H{}'.format(arg_id)]

    def get_action_refs(self, action, current_state, history):
        yield ['A' + action]
        # History
        if history:
            for hist_id, hist in enumerate(history):
                if hist[0] == action:
                    yield [str(hist_id + 1), 'H0']
                    yield [str(hist_id - len(history)), 'H0']

    def extend(self, current_parse_case, new_predicates):
        """Return a new ParseCase caused by extending current_parse_case
        by the predicates in new_predicates.

        Args:
            current_parse_case (ParseCase or None)
            new_predicates (list[RLongPredicate or str])
        returns:
            ParseCase
        """
        for pred in new_predicates:
            if not isinstance(pred, RLongPredicate):
                pred = RLongPredicate(pred)
            if current_parse_case is None:
                current_parse_case = ParseCase.initial(self.context)
            else:
                current_parse_case = ParseCase.extend(current_parse_case)
            current_parse_case.decision = pred
        return current_parse_case
