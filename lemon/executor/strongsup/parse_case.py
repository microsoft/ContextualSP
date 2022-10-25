from abc import ABCMeta, abstractproperty
from collections import Sequence

import numpy as np

from gtd.utils import set_once_attribute


class ParseCase(object, metaclass=ABCMeta):
    """Necessary and sufficient information to make a prediction about the next decision.

    Attributes that must be assigned upon creation:
        - context (Context): Context
        - choices (list[Predicate]): List of possible Predicates

    Attributes that can be assigned later:
        - choice_logits (list[float]): Logit score for each choice.
            Have the same length as self.choices.
        - choice_log_probs (list[float]): Log of softmaxed score for each choice.
            Have the same length as self.choices.
        - decision (Predicate): Predicate that the model decided to predict.
            Must be a member of self.choices

    Implied attributes:
        - denotation (object): Result of the execution on the decided Predicates up to the current self.decision
            Only defined when self.decision is already assigned
        - logit (float): Logit of the decision
        - log_prob (float): Log probability of the decision
    """
    __slots__ = ['_context', '_choices',
            #'_choice_logits', '_choice_log_probs', '_decision',
            # For speed, enable the following line instead of the one above ...
            'choice_logits', 'choice_log_probs', 'decision', 'pretty_embed',
            '_logit', '_log_prob', '_denotation']

    # And comment out these 3 lines
    #choice_logits = set_once_attribute('_choice_logits')
    #choice_log_probs = set_once_attribute('_choice_log_probs')
    #decision = set_once_attribute('_decision')

    @property
    def context(self):
        """The context (Context object)."""
        return self._context

    @property
    def choices(self):
        """A list of possible choices (list[Predicate])."""
        return self._choices

    @abstractproperty
    def _previous_cases(self):
        """A list of the previous cases (list[ParseCase])."""
        pass

    @property
    def logit(self):
        """Logit (score) of the decision on this ParseCase only (float)."""
        if not hasattr(self, '_logit'):
            self._logit = self.choice_logits[self.choices.index(self.decision)]
        return self._logit

    @abstractproperty
    def cumulative_logit(self):
        """Sum of the logits of the decisions up to this ParseCase (float)."""
        pass

    @property
    def log_prob(self):
        """Log-Probability of the decision on this ParseCase only (float)."""
        if not hasattr(self, '_log_prob'):
            self._log_prob = self.choice_log_probs[self.choices.index(self.decision)]
        return self._log_prob

    @abstractproperty
    def cumulative_log_prob(self):
        """Log-Probability of the decisions up to this ParseCase (float)."""
        pass

    @property
    def previous_decisions(self):
        """A list of the previous decisions (List[Predicate])."""
        return [c.decision for c in self._previous_cases]

    def __str__(self):
        return '{{{};utt {}/{};[{}];{} from {} choices}}'.format(
                '|'.join(' '.join(x.encode('utf-8') for x in u)
                    for u in self.context.utterances)[:20] + '...',
                self.current_utterance_idx,
                len(self.context.utterances),
                ' '.join(pred.name for pred in self.previous_decisions),
                (self.decision if hasattr(self, 'decision') else None),
                len(self.choices))
    __repr__ = __str__

    @property
    def path(self):
        """The sequence of ParseCases leading up to and including this one.

        Returns:
            ParsePath
        """
        cases = self._previous_cases + [self]
        return ParsePath(cases)

    @property
    def denotation(self):
        """The denotation of the decisions up to the current decision.
        If the execution is successful, the denotation is an arbitrary object
        returned from the executor. Otherwise, the denotation is an Exception.
        """
        try:
            return self._denotation
        except AttributeError:
            y_toks = [self.decision]
            executor = self.context.executor
            old_denotation = None
            for case in reversed(self._previous_cases):
                if hasattr(case, '_denotation'):
                    old_denotation = case._denotation
                    break
                else:
                    y_toks.append(case.decision)
            if isinstance(old_denotation, Exception):
                self._denotation = old_denotation
            else:
                try:
                    self._denotation = executor.execute(y_toks[::-1], old_denotation)
                except Exception as e:
                    self._denotation = e
            return self._denotation

    @property
    def current_utterance_idx(self):
        """Index of the utterance we are focusing on, PRIOR to making a decision for this ParseCase."""
        previous_cases = self._previous_cases
        if len(previous_cases) == 0:
            utterance_idx = 0  # we always start on the first utterance
        else:
            previous_case = previous_cases[-1]
            denotation = previous_case.denotation
            assert not isinstance(denotation, Exception)
            utterance_idx = denotation.utterance_idx
        return utterance_idx

    @property
    def current_utterance(self):
        """The utterance we are focusing on, PRIOR to making a decision for this ParseCase."""
        return self.context.utterances[self.current_utterance_idx]

    @property
    def next_utterance_idx(self):
        """Index of the utterance we will focus on next, AFTER making a decision for this ParseCase.
        Only callable if decision is already set.
        Return len(context.utterances) if there is no utterance left.
        """
        assert not isinstance(self.denotation, Exception)
        return self.denotation.utterance_idx

    @property
    def next_utterance(self):
        """The utterance we will focus on next, AFTER making a decision for this ParseCase.
        Only callable if decision is already set.
        Return None if there is no utterance left.
        """
        next_utterance_idx = self.next_utterance_idx
        if next_utterance_idx == len(self.context.utterances):
            return None
        return self.context.utterances[next_utterance_idx]

    @classmethod
    def initial(cls, context):
        """Convenience method for creating a new InitialParseCase.

        Args:
            context (Context)
        Returns:
            InitialParseCase
        """
        choices = context.predicates
        return InitialParseCase(context, choices)

    @classmethod
    def extend(cls, previous_case):
        """Convenience method for creating a new RecursiveParseCase.

        Args:
            previous_case (ParseCase)
        Returns:
            RecursiveParseCase
        """
        choices = previous_case.context.predicates
        return RecursiveParseCase(previous_case, choices)

    def __hash__(self):
        return hash((self.context, self.choices, self.decision))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.context == other.context and
                self.choices == other.choices and
                self.decision == other.decision)

    def valid_continuations(self, path_checker):
        """Returns all of the valid continuations of this case extending from
        this path according to the path_checker. A path is valid if it is
        terminated and finalizable or unterminated and checks out with the
        path_checker.

        Args:
            path_checker (PathChecker)

        Returns:
            list[ParsePath]: the continuations
        """
        continuations = []
        for choice in self.choices:
            clone = self.copy_with_decision(choice)
            denotation = clone.denotation
            if not isinstance(denotation, Exception):
                path = clone.path
                if path.terminated:
                    if path.finalizable:
                        continuations.append(path)
                elif path_checker(path):
                    continuations.append(path)
        return continuations


class InitialParseCase(ParseCase):
    """Represents the initial ParseCase."""
    __slots__ = []

    def __init__(self, context, choices):
        self._context = context
        self._choices = choices

    @property
    def _previous_cases(self):
        return []

    @property
    def cumulative_logit(self):
        return self.logit

    @property
    def cumulative_log_prob(self):
        return self.log_prob

    def copy_with_decision(self, decision):
        """Return a copy with a specific decision"""
        clone = InitialParseCase(self._context, self._choices)
        clone.choice_logits = self.choice_logits
        clone.choice_log_probs = self.choice_log_probs
        clone.decision = decision
        clone.pretty_embed = self.pretty_embed
        try:
            clone._denotation = self._context.executor.execute_predicate(decision)
        except Exception as e:
            clone._denotation = e
        return clone

class RecursiveParseCase(ParseCase):
    """Represents a non-initial ParseCase."""
    __slots__ = ['_prev_case', '_cumulative_logit', '_cumulative_log_prob']

    def __init__(self, previous_case, choices):
        """Create a ParseCase from a previous case.

        Args:
            previous_case (ParseCase): the previous ParseCase
            choices (list[Predicate]): a list of possible next decisions
        """
        try:
            previous_case.decision
        except AttributeError:
            raise RuntimeError('Previous ParseCase must already have a decision.')

        self._prev_case = previous_case
        self._context = previous_case.context
        self._choices = choices

    @property
    def _previous_cases(self):
        case = self._prev_case
        p = []
        while True:
            p.append(case)
            if isinstance(case, RecursiveParseCase):
                case = case._prev_case
            else:
                break
        return list(reversed(p))

    @property
    def cumulative_logit(self):
        if not hasattr(self, '_cumulative_logit'):
            self._cumulative_logit = self._prev_case.cumulative_logit + self.logit
        return self._cumulative_logit

    @property
    def cumulative_log_prob(self):
        if not hasattr(self, '_cumulative_log_prob'):
            self._cumulative_log_prob = self._prev_case.cumulative_log_prob + self.log_prob
        return self._cumulative_log_prob

    def copy_with_decision(self, decision):
        """Return a copy with a specific decision"""
        clone = RecursiveParseCase(self._prev_case, self._choices)
        clone.choice_logits = self.choice_logits
        clone.choice_log_probs = self.choice_log_probs
        clone.decision = decision
        clone.pretty_embed = self.pretty_embed
        try:
            clone._denotation = self._context.executor.execute_predicate(decision, self._prev_case.denotation)
        except Exception as e:
            clone._denotation = e
        return clone


class ParsePath(Sequence):
    """Represent an entire Sequence of ParseCases."""
    __slots__ = ['_cases', '_context', '_finalized_denotation', '_is_zombie']

    @classmethod
    def empty(cls, context):
        return ParsePath([], context)

    def __init__(self, cases, context=None):
        self._cases = cases
        if not cases:
            if context is None:
                raise RuntimeError('Must specify context for an empty ParsePath')
            self._context = context
        else:
            self._context = cases[0].context
        self._is_zombie = False

    def __getitem__(self, i):
        return self._cases[i]

    def __len__(self):
        return len(self._cases)

    def __str__(self):
        return 'Path' + str(self._cases)
    __repr__ = __str__

    def __hash__(self):
        return hash((tuple(self._cases), self._context))

    def __eq__(self, other):
        return self._cases == other._cases and self._context == other._context

    @property
    def denotation(self):
        """The intermediate denotation (Denotation object)"""
        assert self._cases
        return self._cases[-1].denotation

    @property
    def finalized_denotation(self):
        """The finalized denotation (list[Value]).
        Only available when the path is terminated."""
        if self._is_zombie:
            return []           # Always incorrect
        if not hasattr(self, '_finalized_denotation'):
            assert self.terminated
            executor = self.context.executor
            denotation = self.denotation
            self._finalized_denotation = executor.finalize(denotation)
        return self._finalized_denotation

    @property
    def context(self):
        """The context (Context object)"""
        return self._context

    @property
    def decisions(self):
        """The entire sequence of decisions."""
        return [case.decision for case in self]

    @property
    def score(self):
        """The overall raw score (total logit) of the path.
        All cases must already have been scored for this method to work.
        """
        if not self._cases:
            return 0.
        return self._cases[-1].cumulative_logit

    @property
    def log_prob(self):
        """The overall log-probability of the path.
        All cases must already have been scored for this method to work.
        """
        if not self._cases:
            return 0.
        return self._cases[-1].cumulative_log_prob

    @property
    def locally_normalized_prob(self):
        """The overall locally normalized probability of the path.
        All cases must already have been scored for this method to work.
        """
        return np.exp(self.log_prob)

    @property
    def terminated(self):
        """Whether the path is terminated.
        A path is terminated when all utterances were consumed
        or the path is a zombie path.
        """
        if not self._cases:
            return False
        if self._is_zombie:
            return True
        return self.denotation.utterance_idx == len(self.context.utterances)

    def extend(self):
        """Create a new ParseCase that would continue from the path.

        Return:
            ParseCase
        """
        if not self._cases:
            return ParseCase.initial(self.context)
        else:
            return ParseCase.extend(self._cases[-1])

    @property
    def finalizable(self):
        """Takes a terminated ParsePath and checks if its denotation
        can be finalized

        Args:
            path (ParsePath): Must be terminated

        Returns:
            bool: Whether the denotation can be finalized or not
        """
        assert self.terminated

        try:
            self.finalized_denotation
            return True
        except ValueError as e:
            return False

    def zombie_clone(self):
        """Make a clone of the path but with is_zombie = True.
        Used in REINFORCE for giving negative reward to futile paths.
        """
        assert len(self._cases) > 0
        path = ParsePath(self._cases)
        path._is_zombie = True
        return path


class PrettyCaseEmbedding(object):
    """Visualize how ParseModel embeds a Case."""

    def __init__(self, history_hash, stack_hash):
        """

        Args:
            history_hash (np.ndarray): of shape [history_length]
            stack_hash (np.ndarray): of shape [max_stack_size]
        """
        self.history_hash = history_hash
        self.stack_hash = stack_hash

    def __repr__(self):
        return 'history: {} stack: {}'.format(self.history_hash, self.stack_hash)

