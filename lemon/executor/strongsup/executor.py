from abc import ABCMeta, abstractmethod, abstractproperty

from strongsup.predicate import Predicate


class Executor(object, metaclass=ABCMeta):
    @abstractmethod
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
        raise NotImplementedError

    def execute_predicate(self, predicate, old_denotation=None):
        """Return the denotation of the formula.

        This method takes only a single Predicate object as the argument.
        This allows more optimization to be performed.

        Args:
            predicate (Predicate): the next predicate
            old_denotation (Denotation): If specified, continue execution
                from this intermediate denotation.
        Returns:
            Denotation
            The denotation is not finalized.
        Throws:
            Exception if the formula is malformed.
        """
        # Default: call execute
        return self.execute([predicate], old_denotation)

    @abstractmethod
    def finalize(self, denotation):
        """Given a Denotation, return its finalized form as list[Value].

        Args:
            denotation (Denotation)
        Returns:
            list[Value] or None
        Raises:
            ValueError if the denotation cannot be finalized
        """
        raise NotImplementedError


class Denotation(object, metaclass=ABCMeta):
    """Intermediate denotation."""

    @abstractproperty
    def utterance_idx(self):
        """Current the utterance index (int).

        Should be incremented every time an end-of-utterance predicate
        or its equivalence is executed.
        """
        raise NotImplementedError
