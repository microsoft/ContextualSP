from abc import ABCMeta, abstractmethod


class PredicatesComputer(object, metaclass=ABCMeta):
    """Compute the set of possible LF predicates for a context, along with
    their alignments to the utterance tokens.

    The resulting predicates are used as `choices` in ParseCase.
    The alignments are used for soft copying and delexicalization.
    """

    @abstractmethod
    def compute_predicates(self, tokens):
        """Compute the possible predicates for the tokens of the utterance.

        Args:
            tokens (list[unicode])
        Returns:
            list[(Predicate, alignment)]
            where alignment is list[(utterance token index, alignment strength)]
        """
        raise NotImplementedError
