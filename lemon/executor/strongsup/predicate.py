"""Predicate: output token."""
from gtd.utils import ComparableMixin


class Predicate(ComparableMixin):
    """Represents a step in the logical form (i.e., an output token)."""

    __slots__ = ['_name', '_original_string', '_types']

    def __init__(self, name, original_string=None, types=None):
        """Create Predicate.

        Args:
            name (unicode)
            original_string (unicode)
            types (tuple[unicode])
        """
        self._name = name
        self._original_string = original_string
        self._types = types or tuple()

    def __eq__(self, other):
        return (isinstance(other, Predicate)
                and self._name == other._name)

    def __hash__(self):
        return hash(self._name)

    @property
    def _cmpkey(self):
        return self._name

    def __str__(self):
        return self._name
    __repr__ = __str__

    @property
    def name(self):
        """Name of the predicate.
        Should be unique among the predicates in the same context.

        Returns:
            unicode
        """
        return self._name

    @property
    def original_string(self):
        """Original string of the predicate. Can be None.

        Returns:
            unicode or None
        """
        return self._original_string

    @property
    def types(self):
        """A collection of types.

        Returns:
            tuple[unicode]
        """
        return self._types

    @property
    def delexicalized_name(self):
        """A placeholder used in a delexicalized utterance.
        Can be None if the predicate should not be used for delexicalization.

        A subclass can customize this method to return different placeholders
        for different predicate types.

        Returns:
            unicode or None
        """
        return 'PRED'
