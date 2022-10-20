# Value interface
from abc import ABCMeta, abstractmethod


class Value(object, metaclass=ABCMeta):
    """A value represents an item in either a denotation (gold or predicted)"""

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value based on the
        official criteria.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    def train_match(self, other):
        """Return a boolean of whether self and other are considered
        equal at train time. This can be used to encourage the model to
        predict values with the right type.

        The default is to use match.

        Args:
            other: Value
        """
        return self.match(other)


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value] or set[Value])
        predicted_values (list[Value] or set[Value])
    Returns:
        bool
    """
    if isinstance(predicted_values, Exception):
        # the executor can return Exceptions as the denotation, if the logical form does not make sense
        return False
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True
