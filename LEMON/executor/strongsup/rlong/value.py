from strongsup.value import Value


class RLongStateValue(Value):
    """Value based on RLongState."""

    def __init__(self, state):
        self._state = state

    def __repr__(self):
        return repr(self._state)

    @property
    def state(self):
        return self._state

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self._state == other._state)

    def match(self, other):
        return self._state == other._state
