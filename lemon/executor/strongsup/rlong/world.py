from strongsup.world import World
from strongsup.rlong.executor import RLongExecutor
from strongsup.rlong.predicates_computer import get_predicates_computer
from strongsup.rlong.state import RLongState


class RLongWorld(World):
    """World for Alchemy, Scene, and Tangrams domains."""

    def __init__(self, initial_state):
        """Create a new RLongWorld.

        Args:
            initial_state (RLongState)
        """
        assert isinstance(initial_state, RLongState)
        self._initial_state = initial_state
        self._executor = RLongExecutor(initial_state)

    @property
    def initial_state(self):
        """Return a RLongState object."""
        return self._initial_state

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.initial_state)

    @property
    def executor(self):
        return self._executor

    @property
    def predicates_computer(self):
        return self._PREDICATES_COMPUTER

    def dump_human_readable(self, fout):
        self.initial_state.dump_human_readable(fout)


class RLongAlchemyWorld(RLongWorld):
    _PREDICATES_COMPUTER = get_predicates_computer('alchemy')

class RLongSceneWorld(RLongWorld):
    _PREDICATES_COMPUTER = get_predicates_computer('scene')

class RLongTangramsWorld(RLongWorld):
    _PREDICATES_COMPUTER = get_predicates_computer('tangrams')

class RLongUndogramsWorld(RLongWorld):
    _PREDICATES_COMPUTER = get_predicates_computer('undograms')
