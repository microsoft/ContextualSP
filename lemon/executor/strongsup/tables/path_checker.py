from strongsup.path_checker import PathChecker
from strongsup.utils import EOU


class TablesPathChecker(PathChecker):

    def __init__(self, config):
        PathChecker.__init__(self, config)
        self._max_stack_size = config.get('max_stack_size')
        self._prune_idempotent = config.get('prune_idempotent')

    def __call__(self, path):
        """Check whether the path should be added to the beam.

        Args:
            path (ParsePath)
        Returns:
            boolean
        """
        if (self._max_stack_size
                and len(path.denotation) > self._max_stack_size):
            return False
        if (self._prune_idempotent
                and len(path) > 1
                and path[-1].decision.name != EOU
                and path[-2].denotation == path.denotation):
            return False
        return True
