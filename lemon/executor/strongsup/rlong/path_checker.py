from strongsup.path_checker import PathChecker


class RLongPathChecker(PathChecker):

    def __init__(self, config):
        PathChecker.__init__(self, config)
        self._max_stack_size = config.get('max_stack_size')
        self._action_must_clear_beam = config.get('action_must_clear_beam')

    def __call__(self, path):
        """Check whether the path should be added to the beam.

        Args:
            path (ParsePath)
        Returns:
            boolean
        """
        if (self._max_stack_size
                and len(path.denotation.execution_stack) > self._max_stack_size):
            return False
        if (self._action_must_clear_beam
                and path.denotation.execution_stack
                and path[-1].decision.name[0] == 'A'):
            return False
        return True
