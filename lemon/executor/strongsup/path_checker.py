from abc import ABCMeta


class PathChecker(object, metaclass=ABCMeta):
    """Check whether a ParsePath should be included in the beam.

    This is used to control the search space especially when the parameters
    are not well initialized.
    """

    def __init__(self, config):
        """Initialize the PathChecker.

        Args:
            config (Config): The decoder.prune section of the configuration.
        """
        self.config = config

    def __call__(self, path):
        """Check whether the path should be added to the beam.

        Args:
            path (ParsePath)
        Returns:
            True if the path should be included; False if it should be pruned.
        """
        raise NotImplementedError
