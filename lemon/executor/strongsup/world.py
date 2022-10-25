from abc import ABCMeta, abstractproperty, abstractmethod


class World(object, metaclass=ABCMeta):
    """Encapsulate the world where the LF execution takes place.

    Depending on the domain, the world can be a graph (tables domain),
    a list of objects (ctx domain), a grid (blocksworld domain),
    or other things.
    """

    @abstractproperty
    def executor(self):
        """Return an Executor."""
        raise NotImplementedError

    @abstractproperty
    def predicates_computer(self):
        """Return a PredicatesComputer."""
        raise NotImplementedError

    @abstractmethod
    def dump_human_readable(self, fout):
        """Dump the human-readable representation of the world to file.

        Args:
            fout (file object): File to write to.
        """
        raise NotImplementedError
