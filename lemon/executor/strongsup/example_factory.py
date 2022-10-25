from abc import ABCMeta, abstractproperty


class ExampleFactory(object, metaclass=ABCMeta):
    @abstractproperty
    def examples(self):
        """Return an iterable of Examples."""
        raise NotImplementedError
