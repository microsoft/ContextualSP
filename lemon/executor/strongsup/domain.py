from abc import ABCMeta, abstractproperty, abstractmethod

from gtd.utils import cached_property


class Domain(object, metaclass=ABCMeta):
    """Encapsulate all domain-dependent information.

    To add a new domain, create a subclass of domain (in a separate file)
    and then add it to the get_domain method below.
    """

    def __init__(self, config):
        """Initialize the Domain object.

        Args:
            config (gtd.util.Config): Top-level config.
        """
        self.config = config

    @abstractmethod
    def load_datasets(self):
        """Load training and validation datasets according to the config.

        Returns: a tuple (train, valid)
            train (Dataset): Training examples
            valid (Dataset): Validation examples (dev set)
            final (Dataset): Final examples (test set)
        """
        raise NotImplementedError

    @cached_property
    def path_checker(self):
        """Get a PathChecker for this domain.

        Returns:
            A callable that takes a ParsePath and returns a boolean
            indicating whether the ParsePath is OK to be on the beam.
        """
        prune_config = self.config.decoder.get('prune')
        if not prune_config:
            return lambda x: True
        return self._get_path_checker(prune_config)

    @abstractmethod
    def _get_path_checker(self, prune_config):
        """Get a PathChecker for this domain according to the configuration.

        Args:
            prune_config (Config): dataset.prune section of the config.

        Returns:
            A callable that takes a ParsePath and returns a boolean
            indicating whether the ParsePath is OK to be on the beam.
        """
        raise NotImplementedError

    @abstractproperty
    def fixed_predicates(self):
        """Return the list of fixed Predicates.

        Returns:
            list(Predicate)
        """
        raise NotImplementedError

    @abstractproperty
    def all_types(self):
        """Return the list of all possible type names.

        Returns:
            list(str)
        """
        raise NotImplementedError


def get_domain(config):
    """Get the domain object according to the config.

    Args:
        config (gtd.util.Config): Top-level config
    """
    domain_name = config.dataset.domain
    if domain_name == 'tables':
        from strongsup.tables.domain import TablesDomain
        return TablesDomain(config)
    elif domain_name == 'rlong':
        from strongsup.rlong.domain import RLongDomain
        return RLongDomain(config)
    else:
        raise ValueError('Domain {} not supported.'.format(domain_name))
