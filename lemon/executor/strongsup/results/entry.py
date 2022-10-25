import operator
import os
from gtd.utils import EqualityMixin
from functools import reduce


class ExperimentType(EqualityMixin):
    """Defines the configs for an experiment

    Args:
        configs (list[string]): the config mixins
        base (string): the base config e.g. "default-base"
    """
    @classmethod
    def parse_configs(cls, configs):
        """Creates a new ExperimentType object from list of configs of the
        form configs/rlong/dataset-mixins/something.txt

        Args:
            configs (list[string]): the configs

        Returns:
            ExperimentType
            string: the dataset
            int: the seed
        """
        base = base_filename(configs[0])
        confs = []
        seed = None
        for config in configs[1:]:
            if config.find("dataset-mixins") != -1:
                dataset = base_filename(config)
            elif config.find("seed-mixins") != -1:
                seed = int(base_filename(config).replace("seed=", ""))
            else:
                confs.append(base_filename(config))
        confs.sort()
        # Default configs
        experiment_type = cls(confs, base)
        # Default seed
        if seed is None:
            seed = 0
        return experiment_type, dataset, seed

    def __init__(self, configs, base):
        self._configs = configs
        self._base = base

    @property
    def configs(self):
        return self._configs

    @property
    def base(self):
        return self._base

    def __str__(self):
        configs = '-'.join(self.configs)
        if configs == "":
            configs = self._base
        return "{}".format(configs)
    __repr__ = __str__

    def __hash__(self):
        return hash((tuple(self.configs), self.base))


class Entry(object):
    """A single entry in the Table. Contains results for all seeds of the
    same ExperimentType

    Args:
        experiment_type (ExperimentType): the experiment type
    """
    def __init__(self, experiment_type):
        self._experiment_type = experiment_type
        self._results = {}  # seed -> result value

    def add_seed(self, seed, result_value):
        """Adds a result value associated with this seed

        Args:
            seed (int)
            result_value (ResultValue)
        """
        if seed in self._results:
            raise ValueError("Seed {} already in Entry {}".format(seed, self))
        self._results[seed] = result_value

    def update_seed(self, seed, result_value):
        """Updates the result value associated with this seed

        Args:
            seed (int)
            result_value (ResultValue)
        """
        self._results[seed] = result_value

    def delete_seed(self, seed):
        """Deletes value associated with this seed.

        Args:
            seed (int)
        """
        self._results.pop(seed, None)

    def contains_seed(self, seed):
        """Returns True if there's a value already associated with this seed.

        Args:
            seed (int)

        Returns:
            bool
        """
        return seed in self._results

    def __eq__(self, other):
        return self._experiment_type == other._experiment_type and \
               self._results == other._results

    @property
    def seeds(self):
        return list(self._results.keys())

    @property
    def experiment_type(self):
        return self._experiment_type

    @property
    def name(self):
        return str(self._experiment_type)

    def get_value(self, seed):
        """Returns the ResultValue associated with this seed."""
        return self._results[seed]

    @property
    def best(self):
        """Returns the seed and ResultValue achieving highest
        result value

        Returns:
            seed (int)
            ResultValue
        """
        return max(iter(self._results.items()), key=operator.itemgetter(1))

    @property
    def avg(self):
        """Returns the ResultValue of the average over all seeds

        Returns:
            ResultValue
        """
        return reduce(
                operator.add, list(self._results.values())) / len(self._results)

    @property
    def var(self):
        """Returns the ResultValue of the var over all seeds

        Returns:
            ResultValue
        """
        return reduce(operator.add, ((value - self.avg).squared()
                for value in list(self._results.values()))) / len(self._results)

    def __str__(self):
        return "Entry({}: {})".format(self._experiment_type, self._results)
    __repr__ = __str__


def base_filename(path):
    """Returns the filename without the extension from the path"""
    return os.path.splitext(os.path.basename(path))[0]
