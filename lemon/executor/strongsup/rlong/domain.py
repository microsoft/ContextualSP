import os
from dependency.data_directory import DataDirectory

from strongsup.domain import Domain
from strongsup.dataset import DatasetFromFile

from strongsup.rlong.path_checker import RLongPathChecker
from strongsup.rlong.predicate import RLongPredicateType
from strongsup.rlong.predicates_computer import get_fixed_predicates


class RLongDomain(Domain):

    def load_datasets(self):
        config = self.config.dataset
        from strongsup.rlong.example_factory import RLongExampleFactory
        train = DatasetFromFile(config.train_file, lambda filename:
                RLongExampleFactory(filename, config.name,
                    config.train_num_steps,
                    config.train_slice_steps_from_middle).examples)
        valid = DatasetFromFile(config.valid_file, lambda filename:
                RLongExampleFactory(filename, config.name,
                    config.valid_num_steps,
                    config.valid_slice_steps_from_middle).examples)
        final = DatasetFromFile(config.final_file, lambda filename:
                RLongExampleFactory(filename, config.name,
                    config.final_num_steps,
                    config.final_slice_steps_from_middle).examples)
        return train, valid, final

    def _get_path_checker(self, prune_config):
        return RLongPathChecker(prune_config)

    @property
    def fixed_predicates(self):
        return get_fixed_predicates(self.config.dataset.name)

    @property
    def all_types(self):
        return list(RLongPredicateType.ALL_TYPES)
