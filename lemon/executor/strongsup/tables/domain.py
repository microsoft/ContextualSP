import os

from gtd.utils import cached_property
from dependency.data_directory import DataDirectory

from strongsup.domain import Domain
from strongsup.dataset import Dataset, DatasetFromFile
from strongsup.tables.predicate import (
        FIXED_PREDICATES,
        WikiTablePredicate, WikiTablePredicateType,
        )


class TablesDomain(Domain):

    def load_datasets(self):
        # TODO: Add final dataset (pristine test set); empty dataset for now
        config = self.config.dataset
        if config.name == 'artificial-steps':
            from artificialtables.artificialdata import ArtificialStepsDataset
            train = ArtificialStepsDataset(
                config.difficulty, config.train_examples,
                config.num_tables, seed=config.train_seed,
                columns=os.path.join(DataDirectory.columns, config.train_columns))
            valid = ArtificialStepsDataset(
                config.difficulty, config.valid_examples,
                config.num_tables, seed=config.valid_seed,
                columns=os.path.join(DataDirectory.columns, config.valid_columns))
            final = Dataset()
            return train, valid, final
        elif config.name == 'artificial-wikitables':
            from artificialtables.artificialdata import ArtificialWikiTablesDataset
            train = ArtificialWikiTablesDataset(
                config.difficulty, config.train_examples,
                config.num_tables, seed=config.train_seed,
                columns=os.path.join(DataDirectory.columns, config.train_columns))
            valid = ArtificialWikiTablesDataset(
                config.difficulty, config.valid_examples,
                config.num_tables, seed=config.valid_seed,
                columns=os.path.join(DataDirectory.columns, config.valid_columns))
            final = Dataset()
            return train, valid, final
        elif config.name == 'steps' or config.name == 'sequential-questions':
            from strongsup.tables.example_factory import WikiTableExampleFactory
            # TODO: Change name of supervised (this just lets you have
            # logical forms)
            filename_to_examples = lambda filename: \
                    WikiTableExampleFactory(filename, supervised=True).examples
            train = DatasetFromFile(config.train_file, filename_to_examples)
            valid = DatasetFromFile(config.valid_file, filename_to_examples)
            final = Dataset()
            return train, valid, final
        else:
            raise ValueError('Dataset {} not supported.'.format(config.name))

    def _get_path_checker(self, prune_config):
        from strongsup.tables.path_checker import TablesPathChecker
        return TablesPathChecker(prune_config)

    @property
    def fixed_predicates(self):
        return FIXED_PREDICATES

    @property
    def all_types(self):
        return list(WikiTablePredicateType.ALL_TYPES)
