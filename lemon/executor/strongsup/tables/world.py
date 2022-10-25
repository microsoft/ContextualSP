import os
import re
from abc import ABCMeta, abstractproperty

from gtd.utils import cached_property
from dependency.data_directory import DataDirectory
from strongsup.world import World
from strongsup.tables.executor import TablesPostfixExecutor
from strongsup.tables.graph import TablesKnowledgeGraph
from strongsup.tables.predicates_computer import TablesPredicatesComputer


class TableWorld(World, metaclass=ABCMeta):
    """World based on a table.

    The table is actually represented as a TableKnowledgeGraph.
    """

    @abstractproperty
    def graph(self):
        """Return a TablesKnowledgeGraph object."""
        raise NotImplementedError

    @abstractproperty
    def human_readable_path(self):
        """Return the relative path from the root data directory
        to the human readable version of the table.
        """
        raise NotImplementedError

    @cached_property
    def executor(self):
        return TablesPostfixExecutor(self.graph)

    @cached_property
    def predicates_computer(self):
        return TablesPredicatesComputer(self.graph)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.human_readable_path)

    def dump_human_readable(self, fout):
        full_path = os.path.join(DataDirectory.root, self.human_readable_path)
        with open(full_path) as fin:
            print(fin.read(), file=fout)


################################
# WikiTableQuestions tables

class WikiTableWorld(TableWorld):

    def __init__(self, xxx, yyy):
        """Constructs a new GraphPath within the WikiTableQuestions dataset.

        Args:
            xxx, yyy (int): graph path ID; the graph will be loaded from
                `{base_dir}/tagged/{xxx}-tagged/{yyy}.tagged`
        """
        self.xxx, self.yyy = xxx, yyy

    @cached_property
    def graph(self):
        return TablesKnowledgeGraph(
                os.path.join(DataDirectory.wiki_table_questions, 'tagged',
                    str(self.xxx) + '-tagged', str(self.yyy) + '.tagged'))

    @property
    def human_readable_path(self):
        abs_path = os.path.join(
                DataDirectory.wiki_table_questions, 'csv',
                '{}-csv'.format(self.xxx), '{}.table'.format(self.yyy))
        return DataDirectory.relative_path(abs_path)


# Add a wrapper to enable caching
# Usage: one of the following
#     WikiTableWorld('csv/204-csv/56.csv')
#     WikiTableWorld(204, 56)

def wiki_table_world_cache_wrapper(cls):
    CACHE = {}
    def get(*args):
        """Get a WikiTableWorld instance from the ID.

        Args:
            Either a single string argument 'csv/{xxx}-csv/{yyy}.csv'
                or two int arguments xxx, yyy. The graph will be loaded from
                `{base_dir}/tagged/{xxx}-tagged/{yyy}.tagged`
        Returns:
            WikiTableWorld
        """
        if len(args) == 1 and isinstance(args[0], str):
            match = re.match(r'csv/(\d+)-csv/(\d+)\.csv', args[0])
            if not match:
                raise ValueError('wikitable id must have the form '
                        'csv/{xxx}-csv/{yyy}.csv; got ' + args[0])
            xxx, yyy = int(match.group(1)), int(match.group(2))
        elif len(args) == 2:
            xxx, yyy = int(args[0]), int(args[1])
        else:
            raise ValueError('Unrecognized arguments: {}'.format(args))
        if (xxx, yyy) not in CACHE:
            CACHE[xxx, yyy] = cls(xxx, yyy)
        return CACHE[xxx, yyy]
    return get

WikiTableWorld = wiki_table_world_cache_wrapper(WikiTableWorld)


################################
# Artificial tables

class ArtificialTableWorld(TableWorld):

    def __init__(self, path):
        self._path = path

    @cached_property
    def graph(self):
        return TablesKnowledgeGraph(
                os.path.join(DataDirectory.root, self._path))

    @property
    def human_readable_path(self):
        return self._path + '.human'

# Add a wrapper to enable caching

def artificial_table_cache_wrapper(cls):
    CACHE = {}
    def get(path):
        if path not in CACHE:
            CACHE[path] = cls(path)
        return CACHE[path]
    return get

ArtificialTableWorld = artificial_table_cache_wrapper(ArtificialTableWorld)
