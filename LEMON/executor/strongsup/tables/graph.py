"""Knowledge graph constructed from a table.

The graph is stored as a list of triples.
"""
import os
import re
import sys
from collections import Counter
from itertools import chain

from strongsup.tables.structure import parse_number, parse_date, InfiniteSet
from strongsup.tables.utils import tsv_unescape, tsv_unescape_list
from dependency.data_directory import DataDirectory


################################
# Constants

REL_NEXT = 'fb:row.row.next'
REL_INDEX = 'fb:row.row.index'
REL_NUMBER = 'fb:cell.cell.number'
REL_DATE = 'fb:cell.cell.date'
REL_NUM2 = 'fb:cell.cell.num2'
REL_PART = 'fb:cell.cell.part'

ALL_GRAPH_BUILT_INS = (REL_NEXT, REL_INDEX, REL_NUMBER, REL_DATE, REL_NUM2, REL_PART)
ALL_GRAPH_BUILT_INS += tuple('!' + x for x in ALL_GRAPH_BUILT_INS)

NULL_CELL = 'fb:cell.null'


################################
# TablesKnowledgeGraph

class TablesKnowledgeGraph(object):
    """A knowledge graph constructed from a table."""

    # Whether to start the row indices from 1 (default) or 0 (legacy).
    FIRST_ROW_INDEX = 1

    def __init__(self, fin, name=None):
        """Construct a TablesKnowledgeGraph from a CoreNLP-tagged table TSV file.
        Each line in the TSV file describe a cell in the context table.

        The following fields must be present:
        - row: Row index (-1 = header row, body row index starts from 0)
        - col: Column index (starts from 0)
        - id: ID of the cell
        - content: Original string content of the cell

        The following fields are optional:
        - number: Possible number normalization ("40 cakes" --> 40)
        - date: Possible date normalization ("Jan 5" --> xx-01-05)
        - num2: Possible second-number normalization ("3-1" --> 1)
        - list and listID: Possible list normalization
            ("Apple, Banana, Orange" --> Apple|Banana|Orange)
            listID contains the ID of the list items,
            while list contains the original strings

        Args:
            fin: A filename string or a file object.
            name: Unique identifier
        """
        if isinstance(fin, str):
            with open(fin) as fin_file:
                self.__init__(fin_file, name=(name or fin))
            return
        self._name = name or (fin.name if hasattr(fin, 'name') else str(fin))
        # Map from relation -> [{first -> seconds}, {second -> firsts}]
        self._relations = {}
        # Map from id -> original string
        self._original_strings = {}
        # Set of all row IDs
        self._rows = set()
        # List of column IDs
        self._columns = []
        # _grid[i][j] = cell id at row i column j
        self._grid = []
        # Now fin is a file object
        current_row, current_row_id = None, None
        header = fin.readline().rstrip('\n').split('\t')
        for line in fin:
            line = line.rstrip('\n').split('\t')
            if len(line) < len(header):
                line.extend([''] * (len(header) - len(line)))
            record = dict(list(zip(header, line)))
            if record['row'] == '-1':
                # column headers
                self._columns.append(record['id'])
                self._original_strings[record['id']] = tsv_unescape(record['content'])
                self._original_strings['!' + record['id']] = tsv_unescape(record['content'])
            else:
                # normal cell
                # Define a bunch of knowledge graph edges.
                row, col = int(record['row']), int(record['col'])
                if current_row != row:
                    current_row = row
                    actual_row_index = current_row + self.FIRST_ROW_INDEX
                    previous_row_id = current_row_id
                    current_row_id = 'fb:row.r{}'.format(actual_row_index)
                    if previous_row_id is not None:
                        # Row --> Next Row relation
                        self._add_relation(REL_NEXT, previous_row_id, current_row_id)
                    # Row --> Index relation
                    self._add_relation(REL_INDEX, current_row_id, float(actual_row_index))
                    self._rows.add(current_row_id)
                    self._grid.append([])
                # Assume that the cells are listed in the correct order
                assert len(self._grid[row]) == col
                self._grid[row].append(record['id'])
                self._original_strings[record['id']] = tsv_unescape(record['content'])
                # Row --> Cell relation
                self._add_relation(self._columns[col], current_row_id, record['id'])
                # Normalization relations
                if record.get('number'):
                    for second in (parse_number(x) for x in record['number'].split('|')):
                        self._add_relation(REL_NUMBER, record['id'], second)
                if record.get('date'):
                    for second in (parse_date(x) for x in record['date'].split('|')):
                        self._add_relation(REL_DATE, record['id'], second)
                if record.get('num2'):
                    for second in (parse_number(x) for x in record['num2'].split('|')):
                        self._add_relation(REL_NUM2, record['id'], second)
                if record.get('listId'):
                    list_ids = record['listId'].split('|')
                    for second in list_ids:
                        self._add_relation(REL_PART, record['id'], second)
                    # Original strings for listIds
                    list_strings = tsv_unescape_list(record['list'])
                    for list_id, list_string in zip(list_ids, list_strings):
                        self._original_strings[list_id] = list_string

    def _add_relation(self, relation, first, second):
        """Internal function for adding a knowledge graph edge (x, r, y).

        Args:
            relation: Relation r (string)
            first: Entity x (string, number, or date)
            second: Entity y (string, number, or date)
        """
        mapping = self._relations.setdefault(relation, [{}, {}])
        mapping[0].setdefault(first, set()).add(second)
        mapping[1].setdefault(second, set()).add(first)

    ################################
    # Queries

    @property
    def name(self):
        return self._name

    def __str__(self):
        return '<TablesKnowledgeGraph {}>'.format(self._name.encode('utf8', 'ignore'))
    __repr__ = __str__

    @property
    def executor(self):
        try:
            return self._executor
        except AttributeError:
            # Import here to prevent recursive import
            from strongsup.tables.executor import TablesPostfixExecutor
            self._executor = TablesPostfixExecutor(self)
            return self._executor

    def join(self, relation, seconds):
        """Return the set of all x such that for some y in seconds,
        (x, relation, y) is in the graph.

        Note that the shorthand reversed relations (e.g., !fb:row.row.name) does not work.

        Args:
            relation (basestring): relation r
            seconds (set, InfiniteSet, or list): the set of y's
        Returns:
            the set of x's
        """
        second_to_firsts = self._relations.get(relation, [{}, {}])[1]
        if isinstance(seconds, (list, set)):
            return set(chain.from_iterable(second_to_firsts.get(y, []) for y in seconds))
        elif isinstance(seconds, InfiniteSet):
            return set(chain.from_iterable(xs for (y, xs) in second_to_firsts.items() if y in seconds))
        else:
            raise NotImplementedError('? . {} . {}'.format(relation, seconds))

    def reversed_join(self, relation, firsts):
        """Return the collection of all y such that for some x in firsts,
        (x, relation, y) is in the graph.

        Note that the shorthand reversed relations (e.g., !fb:row.row.name) does not work.

        Args:
            relation (basestring): Relation r (string)
            firsts (set, InfiniteSet, or list): the set of x's
        Returns:
            the set of y's
        """
        first_to_seconds = self._relations.get(relation, [{}, {}])[0]
        if isinstance(firsts, (list, set)):
            return set(chain.from_iterable(first_to_seconds.get(x, []) for x in firsts))
        elif isinstance(firsts, InfiniteSet):
            return set(chain.from_iterable(ys for (x, ys) in first_to_seconds.items() if x in firsts))
        else:
            raise NotImplementedError('{} . {} . ?'.format(firsts, relation))

    @property
    def all_rows(self):
        """Return the set of all rows fb:row.r0, ..., fb:row.r(M-1)"""
        return self._rows

    @property
    def all_columns(self):
        """Return the set of all column IDs"""
        return set(self._columns)

    def has_id(self, id_):
        return id_ in self._original_strings

    def original_string(self, id_):
        """Return the original string (e.g., fb:cell.obama --> "Obama")"""
        return self._original_strings[id_]


################################
# Debug

if __name__ == '__main__':
    table = TablesKnowledgeGraph(sys.argv[1])
    print(table._rows)
    print(table._columns)
    print(table._grid)
    print(table._relations)
    print(table._original_strings)
