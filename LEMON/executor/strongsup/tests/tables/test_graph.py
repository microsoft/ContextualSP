import os
import pytest

from strongsup.tables.graph import TablesKnowledgeGraph
from strongsup.tables.structure import Date, NeqInfiniteSet, RangeInfiniteSet
from abc import ABCMeta, abstractmethod


class KnowledgeGraphTester(object, metaclass=ABCMeta):
    def test_properties(self, graph):
        assert graph.all_rows == {'fb:row.r{}'.format(x) for x in range(1, self.num_rows + 1)}
        assert graph.all_columns == {'fb:row.row.{}'.format(x) for x in self.columns}
        for name, original_string in list(self.sample_original_strings.items()):
            if original_string is None:
                assert not graph.has_id(name)
            else:
                assert graph.has_id(name)
                assert graph.original_string(name) == original_string

    def test_joins(self, graph):
        for x, r, y in self.symmetric_triples:
            assert graph.join(r, y) == x
            assert graph.reversed_join(r, x) == y
        for x, r, y in self.join_triples:
            assert graph.join(r, y) == x
        for x, r, y in self.reversed_join_triples:
            assert graph.reversed_join(r, x) == y


################################
# Individual cases

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class TestGraphNt0(KnowledgeGraphTester):

    @pytest.fixture()
    def graph(self):
        return TablesKnowledgeGraph(os.path.join(DATA_PATH, 'nt-0.graph'))

    num_rows = 10
    columns = {'year', 'division', 'league', 'regular_season', 'playoffs', 'open_cup', 'avg_attendance'}
    sample_original_strings = {
            'fb:row.row.year': 'Year',
            'fb:row.row.avg_attendance': 'Avg. Attendance',
            'fb:cell.2007': '2007',
            'fb:cell.3rd_usl_3rd': '3rd, USL (3rd)',
            'fb:part.western': 'Western',
            'fb:cell.does_not_exist': None,
            'fb:row.row.index': None,
            'N4': None,
            }
    symmetric_triples = [
            ({'fb:row.r1'}, 'fb:row.row.year', {'fb:cell.2001'}),
            ({'fb:row.r6', 'fb:row.r9', 'fb:row.r10'}, 'fb:row.row.open_cup', {'fb:cell.3rd_round'}),
            ({'fb:row.r4', 'fb:row.r5', 'fb:row.r8'}, 'fb:row.row.open_cup',
                {'fb:cell.4th_round', 'fb:cell.1st_round'}),
            ({'fb:cell.7_169', 'fb:cell.8_567'}, 'fb:cell.cell.number', {7169.0, 8567.0}),
            ({'fb:cell.2001', 'fb:cell.2002', 'fb:cell.2003'}, 'fb:cell.cell.date',
                {Date(2001, -1, -1), Date(2002, -1, -1), Date(2003, -1, -1)}),
            ({'fb:cell.3rd_usl_3rd'}, 'fb:cell.cell.num2', {3.0}),
            ({'fb:row.r4', 'fb:row.r7'}, 'fb:row.row.next', {'fb:row.r5', 'fb:row.r8'}),
            ({'fb:row.r3', 'fb:row.r10', 'fb:row.r4'}, 'fb:row.row.index', {3.0, 10.0, 4.0}),
            ]
    join_triples = [
            ({'fb:row.r2'}, 'fb:row.row.next', {'fb:row.r3', 'fb:row.r42', 'fb:row.r1'}),
            ({'fb:row.r3', 'fb:row.r5'}, 'fb:row.row.index', {3.0, 5.0, 3.14, '4.0'}),
            #({'fb:row.r{}'.format(x) for x in xrange(1, 11)}, 'fb:type.object.type', {'fb:type.row'}),     # Not supported anymore; use `type-row` instead
            ({'fb:row.r{}'.format(x) for x in range(1, 11) if x != 5}, 'fb:row.row.index',
                NeqInfiniteSet(5.0)),
            ({'fb:row.r{}'.format(x) for x in range(1, 11) if x != 5}, 'fb:row.row.year',
                NeqInfiniteSet('fb:cell.2005')),
            ({'fb:cell.2003', 'fb:cell.2004', 'fb:cell.2005'}, 'fb:cell.cell.number',
                RangeInfiniteSet('>=', 2003.0, '<', 2006.0)),
            ({'fb:cell.2003', 'fb:cell.2004', 'fb:cell.2005'}, 'fb:cell.cell.date',
                RangeInfiniteSet('>=', Date(2003, -1, -1), '<', Date(2006, -1, -1))),
            ]
    reversed_join_triples = [
            ({'fb:cell.1st_western', 'fb:5th', 'fb:row.row.year'}, 'fb:cell.cell.part',
                {'fb:part.1st', 'fb:part.western'}),
            ]
