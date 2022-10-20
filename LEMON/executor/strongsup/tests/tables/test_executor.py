import os
import pytest

from strongsup.predicate import Predicate
from strongsup.value import check_denotation
from strongsup.tables.graph import TablesKnowledgeGraph
from strongsup.tables.executor import TablesPostfixExecutor
from strongsup.tables.structure import Date, NeqInfiniteSet, RangeInfiniteSet, GenericDateInfiniteSet
from strongsup.tables.value import StringValue, NumberValue, DateValue

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class TestExecutorNt0(object):

    @pytest.fixture()
    def graph(self):
        return TablesKnowledgeGraph(os.path.join(DATA_PATH, 'nt-0.graph'))

    @pytest.fixture()
    def executor(self, graph):
        return TablesPostfixExecutor(graph)

    def run(self, executor, lf, old_deno=None, expected_deno=None):
        """Assert the executed denotation and return it."""
        if isinstance(lf, str):
            lf = [Predicate(x) for x in lf.split()]
        try:
            d = executor.execute(lf, old_deno)
            if expected_deno is not None:
                assert list(d) == expected_deno
        except:
            # See what is going wrong in more details
            executor.debug = True
            d = executor.execute(lf, old_deno)
            if expected_deno is not None:
                assert list(d) == expected_deno
        # Also test execute_predicate
        current_deno = old_deno
        for predicate in lf:
            current_deno = executor.execute_predicate(predicate, current_deno)
        assert list(d) == list(current_deno)
        return d

    def run_error(self, executor, lf, old_deno=None):
        """Assert that an exception is thrown."""
        if isinstance(lf, str):
            lf = [Predicate(x) for x in lf.split()]
        with pytest.raises(Exception):
            executor.execute(lf, old_deno)

    def test_basic_join(self, executor):
        e = executor
        d = self.run(e, 'fb:cell.2001', None,
                [{'fb:cell.2001'}])
        self.run(e, 'fb:row.row.year', d,
                [{'fb:row.r1'}])
        d = self.run(e, 'fb:cell.2001 fb:row.row.year', None,
                [{'fb:row.r1'}])
        assert d.utterance_idx == 0
        d = self.run(e, 'fb:cell.2001 <EOU> fb:row.row.year', None,
                [{'fb:row.r1'}])
        assert d.utterance_idx == 1
        d = self.run(e, '<EOU> fb:cell.2001 fb:row.row.year <EOU>', None,
                [{'fb:row.r1'}])
        assert d.utterance_idx == 2
        d = self.run(e, 'fb:cell.2001 fb:row.row.year !fb:row.row.league <EOU>', None,
                [{'fb:cell.usl_a_league'}])
        assert d.utterance_idx == 1
        assert e.finalize(d) == [StringValue('USL A-League')]

    def test_errors(self, executor):
        e = executor
        self.run_error(e, 'argmax')
        d = self.run(e, 'fb:cell.2001', None,
                [{'fb:cell.2001'}])
        self.run_error(e, 'fb:row.row.league', d)
        self.run_error(e, '!fb:row.row.year', d)
        self.run_error(e, 'max', d)
        d = self.run(e, 'fb:cell.2002', d,
                [{'fb:cell.2001'}, {'fb:cell.2002'}])
        self.run_error(e, 'and', d)
        d = self.run(e, 'or !fb:cell.cell.number', d,
                [{2001., 2002.}])
        # Test finalization
        d = self.run(e, 'N2003 or', d)
        d = e.finalize(d)
        assert set(d) == {NumberValue(2001.0), NumberValue(2002.0), NumberValue(2003.0)}
        d = self.run(e, 'fb:cell.does_not_exist')
        with pytest.raises(Exception):
            e.finalize(d)

    def test_infinite_set(self, executor):
        e = executor
        self.run(e, 'N8000 > fb:cell.cell.number', None,
            [{'fb:cell.8_567', 'fb:cell.9_734', 'fb:cell.10_727'}])
        d = self.run(e, 'fb:cell.2005 !=')
        assert isinstance(d[0], NeqInfiniteSet)
        d = self.run(e, 'fb:row.row.year', d)
        assert d[0] == {'fb:row.r{}'.format(x) for x in range(1, 11) if x != 5}
        d = self.run(e, 'N2005 != fb:cell.cell.number fb:row.row.year')
        assert d[0] == {'fb:row.r{}'.format(x) for x in range(1, 11) if x != 5}
        d = self.run(e, 'N2005 > N2009 <= and fb:cell.cell.number fb:row.row.year', None,
                [{'fb:row.r6', 'fb:row.r7', 'fb:row.r8', 'fb:row.r9'}])

    def test_operations(self, executor):
        e = executor
        # aggregates
        d = self.run(e, 'fb:cell.usl_a_league fb:row.row.league '
            '!fb:row.row.avg_attendance !fb:cell.cell.number')
        self.run(e, 'sum', d, [{24928.0}])
        self.run(e, 'avg', d, [{6232.0}])
        self.run(e, 'min', d, [{5628.0}])
        self.run(e, 'max', d, [{7169.0}])
        self.run(e, 'count', d, [{4.0}])
        self.run(e, 'fb:cell.usl_a_league fb:row.row.league '
            '!fb:row.row.division count', None, [{1.0}])
        d = self.run(e, 'fb:cell.usl_a_league fb:row.row.league '
            '!fb:row.row.avg_attendance')
        self.run_error(e, 'sum', d)
        d = self.run(e, 'type-row !fb:row.row.year !fb:cell.cell.date')
        self.run(e, 'min', d, [{Date(2001, -1, -1)}])
        self.run(e, 'max', d, [{Date(2010, -1, -1)}])
        self.run(e, 'count', d, [{10.0}])
        # merge
        d = self.run(e, 'fb:cell.usl_a_league fb:row.row.league '
            'fb:cell.quarterfinals fb:row.row.playoffs')
        self.run(e, 'and', d, [{'fb:row.r1', 'fb:row.r4'}])
        self.run(e, 'or count', d, [{6.0}])
        self.run(e, 'type-row N3 fb:row.row.index != and count', None, [{9.0}])
        self.run(e, 'type-row !fb:row.row.avg_attendance !fb:cell.cell.number '
            'N6000 > N8000 < and and count', None, [{4.0}])
        # diff
        self.run(e, 'N11 fb:cell.2001 fb:row.row.year '
            '!fb:row.row.regular_season !fb:cell.cell.number diff',
            None, [{7.0}])
        self.run(e, 'fb:cell.2001 fb:cell.2004 or fb:row.row.year '
            '!fb:row.row.regular_season !fb:cell.cell.number N3 diff',
            None, [{1.0, 2.0}])

    def test_superlative(self, executor):
        e = executor
        unary = self.run(e, 'fb:cell.did_not_qualify fb:row.row.playoffs')
        assert len(unary[0]) == 3
        self.run(e, 'x !fb:row.row.index argmin', unary,
                [{'fb:row.r3'}])
        self.run(e, 'x !fb:row.row.index argmax', unary,
                [{'fb:row.r8'}])
        self.run(e, 'x !fb:row.row.avg_attendance !fb:cell.cell.number argmin', unary,
                [{'fb:row.r6'}])
        self.run_error(e, 'x !fb:row.row.next argmax', unary)
        # Another example
        unary = self.run(e, 'type-row')
        assert len(unary[0]) == 10
        self.run(e, 'x fb:row.row.next !fb:row.row.index argmin', unary,
                [{'fb:row.r2'}])
        self.run(e, 'x !fb:row.row.regular_season !fb:cell.cell.number argmin', unary,
                [{'fb:row.r4', 'fb:row.r9'}])
        # Yet another one
        unary = self.run(e, 'type-row !fb:row.row.league')
        assert len(unary[0]) == 3
        self.run(e, 'x fb:row.row.league count argmax', unary,
                [{'fb:cell.usl_first_division'}])

    def test_finalization(self, executor):
        e = executor
        f = executor.finalize
        d = f(self.run(e, 'N2002 N2003 or'))
        assert set(d) == {NumberValue(2002.0), NumberValue(2003.0)}
        d = f(self.run(e, 'N2 fb:row.row.index !fb:row.row.year !fb:cell.cell.date'))
        assert set(d) == {DateValue(2002, -1, -1)}
        d = f(self.run(e, 'type-row !fb:row.row.league'))
        assert set(d) == {
            StringValue('USL A-League'),
            StringValue('USL First Division'),
            StringValue('USSF D-2 Pro League')}
        with pytest.raises(Exception):
            f(self.run(e, 'type-row'))


class TestDates(object):

    @pytest.fixture()
    def graph(self):
        return TablesKnowledgeGraph(os.path.join(DATA_PATH, 'nt-4.graph'))

    @pytest.fixture()
    def executor(self, graph):
        return TablesPostfixExecutor(graph)

    def run(self, executor, lf, old_deno=None, expected_deno=None):
        """Assert the executed denotation and return it."""
        if isinstance(lf, str):
            lf = [Predicate(x) for x in lf.split()]
        try:
            d = executor.execute(lf, old_deno)
            if expected_deno is not None:
                assert list(d) == expected_deno
        except:
            # See what is going wrong in more details
            executor.debug = True
            d = executor.execute(lf, old_deno)
            if expected_deno is not None:
                assert list(d) == expected_deno
        return d

    def test_date_logic(self, executor):
        e = executor
        d = self.run(e, 'Dxx-10-xx')
        assert isinstance(d[0], GenericDateInfiniteSet)
        self.run(e, 'Dxx-10-xx fb:cell.cell.date fb:row.row.date', None,
                [{'fb:row.r10', 'fb:row.r11', 'fb:row.r12', 'fb:row.r13'}])
        self.run(e, 'D1987-xx-xx fb:cell.cell.date fb:row.row.date count', None,
                [{21.0}])
        self.run(e, 'D1987-xx-xx > fb:cell.cell.date fb:row.row.date count', None,
                [{19.0}])
        self.run(e, 'D1987-xx-xx >= fb:cell.cell.date fb:row.row.date count', None,
                [{40.0}])
        self.run(e, 'fb:cell.home fb:row.row.venue x !fb:row.row.date !fb:cell.cell.date argmax', None,
                [{'fb:row.r39'}])


class TestEndToEnd(object):
   
    def run(self, ex_id, formula, target):
        formula = self.expand_shorthands(formula)
        formula = [Predicate(x) for x in formula]
        if not isinstance(target, list):
            target = [target]
        graph = TablesKnowledgeGraph(os.path.join(DATA_PATH, 'nt-{}.graph'.format(ex_id)))
        executor = TablesPostfixExecutor(graph)
        try:
            d = executor.execute(formula)
            d = executor.finalize(d)
            assert check_denotation(d, target)
        except:
            executor.debug = True
            d = executor.execute(formula)
            d = executor.finalize(d)
            assert check_denotation(d, target)

    def expand_shorthands(self, formula):
        formula = formula.split()
        expanded = []
        for x in formula:
            if x.startswith('c.'):
                expanded.append('fb:cell.' + x[2:])
            elif x.startswith('r.'):
                expanded.append('fb:row.row.' + x[2:])
            elif x.startswith('!r.'):
                expanded.append('!fb:row.row.' + x[3:])
            elif x.startswith('p.'):
                expanded.append('fb:cell.cell.' + x[2:])
            elif x.startswith('!p.'):
                expanded.append('!fb:cell.cell.' + x[3:])
            else:
                expanded.append(x)
        return expanded

    def test_end_to_end(self):
        self.run(0,
                'c.usl_a_league r.league x !r.index argmax !r.year !p.number',
                NumberValue(2004))
        self.run(1,
                'c.1st r.position x !r.index argmax !r.venue',
                StringValue('Bangkok, Thailand'))
        self.run(2,
                'c.crettyard r.team !r.next !r.team',
                StringValue('Wolfe Tones'))
        self.run(3,
                'c.united_states_los_angeles r.city !r.passengers !p.number '
                'c.canada_saskatoon r.city !r.passengers !p.number diff',
                NumberValue(12467))
        self.run(4,
                'type-row x !r.date !p.date argmin !r.opponent',
                StringValue('Derby County'))
        self.run(7,
                'c.lake_tuz c.lake_palas_tuzla or x r.name_in_english !r.depth !p.number argmax',
                StringValue('Lake Palas Tuzla'))
        self.run(8,
                'c.full_house r.hand !r.4_credits',
                NumberValue(32))
        self.run(9,
                'c.ardo_kreek != c.ardo_kreek r.player !r.position r.position !r.player and',
                [StringValue('Siim Ennemuist'), StringValue('Andri Aganits')])
        self.run(12,
                'c.matsuyama r.city_town_village count c.imabari r.city_town_village count diff',
                NumberValue(2))
        self.run(14,
                'c.south_korea_kor r.nation N2010 >= p.number r.olympics and !r.athlete',
                StringValue('Kim Yu-Na'))
        self.run(15,
                'N1 p.number r.position !r.venue',
                StringValue('New Delhi, India')),
        # This example shows that using set as intermediate denotation is not sufficient
        #self.run(16,
        #        'c.vs_bc_lions c.at_bc_lions or r.opponent !r.score !p.number sum',
        #        NumberValue(58))
        # This example shows that empty intermediate denotation might still be fine
        #self.run(19,
        #        'N4 > p.number r.score N4 > p.num2 r.score or count',
        #        NumberValue(3))
        self.run(20,
                'type-row !r.album x r.album c.null != r.peak_chart_positions_aus and count argmax',
                StringValue('The Sound Of Trees'))
        self.run(21,
                'type-row x !r.in_service !p.number argmax !r.model',
                StringValue('KM-45 Series'))
        self.run(22,
                'c.auckland r.port x !r.propulsion !p.num2 argmax !r.name',
                StringValue('Manawanui i'))
        self.run(23,
                'type-row !r.nationality x r.nationality count argmin',
                [StringValue('Morocco'), StringValue('France'), StringValue('Spain')])
        self.run(24,
                'c.turkey r.nation !r.next !r.nation',
                StringValue('Sweden'))
        self.run(25,
                'N1800 >= p.number r.founded N1900 < p.number r.founded and count',
                NumberValue(4))
        self.run(25,
                'N1800 >= N1900 < and p.number r.founded count',
                NumberValue(4))
        self.run(28,
                'type-row !r.computer !p.part x p.part r.computer count argmax',
                StringValue('Windows'))
        # Another example showing that using set as intermediate denotation is not sufficient
        #self.run(30,
        #        'c.totals != r.tenure !r.years !p.number avg',
        #        NumberValue(4))
        self.run(35,
                'N24 p.number r.age !r.contestant c.reyna_royo != and',
                StringValue('Marisela Moreno Montero'))
        self.run(37,
                'c.desktop_with_integrated_color_display r.case fb:part.enhanced_keyboard p.part r.notes and count',
                NumberValue(4))
        self.run(49,
                'c.new_zealand r.victor D2010-xx-xx p.date r.date and count',
                NumberValue(3))
        self.run(60,
                'D2010-05-xx p.date r.date count',
                NumberValue(2))
        self.run(60,
                'D2010-05-01 >= D2010-06-01 < and p.date r.date count',
                NumberValue(2))
        self.run(60,
                'D2010-05-01 >= p.date D2010-06-01 < p.date and r.date count',
                NumberValue(2))
