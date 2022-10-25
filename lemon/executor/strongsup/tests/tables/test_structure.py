import pytest

from strongsup.tables.structure import (
        parse_number, parse_date, parse_value, Date,
        get_type, ensure_same_type,
        NeqInfiniteSet, RangeInfiniteSet, GenericDateInfiniteSet,
        )


class TestValues(object):

    def test_date(self):
        assert Date(2012, 12, -1) == Date(2012, 12, -1)
        assert len({Date(-1, 4, 14), Date(-1, 4, 14)}) == 1
        with pytest.raises(Exception):
            Date(-1, -1, -1)
        with pytest.raises(Exception):
            Date(1990, 0, 12)
        with pytest.raises(Exception):
            Date(1990, 4, 32)
        assert Date(2012, 8, -1) < Date(2012, 12, 4)
        # Not sure if this is the behavior we want ...
        assert Date(-1, 8, 24) < Date(2012, 8, 29)
        with pytest.raises(Exception):
            # Cannot compare across types
            Date(1984, -1, -1) > 1985.0

    def test_parse_value(self):
        assert parse_number('2.3') == 2.3
        assert parse_number('-4') == -4
        with pytest.raises(Exception):
            parse_number('3.45m')
        assert parse_date('1961-08-04') == Date(1961, 8, 4)
        assert parse_date('XXXX-12-xx') == Date(-1, 12, -1)
        with pytest.raises(Exception):
            parse_date('xx-xx-xx')
        assert parse_value('10') == 10.0
        assert parse_value('-3.14') == -3.14
        assert parse_value('xx-8-24') == Date(-1, 8 ,24)
        assert parse_value('40 kg') == '40 kg'
        assert parse_value('xx-xx-xx') == 'xx-xx-xx'

    def test_get_type(self):
        assert get_type(4.0) == 'N'
        assert get_type(Date(-1, -1, 2)) == 'D'
        assert get_type('fb:cell.puppy') == 'fb:cell'
        with pytest.raises(Exception):
            get_type('argmax')
        with pytest.raises(Exception):
            get_type('fb:row.row.name')

    def test_ensure_same_type(self):
        assert ensure_same_type({4.0}) == 'N'
        assert ensure_same_type({'fb:cell.puppy': {4.0}, 'fb:cell.kitten': {6.0, 7.0}}) == 'N'
        assert ensure_same_type({Date(2010, 1, 2): {4.0}, 'fb:cell.kitten': {6.0, 7.0}}) == 'N'
        assert ensure_same_type({4.0, 5.0, 20.0, -2.5}) == 'N'
        assert ensure_same_type({4.0, 5.0, 20.0, -2.5}, 'N') == 'N'
        assert ensure_same_type({4.0, 5.0, 20.0, -2.5}, ['D', 'N']) == 'N'
        assert ensure_same_type({Date(-1, 11, 14), Date(-1, 12, 3)}) == 'D'
        assert ensure_same_type({'fb:cell.puppy', 'fb:cell.kitten'}) == 'fb:cell'
        assert ensure_same_type({'fb:cell.puppy', 'fb:cell.kitten'}, 'fb:cell') == 'fb:cell'
        assert ensure_same_type({x: {(x*1.)**y for y in range(x)} for x in [2, 3, 5, 7]}) == 'N'
        assert ensure_same_type({x: {'fb:hello.' + str(y) for y in range(x)} for x in [2, 3, 5, 7]}) == 'fb:hello'
        with pytest.raises(ValueError):
            ensure_same_type('4.0')
        with pytest.raises(ValueError):
            ensure_same_type(set())
        with pytest.raises(ValueError):
            ensure_same_type(set(), 'N')
        with pytest.raises(ValueError):
            ensure_same_type({4.0: set(), 5.0: set()}, 'D')
        with pytest.raises(ValueError):
            ensure_same_type({4.0: {5.0}, 6.0: {2.0, 'fb:cell.kitten'}})
        with pytest.raises(ValueError):
            ensure_same_type({'fb:row.row.name'})
        with pytest.raises(ValueError):
            ensure_same_type({2.25, 4.6, -5}, 'D')
        with pytest.raises(ValueError):
            ensure_same_type({'fb:part.puppy': {1.2}, 'fb:cell.kitten': {2.4}}, ['D', 'fb:cell'])


class TestInfiniteSet(object):

    def test_neq(self):
        a = NeqInfiniteSet(3.0)
        assert 3.0 not in a
        assert 6.0 in a
        assert Date(2010, 1, 2) not in a
        assert 'fb:cell.puppy' not in a
        a = NeqInfiniteSet(Date(2010, 1, 2))
        assert 3.0 not in a
        assert Date(2010, 1, 2) not in a
        assert Date(2010, -1, 2) in a
        assert 'fb:cell.puppy' not in a
        a = NeqInfiniteSet('fb:cell.puppy')
        assert 'fb:cell.puppy' not in a
        assert 'fb:cell.kitten' in a
        assert 'fb:part.robot' not in a

    def test_neq_and(self):
        assert NeqInfiniteSet(3.0) & {3.0, 4.0, Date(2010, 1, 2)} == {4.0}
        assert {3.0, 4.0, Date(2010, 1, 2)} & NeqInfiniteSet(3.0) == {4.0}
        assert NeqInfiniteSet(Date(2010, -1, 2)) & \
                {3.0, 4.0, Date(2010, 1, 2), Date(2010, -1, 2), Date(2010, -1, -1)} == \
                {Date(2010, 1, 2), Date(2010, -1, -1)}

    def test_basic_range(self):
        a = RangeInfiniteSet('>', 4.0)
        assert 2.0 not in a
        assert 4.0 not in a
        assert 8.0 in a
        assert Date(2010, -1, -1) not in a
        a = RangeInfiniteSet('>=', 4.0)
        assert 2.0 not in a
        assert 4.0 in a
        assert 8.0 in a
        a = RangeInfiniteSet('<', 4.0)
        assert 2.0 in a
        assert 4.0 not in a
        assert 8.0 not in a
        a = RangeInfiniteSet('<=', 4.0)
        assert 2.0 in a
        assert 4.0 in a
        assert 8.0 not in a
        a = RangeInfiniteSet('>', 4.0, '<=', 8.0)
        assert 2.0 not in a
        assert 4.0 not in a
        assert 6.0 in a
        assert 8.0 in a
        assert 10.0 not in a
        assert 'fb:cell.obama' not in a
    
    def test_date_range(self):
        a = RangeInfiniteSet('>', Date(2010, 2, 14), '<=', Date(2011, 12, 1))
        assert Date(2010, 2, 13) not in a
        assert Date(2010, 2, 14) not in a
        assert Date(2010, 2, 15) in a
        assert Date(2010, 3, 1) in a
        assert Date(2011, 2, 1) in a
        assert Date(2011, 12, 1) in a
        assert Date(2012, 5, 7) not in a

    def test_range_and(self):
        a = RangeInfiniteSet('<', 4.0)
        b = RangeInfiniteSet('<', 1.0)
        c = a & b
        assert 0.0 in c
        assert 1.0 not in c
        assert 4.0 not in c
        assert a & {0.0, 1.0, 4.0, 'fb:cell.puppy'} == {0.0, 1.0}
        assert {0.0, 1.0, 4.0, 'fb:cell.puppy'} & a == {0.0, 1.0}
        a = RangeInfiniteSet('>=', 4.0, '<', 10.0)
        b = RangeInfiniteSet('<', 7.0, '>=', 2.0)
        c = a & b
        assert 2.0 not in c
        assert 4.0 in c
        assert 6.0 in c
        assert 7.0 not in c
        assert 10.0 not in c
        a = RangeInfiniteSet('>', 4.0)
        b = RangeInfiniteSet('<', 4.0)
        assert a & b == set()
        a = RangeInfiniteSet('>=', 4.0)
        b = RangeInfiniteSet('<=', 4.0)
        assert a & b == {4.0}
        a = RangeInfiniteSet('>=', 4.0)
        b = RangeInfiniteSet('<', 4.0)
        assert a & b == set()
        a = RangeInfiniteSet('>', 4.0, '<', 8.0)
        b = RangeInfiniteSet('<', 4.0)
        assert a & b == set()
        a = RangeInfiniteSet('>=', 4.0, '<=', 8.0)
        b = RangeInfiniteSet('<=', 4.0)
        assert a & b == {4.0}

    def test_generic_date(self):
        a = GenericDateInfiniteSet(Date(2010, 4, -1))
        assert Date(2010, 4, 2) in a
        assert Date(2010, 5, 3) not in a
        assert Date(2010, -1, -1) not in a
        assert 4.0 not in a
        assert a.min_() == Date(2010, 4, 1)
        assert a.max_() == Date(2010, 4, 30)
        a = GenericDateInfiniteSet(Date(-1, 4, 20))
        assert Date(2010, 4, 20) in a
        assert Date(2010, 5, 20) not in a
        assert Date(-1, 4, -1) not in a
        assert 4.0 not in a
        assert a.min_() == a.max_() == Date(-1, 4, 20)

    def test_generic_date_and(self):
        a = GenericDateInfiniteSet(Date(-1, 4, -1))
        assert a & {Date(2010, 4, 2), Date(2010, 5, 3), Date(2011, 4, 7)} == \
                {Date(2010, 4, 2), Date(2011, 4, 7)}
        assert {Date(2010, 4, 2), Date(2010, 5, 3), Date(2011, 4, 7)} & a== \
                {Date(2010, 4, 2), Date(2011, 4, 7)}
