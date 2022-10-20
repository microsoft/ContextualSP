"""Data structures for the tables domain.

We represent denotations with various Python data structures.

Possible denotation types include:
- Unary = set of Things
        | InfiniteSet
- ScopedBinary = dict {Object: Unary, ...} (the domain must be finite)
- Relation = string (Used when the relation is mentioned before the entity)
where
- Thing = string (NameValue)
        | float (NumberValue)
        | Date (DateValue)
- InfiniteSet = NeqInfiniteSet [e.g., (!= obama)]
              | RangeInfiniteSet [e.g., (> 9000), (and (>= 4) (< 5))]
              | GenericDateInfiniteSet [e.g., (date 2001 7 -1) = July 2001]
"""
import os
import re
import sys
from collections import Container as ContainerABC


################################
# Thing

def parse_number(x):
    """Parse a number from a string."""
    return round(float(x), 6)


class Date(object):
    """A date consisting of a year, a month, and a day.
    Some but not all fields can be absent (using placeholder -1)
    """

    def __init__(self, year, month, day):
        if year == -1 and month == -1 and day == -1:
            raise ValueError('Invalid date (-1 -1 -1)')
        self.year = year
        self.month = month
        self.day = day
        assert month == -1 or 1 <= month <= 12, 'Invalid month: {}'.format(month)
        assert day == -1 or 1 <= day <= 31, 'Invalid day: {}'.format(day)
        self._hash = hash((self.year, self.month, self.day))

    def __str__(self):
        return 'Date({}, {}, {})'.format(self.year, self.month, self.day)
    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Date):
            return False
        return (self.year == other.year
                and self.month == other.month
                and self.day == other.day)

    def __hash__(self):
        return self._hash

    def __ne__(self, other):
        return not (self == other)

    def __cmp__(self, other):
        if not isinstance(other, Date):
            raise ValueError('Cannot compare Date to {}'.format(type(other)))
        if self.year == other.year or self.year == -1 or other.year == -1:
            if self.month == other.month or self.month == -1 or other.month == -1:
                if self.day == other.day or self.day == -1 or other.day == -1:
                    return 0
                return cmp(self.day, other.day)
            return cmp(self.month, other.month)
        return cmp(self.year, other.year)


def parse_date(x):
    """Parse a date from a string with format yy-mm-dd."""
    x = x.split('-')
    assert len(x) == 3, 'Not a valid date: {}'.format(x)
    year = -1 if x[0][0].lower() == 'x' else int(x[0])
    month = -1 if x[1][0].lower() == 'x' else int(x[1])
    day = -1 if x[2][0].lower() == 'x' else int(x[2])
    return Date(year, month, day)


def parse_value(x):
    """Parse the string, which may be a number, a date, or a non-numeric string."""
    try:
        return parse_number(x)
    except:
        try:
            return parse_date(x)
        except:
            return x


################################
# Type processing

def get_type(x):
    """Return the type signature of x. Used to prevent comparison across types."""
    if isinstance(x, float):
        # NumberValue
        return 'N'
    elif isinstance(x, Date):
        # DateValue
        return 'D'
    elif isinstance(x, str):
        # NameValue: take the fb:xxx part of fb:xxx.yyy
        if not x.startswith('fb:'):
            raise ValueError('NameValue does not start with "fb:": {}'.format(x))
        tokens = x.split('.')
        if len(tokens) != 2:
            raise ValueError('{} is not an entity'.format(x))
        return tokens[0]
    else:
        raise ValueError('Unknown type for {}'.format(type(x)))


def ensure_same_type(collection, allowed_types=None):
    """Ensure that all values in the collection have the same type.
    Return the agreed type. Throw an error if the type is not agreed.

    Args:
        collection: A set or a dict where values are sets.
        allowed_types: Restriction on the agreed type.
            Can be a string, a collection of strings, or None (= allow all).
    Returns:
        The agreed type
    Throws:
        ValueError if one of the following happens:
        - The collection is not a set or a set-valued dict
        - The collection is empty
        - Some two items have different types
        - Some item does not agree with the allowed types (if specified)
    """
    if isinstance(collection, set):
        itr = iter(collection)
    elif isinstance(collection, dict):
        # Iterate over all items in values
        itr = (x for v in collection.values() for x in v)
    else:
        raise ValueError('Bad data type: {}'.format(type(collection)))
    if allowed_types and isinstance(allowed_types, str):
        allowed_types = [allowed_types]
    agreed_type = None
    for value in itr:
        if agreed_type is None:
            agreed_type = get_type(value)
            if allowed_types is not None and agreed_type not in allowed_types:
                raise ValueError('Type {} is not in allowed types {}'\
                        .format(agreed_type, allowed_types))
        else:
            t = get_type(value)
            if t != agreed_type:
                raise ValueError('Value {} does not have agreed type {}'\
                        .format(value, agreed_type))
    if agreed_type is None:
        raise ValueError('The collection is empty: {}'.format(collection))
    return agreed_type


################################
# InfiniteSet

class InfiniteSet(ContainerABC):
    """An abstract class representing an infinite set of items."""

    def __and__(self, stuff):
        if isinstance(stuff, set):
            return {x for x in stuff if x in self}
        raise NotImplementedError

    def __rand__(self, stuff):
        return self & stuff


THINGS = (str, float, Date)
COMPARABLES = (float, Date)
COMPUTABLES = (float,)
UNARIES = (set, InfiniteSet)
BINARIES = (dict,)


class NeqInfiniteSet(InfiniteSet):
    """Represent (!= xxx).

    Note that the semantics of (!= xxx) is
    "things that are not xxx but have the same type as xxx"
    """

    def __init__(self, value):
        assert isinstance(value, THINGS), 'Invalid value for !=: {}'.format(value)
        self.value = value
        self.value_type = get_type(value)

    def __eq__(self, other):
        return (isinstance(other, NeqInfiniteSet)
                and self.value == other.value)

    def __hash__(self):
        return 12345 + hash(self.value)

    def __repr__(self):
        return '{{ != {} }}'.format(self.value)

    def __contains__(self, x):
        # Need to type check
        return x != self.value and get_type(x) == self.value_type


class RangeInfiniteSet(InfiniteSet):
    """Represent ranges like (> xxx) or (and (> xxx) (< yyy)).

    xxx, yyy can be numbers or dates, but the types must agree.
    """

    def __init__(self, sign, value, sign2=None, value2=None):
        self.left_sign = self.left_value = None
        self.right_sign = self.right_value = None
        assert isinstance(value, COMPARABLES), \
                'Invalid value for comparison: {}'.format(value)
        self.value_type = get_type(value)
        if sign in ('>', '>='):
            self.left_sign = sign
            self.left_value = value
        elif sign in ('<', '<='):
            self.right_sign = sign
            self.right_value = value
        else:
            raise NotImplementedError(sign)
        if sign2 is not None:
            assert self.value_type == get_type(value2), \
                    'Invalid value for comparison: {}'.format(value2)
            if sign2 in ('>', '>='):
                assert self.left_sign is None
                self.left_sign = sign2
                self.left_value = value2
            elif sign2 in ('<', '<='):
                assert self.right_sign is None
                self.right_sign = sign2
                self.right_value = value2
            else:
                raise NotImplementedError(sign2)

    def __eq__(self, other):
        return (isinstance(other, RangeInfiniteSet)
                and self.left_sign == other.left_sign
                and self.right_sign == other.right_sign
                and self.left_value == other.left_value
                and self.right_value == other.right_value)

    def __hash__(self):
        return hash((self.left_sign, self.right_sign, self.left_value, self.right_value))

    def __repr__(self):
        if self.left_sign is None:
            return '{{ {} {} }}'.format(self.right_sign, self.right_value)
        if self.left_sign is None:
            return '{{ {} {} }}'.format(self.left_sign, self.left_value)
        else:
            return '{{ {} {} ; {} {} }}'.format(
                    self.left_sign, self.left_value,
                    self.right_sign, self.right_value)

    def __contains__(self, x):
        if self.value_type != get_type(x):
            return False
        if self.left_sign:
            if ((self.left_sign == '>' and x <= self.left_value)
                    or (self.left_sign == '>=' and x < self.left_value)):
                return False
        if self.right_sign:
            if ((self.right_sign == '<' and x >= self.right_value)
                    or (self.right_sign == '<=' and x > self.right_value)):
                return False
        return True

    def __and__(self, stuff):
        try:
            return super(RangeInfiniteSet, self).__and__(stuff)
        except NotImplementedError:
            if isinstance(stuff, RangeInfiniteSet):
                # ULTIMATE RANGE MERGE!!!
                assert self.value_type == stuff.value_type,\
                        'Incompatible types: {} and {}'.format(self.value_type, stuff.value_type)
                # Left
                if (not self.left_sign
                        or (stuff.left_sign and (
                            stuff.left_value > self.left_value
                            or (stuff.left_value == self.left_value
                                and stuff.left_sign == '>')))):
                    new_left_sign = stuff.left_sign
                    new_left_value = stuff.left_value
                else:
                    new_left_sign = self.left_sign
                    new_left_value = self.left_value
                # Right
                if (not self.right_sign
                        or (stuff.right_sign and (
                            stuff.right_value < self.right_value
                            or (stuff.right_value == self.right_value
                                and stuff.right_sign == '<')))):
                    new_right_sign = stuff.right_sign
                    new_right_value = stuff.right_value
                else:
                    new_right_sign = self.right_sign
                    new_right_value = self.right_value
                # Return value
                if not new_left_sign:
                    if not new_right_sign:
                        return set()
                    return RangeInfiniteSet(new_right_sign, new_right_value)
                elif not new_right_sign:
                    return RangeInfiniteSet(new_left_sign, new_left_value)
                if new_left_value > new_right_value:
                    return set()
                elif new_left_value == new_right_value:
                    if new_left_sign == '>' or new_right_sign == '<':
                        return set()
                    return {new_left_value}
                else:
                    return RangeInfiniteSet(new_left_sign, new_left_value,
                            new_right_sign, new_right_value)


class GenericDateInfiniteSet(InfiniteSet):
    """Represent a generic date where a year, month, or day is left unspecified.

    For example, (-1 7 -1) represents the set of all dates with month = 7,
    and (1990 7 -1) is the set of dates with year = 1990 and month = 7.
    """

    def __init__(self, date):
        self.year = date.year
        self.month = date.month
        self.day = date.day

    def __eq__(self, other):
        return (isinstance(other, GenericDateInfiniteSet)
                and self.year == other.year
                and self.month == other.month
                and self.day == other.day)

    def __hash__(self):
        return hash((self.year, self.month, self.day))

    def __repr__(self):
        return '{{ {} {} {} }}'.format(self.year, self.month, self.day)

    def __contains__(self, x):
        if not isinstance(x, Date):
            return False
        return (self.year in (-1, x.year)
                and self.month in (-1, x.month)
                and self.day in (-1, x.day))

    def min_(self):
        # Note that the returned value might not be a valid date.
        # The value is nevertheless ok for comparison.
        if self.day != -1:      # ....-..-07
            return Date(self.year, self.month, self.day)
        if self.month != -1:    # ....-07-xx
            return Date(self.year, self.month, 1)
        if self.year != -1:     # 1907-xx-xx
            return Date(self.year, 1, 1)

    def max_(self):
        if self.day != -1:      # ....-..-07
            return Date(self.year, self.month, self.day)
        if self.month != -1:    # ....-07-xx
            if self.month == 2:
                if self.year % 4 == 0 and self.year % 400 != 0:
                    num_days = 29
                else:
                    num_days = 28
            elif self.month in (1, 3, 5, 7, 8, 10, 12):
                num_days = 31
            else:
                num_days = 30
            return Date(self.year, self.month, num_days)
        if self.year != -1:     # 1907-xx-xx
            return Date(self.year, 12, 31)
