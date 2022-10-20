from unittest import TestCase
from os.path import join

import pytest
from gtd.text import PhraseMatcher
from gtd.utils import FileMemoized, SimpleExecutor, as_batches, Failure, NestedDict, EqualityMixinSlots, \
    memoize_with_key_fxn, DictMemoized


def test_as_batches():
    items = [0, 1, 2, 3, 4, 5, 6]
    assert list(as_batches(items, 2)) == [[0, 1], [2, 3], [4, 5], [6]]


def test_file_memoized_represent_args(tmpdir):
    path = str(tmpdir.join('fxn'))

    fm = FileMemoized(None, path, None, None)
    key = fm._cache_key(['a', 'b'], {'c': 2, 'd': 'e'})
    assert key == join(path, 'a_b_c=2_d=e.txt')
    key = fm._cache_key([], {'c': 2, 'd': 'e'})
    assert key == join(path, 'c=2_d=e.txt')
    key = fm._cache_key([], dict())
    assert key == join(path, 'NO_KEY.txt')


class TestUtils(TestCase):

    def test_phrase_matcher(self):
        phrases = [[1, 2, 3], [1, ], [2, ], [2, 4]]
        not_phrases = [[1, 2], [4, ]]

        pm = PhraseMatcher(phrases)

        for phrase in phrases:
            self.assertTrue(pm.has_phrase(phrase))

        for phrase in not_phrases:
            self.assertFalse(pm.has_phrase(phrase))

        tokens = [1, 2, 1, 2, 3, 2, 3, 2, 4]

        matches = pm.match(tokens)

        correct = [((1,), 0, 1),
                   ((2,), 1, 2),
                   ((1,), 2, 3),
                   ((2,), 3, 4),
                   ((1, 2, 3), 2, 5),
                   ((2,), 5, 6),
                   ((2,), 7, 8),
                   ((2, 4), 7, 9)]

        self.assertEqual(matches, correct)


class TestSimpleExecutor(object):

    def test_context_manager(self):
        fxn = lambda x: 2 * x
        with SimpleExecutor(fxn, max_workers=2) as ex:
            for i, x in enumerate(range(10)):
                ex.submit(i, x)
            results = {k: v for k, v in ex.results()}

        correct = {k: 2 * k for k in range(10)}
        assert results == correct


class TestFailure(object):
    def test_eq(self):
        f0 = Failure()
        f1 = Failure()
        f2 = Failure(uid=1)
        f3 = Failure(uid=1, message='different message')
        assert f0 != f1  # different id
        assert f1 != f2  # different id
        assert f2 == f3  # same id


class TestNestedDict(object):
    @pytest.fixture
    def normal_dict(self):
        return {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3,
            },
        }

    @pytest.fixture
    def nested_dict(self, normal_dict):
        return NestedDict(normal_dict)

    def test_as_dict(self, nested_dict, normal_dict):
        assert nested_dict.as_dict() == normal_dict

    def test_iter(self, nested_dict):
        assert set(nested_dict) == {'a', 'b'}

    def test_len(self, nested_dict):
        assert len(nested_dict) == 3

    def test_nested(self):
        d = NestedDict()
        d.set_nested(('a', 'b', 'c'), 1)
        d.set_nested(('a', 'd'), 2)

        assert d.as_dict() == {
            'a': {
                'b': {
                    'c': 1
                },
                'd': 2,
            }
        }
        assert d.get_nested(('a', 'd')) == 2

        with pytest.raises(KeyError):
            d.get_nested(('a', 'd', 'e'))

    def test_leaves(self, nested_dict):
        assert set(nested_dict.leaves()) == {1, 2, 3}


class DummySlotsObject(EqualityMixinSlots):
    __slots__ = ['a', 'b', 'c']

    def __init__(self, a, b, c=None):
        self.a = a
        self.b = b

        if c:
            self.c = c


class TestEqualityMixinSlot(object):
    def test_equality(self):
        d1 = DummySlotsObject(5, 10)
        d2 = DummySlotsObject(5, 10)
        assert d1 == d2

        d3 = DummySlotsObject(5, 10, 20)
        d4 = DummySlotsObject(5, 11)
        assert d1 != d3
        assert d1 != d4


class MemoizedClass(object):
    def __init__(self):
        self.calls = 0

    @memoize_with_key_fxn(lambda self, a, b: b)  # key fxn only uses b
    def fxn_to_memoize(self, a, b):
        self.calls += 1
        return a + b


class MemoizedClass2(object):
    def __init__(self):
        self.calls = 0

    def fxn(self, a, b):
        self.calls += 1
        return a + b

    fxn_memoized = DictMemoized(fxn)


class TestDictMemoized(object):
    def test(self):
        mc = MemoizedClass2()
        result = mc.fxn_memoized('a', 'b')
        assert result == 'ab'
        assert mc.calls == 1

        result2 = mc.fxn_memoized('a', 'b')
        assert result2 == 'ab'
        assert mc.calls == 1

        result2 = mc.fxn_memoized('b', 'b')
        assert result2 == 'bb'
        assert mc.calls == 2


class TestMemoizeWithKey(object):
    def test_caching(self):
        mc = MemoizedClass()
        result = mc.fxn_to_memoize('hey', 'there')
        assert mc.calls == 1
        assert result == 'heythere'

        # returns cached result
        result2 = mc.fxn_to_memoize('hey', 'there')
        assert result2 == 'heythere'
        assert mc.calls == 1

        # computes new result
        result3 = mc.fxn_to_memoize('hey', 'what')
        assert mc.calls == 2

        # only caches on 2nd arg, 'there', not 'you'
        result4 = mc.fxn_to_memoize('you', 'there')
        assert result4 == 'heythere'
        assert mc.calls == 2