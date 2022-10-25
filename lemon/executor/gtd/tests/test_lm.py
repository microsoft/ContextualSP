from collections import Counter
from numpy.testing import assert_approx_equal
from gtd.lm import last_k, CountLM, LMSampler, normalize_counts
import pytest


@pytest.fixture
def lm():
    return CountLM(3)


@pytest.fixture
def lm_sampler(lm):
    return LMSampler(lm)


def test_last_k():
    tokens = [1, 2, 3, 4]
    assert last_k(tokens, 2) == (3, 4)
    assert last_k(tokens, 4) == (1, 2, 3, 4)
    assert last_k(tokens, 0) == tuple()


def test_get_contexts(lm):
    tokens = [1, 2, 3, 4, 5]
    assert list(lm._get_contexts(tokens)) == [tuple(), (5,), (4, 5), (3, 4, 5)]

    assert list(lm._get_contexts([1, 2])) == [tuple(), (2,), (1, 2)]


def test_largest_known_context(lm):
    contexts = {tuple(), (3,), (2, 3), (1, 2)}
    assert lm._largest_context([1, 2, 3], contexts) == (2, 3)
    assert lm._largest_context([2, 3, 0], contexts) == tuple()


def test_normalize_counts():
    c = Counter([1, 1, 2, 2, 3])
    assert normalize_counts(c) == Counter({1: .4, 2: .4, 3: .2})


@pytest.mark.skip
def test_sample_from_distribution(lm_sampler):
    distr = {'a': 0.3, 'b': 0.7}
    ctr = Counter()
    # law of large numbers test
    for k in range(100000):
        ctr[lm_sampler._sample_from_distribution(distr)] += 1
    empirical = normalize_counts(ctr)
    for key in list(distr.keys()) + list(empirical.keys()):
        assert_approx_equal(empirical[key], distr[key], significant=2)

def test_sequence_probability(lm):
    lm = CountLM(3)
    lines = ['apple pear banana', 'pear banana apple', 'banana pear banana']
    for line in lines:
        tokens = line.split()
        lm.record_counts(tokens, append_end=True)

    probs = lm.sequence_probability(['pear', 'apple', 'pear'])
    assert probs == [('pear', 0.3333333333333333), ('apple', 0.0), ('pear', 0.5)]