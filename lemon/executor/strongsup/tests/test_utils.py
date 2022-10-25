
import math
import operator
from numpy.testing import assert_allclose

from strongsup.utils import (
        epsilon_greedy_sample,
        softmax, softmax_with_alpha_beta,
        )
from functools import reduce


def test_epsilon_greedy_sample():
    num_choices = 8
    num_iters = 100000
    to_sample = 4
    epsilon = 0.9
    def expected_count():
        expected_count = epsilon * (num_choices - 1)/num_choices
        expected_count *= reduce(
                operator.mul,
                (1 - epsilon * 1/(num_choices - num) for num in range(
                    1, to_sample)),
                1)
        expected_count = (1 - expected_count) * num_iters
        return expected_count

    choices = list(range(num_choices))
    counts = [0] * (num_choices + 1)
    for i in range(num_iters):
        sample = epsilon_greedy_sample(choices, to_sample, epsilon)
        for val in sample:
            counts[val] += 1
    expected = expected_count()
    assert(0.98 * expected <= counts[1] <= 1.02 * expected)
    
#test_epsilon_greedy_sample()


def test_softmax():
    stuff = [-1, -2, -3, -20, -400]
    exped = [math.exp(x) for x in stuff]
    target = [x / sum(exped) for x in exped]
    assert_allclose(target, softmax(stuff))
    assert_allclose(target, softmax_with_alpha_beta(stuff, 1, 1))

def test_softmax_with_alpha_beta():
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            for stuff in [
                    [-1, -2, -3, -20, -400],
                    [-30, -30.4, -30.2, -31],
                    [float('-inf'), -30.4, -30.2, float('-inf'), float('-inf'), -31]]:
                exped = [math.exp(x) for x in stuff]
                exped_with_beta = [math.exp(x * beta) if x != float('-inf') else 0. for x in stuff]
                target = [x / sum(exped_with_beta) * sum(exped)**(1-alpha) for x in exped_with_beta]
                actual = softmax_with_alpha_beta(stuff, alpha, beta)
                assert_allclose(target, actual)
