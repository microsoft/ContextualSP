import copy
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from strongsup.parse_case import ParseCase, ParsePath
from strongsup.predicate import Predicate
from strongsup.tests.utils import PredicateGenerator, softmax


class ParseCaseTester(object):
    def test_previous_decisions(self, case, previous_decisions):
        assert case.previous_decisions == previous_decisions

    def test_eq(self, case, equal_case, diff_case):
        assert case == equal_case
        assert case != diff_case

    def test_no_dict(self, case):
        with pytest.raises(AttributeError):
            case.__dict__

    def test_set_once(self, case, decision, logits):
        p = PredicateGenerator(case.context)
        c3 = ParseCase.extend(case, [p('a'), p('b')])
        c3.decision = p('b')
        c3.choice_logits = [1.0, 2.0]

        assert c3.decision == p('b')
        assert c3.choice_logits == [1.0, 2.0]

        with pytest.raises(RuntimeError):
            c3.decision = p('a')
        with pytest.raises(RuntimeError):
            c3.choice_logits = [3.0, 4.0]

    def test_previous_cases(self, case, previous_cases):
        for c1, c2 in zip(case._previous_cases, previous_cases):
            assert c1 == c2

    def test_path(self, case, path):
        assert case.path == path


class BasicTestCase(object):
    @pytest.fixture
    def context(self):
        return 'some context'

    @pytest.fixture
    def predicate_generator(self, context):
        return PredicateGenerator(context)

    @classmethod
    def create_cases(cls, context):
        p = PredicateGenerator(context)

        c0 = ParseCase.initial(context, [p('a'), p('b'), p('c')])
        c0.decision = p('b')
        c0.choice_logits = [1., 2., 3.]
        c0.choice_probs = softmax(c0.choice_logits)

        c1 = ParseCase.extend(c0, [p('c'), p('d'), p('e')])
        c1.decision = p('e')
        c1.choice_logits = [1., 2., 3.]
        c1.choice_probs = softmax(c1.choice_logits)

        c2 = ParseCase.extend(c1, [p('f'), p('g')])
        c2.decision = p('f')
        c2.choice_logits = [5., 6.]
        c2.choice_probs = softmax(c2.choice_logits)

        return [c0, c1, c2]


class TestRecursiveParseCase(ParseCaseTester, BasicTestCase):
    @pytest.fixture
    def cases(self, context):
        return self.create_cases(context)

    @pytest.fixture
    def previous_cases(self, cases):
        return cases[:-1]

    @pytest.fixture
    def path(self, cases):
        return ParsePath(cases)

    @pytest.fixture
    def case(self, cases):
        return cases[-1]  # last case

    @pytest.fixture
    def equal_case(self, context):
        cases = self.create_cases(context)
        return cases[-1]  # just like case

    @pytest.fixture
    def diff_case(self, context):
        cases = self.create_cases(context)
        return cases[0]

    @pytest.fixture
    def previous_decisions(self, predicate_generator):
        p = predicate_generator
        return [p('b'), p('e')]

    @pytest.fixture
    def decision(self, predicate_generator):
        p = predicate_generator
        return p('f')

    @pytest.fixture
    def logits(self):
        return [5.0, 6.0]

    def test_previous_decided(self, case, predicate_generator):
        p = predicate_generator
        c1 = ParseCase.extend(case, [p('1'), p('2')])

        with pytest.raises(RuntimeError):
            # didn't set a decision on c1
            ParseCase.extend(c1, [p('3'), p('4')])

    def test_copy(self, case, predicate_generator):
        p = predicate_generator
        c = copy.copy(case)
        assert c.decision == p('f')
        assert c.choice_logits == [5., 6.]
        assert c == case


class TestParsePath(BasicTestCase):
    @pytest.fixture
    def cases(self, context):
        return TestRecursiveParseCase.create_cases(context)

    @pytest.fixture
    def case(self, cases):
        return cases[-1]

    def test_decisions(self, case, predicate_generator):
        p = predicate_generator
        assert case.path.decisions == [p('b'), p('e'), p('f')]

    def test_prob(self, case):
        e = math.exp
        assert_allclose(case.prob, e(5) / (e(5) + e(6)))
        path_prob = (
                e(2) / (e(1) + e(2) + e(3))
                * e(3) / (e(1) + e(2) + e(3))
                * e(5) / (e(5) + e(6)))
        assert_allclose(case.cumulative_prob, path_prob)
        assert_allclose(case.path.prob, path_prob)

    def test_prob_some_more(self):
        empty_path = ParsePath([], context='hello')
        assert empty_path.prob == 1.
