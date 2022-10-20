import pytest
import math
import tensorflow as tf
import numpy as np
from numpy.testing import assert_almost_equal

from gtd.utils import Bunch
from strongsup.example import Context
from strongsup.decoder import Decoder, DecoderConfig
from strongsup.predicate import Predicate
from strongsup.utils import EOS
from strongsup.tests.utils import PredicateGenerator, softmax


class DummyParseModel(object):
    def __init__(self, logits):
        self.logits = logits

    def score(self, cases):
        for case in cases:
            case.choice_logits = self.logits[:len(case.choices)]
            case.choice_probs = softmax(self.logits[:len(case.choices)])


class DummyExecutor(object):

    def execute(self, y_toks, old_denotation=None):
        new_denotation = (old_denotation or []) + [x.name for x in y_toks]
        # Let's disallow some sequences
        if new_denotation == ['a', 'c']:
            raise ValueError
        return new_denotation


class DummyContext(Context):
    def __init__(self):
        self._table_path = None
        self._utterance = None
        self._executor = DummyExecutor()
        self._predicates = None

    @property
    def predicates(self):
        return self._predicates


class TestSimpleDecode(object):
    @pytest.fixture
    def context(self):
        context = DummyContext()
        p = PredicateGenerator(context)
        context._predicates = [p('a'), p('b'), p('c'), p(EOS)]
        return context

    @pytest.fixture
    def parse_model(self):
        return DummyParseModel([0, math.log(2), math.log(4), math.log(3)])

    @pytest.fixture
    def config(self):
        return DecoderConfig(10, 3)

    @pytest.fixture
    def decoder(self, parse_model, config):
        caching = False
        return Decoder(parse_model, None, None, caching, config)

    def test_initial_beam(self, decoder, context):
        beam = decoder.initial_beam(context)
        assert len(beam) == 1
        assert len(beam[0]) == 0
        assert beam[0].context == context

    def test_advance(self, decoder, context):
        beam = decoder.initial_beam(context)
        new_beams = decoder.advance([beam])
        assert len(new_beams) == 1
        new_beam = new_beams[0]
        assert len(new_beam) == 4
        ranked = [' '.join([y.name for y in x.decisions]) for x in new_beam]
        assert ranked == ['c', EOS, 'b', 'a']

    def test_advance_twice(self, config, context):
        logits = [math.log(1), math.log(2), math.log(4), math.log(3)]
        parse_model = DummyParseModel(logits)
        caching = False
        decoder = Decoder(parse_model, None, None, caching, config)
        beam = decoder.initial_beam(context)
        new_beams = decoder.advance([beam])
        parse_model.logits = [math.log(5), math.log(2), math.log(7), math.log(6)]
        new_beams = decoder.advance(new_beams)
        assert len(new_beams) == 1
        new_beam = new_beams[0]
        assert len(new_beam) == 10
        ranked = [' '.join([y.name for y in x.decisions]) for x in new_beam]
        assert ranked == [
                EOS, 'c c', 'c ' + EOS, 'c a', 'b c',
                'b ' + EOS, 'b a', 'c b', 'a ' + EOS, 'a a']

    def test_normalized_path_probs(self):
        beam = [Bunch(prob=0.01), Bunch(prob=0.5), Bunch(prob=0.2)]
        assert_almost_equal(Decoder._normalized_path_probs(beam), [1./71, 50./71, 20./71], decimal=5)

    # TODO Test predictions and train_step
