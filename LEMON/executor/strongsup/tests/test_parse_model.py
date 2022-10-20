import math

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_almost_equal

from gtd.ml.framework import Feedable
from gtd.ml.model import TokenEmbedder
from gtd.ml.seq_batch import SequenceBatch
from gtd.ml.utils import guarantee_initialized_variables
from gtd.ml.vocab import SimpleVocab, SimpleEmbeddings
from gtd.tests.ml.test_framework import FeedableTester, clean_test_session
from gtd.utils import Bunch
from strongsup.embeddings import RLongPrimitiveEmbeddings
from strongsup.parse_case import ParseCase
from strongsup.parse_model import PredicateScorer, CrossEntropyLossModel, TrainParseModel, \
    CombinedPredicateEmbedder, Embedder, ExecutionStackEmbedder, DummyStackObjectEmbedder, HistoryEmbedder
from strongsup.rlong.state import RLongObject
from strongsup.tests.utils import PredicateGenerator


@pytest.fixture
def cases():
    context = 'nothing'
    p = PredicateGenerator(context)
    c0 = ParseCase.initial(context, [p('a'), p('c'), p('d')])
    c1 = ParseCase.initial(context, [p('a'), p('b'), p('c'), p('d'), p('e')])
    c2 = ParseCase.initial(context, [])  # empty

    c0.decision = p('d')
    c1.decision = p('c')
    # can't decide for c2, since no options

    return [c0, c1, c2]


class TestCrossEntropyLossModel(FeedableTester):
    @pytest.fixture
    def inputs(self, cases):
        return self.as_args_kwargs(cases[:2])  # don't use the last one, because it has no decision set

    @pytest.fixture
    def logits(self):
        # some made up logit scores
        ninf = float('-inf')
        arr = [
            [1., 2., 3., ninf],
            [1., 2., 3.5, 4.],
        ]
        return np.array(arr)

    @pytest.fixture
    def model(self, logits):
        logits_tensor = tf.constant(logits, dtype=tf.float32)
        return CrossEntropyLossModel(logits_tensor)

    @pytest.fixture
    def feed_dict(self, model):
        return {model._labels: np.array([2, 2])}
        # for both cases, the selected decision is at index 2 of case.choices

    @pytest.fixture
    def outputs(self, logits):
        e = math.exp(1.)
        p0 = e**3 / (e + e**2 + e**3)
        p1 = e**3.5 / (e + e**2 + e**3.5 + e**4)
        probs = np.array([p0, p1])

        nll = -np.log(probs)
        return [nll]

    @pytest.fixture
    def output_tensors(self, model):
        return [model.losses]


class DummyParseModel(Feedable):
    @property
    def logits(self):
        return tf.get_variable('logits', shape=[4], initializer=tf.constant_initializer([1, 1, 1, 1]))
        # not actually used
        # just needed because TrainParseModel needs some variables to optimize.

    def inputs_to_feed_dict(self, *args, **kwargs):
        return {}


class DummyLossModel(Feedable):
    def __init__(self, logits):
        # logits are not actually used to compute loss
        self.losses = tf.get_variable('losses', shape=[4],
                                      initializer=tf.constant_initializer([1, 2, 6, 3.5], dtype=tf.float32))

    def inputs_to_feed_dict(self, *args, **kwargs):
        return {}


class TestTrainParseModel(FeedableTester):
    @pytest.fixture
    def model(self):
        parse_model = DummyParseModel()
        loss_model_factory = lambda logits: DummyLossModel(logits)
        return TrainParseModel(parse_model, loss_model_factory, learning_rate=2.0)

    @pytest.fixture
    def inputs(self):
        # A list of (ParseCase, train_weight) pairs.
        # ParseCases can be None, because DummyLossModel doesn't look at them anyway to produce losses.
        return self.as_args_kwargs([None, None, None, None], [1, 8, 0, 2], caching=False)

    @pytest.fixture
    def feed_dict(self, model):
        return {
            model._weights: np.array([1., 8., 0., 2.])
        }

    @pytest.fixture
    def outputs(self):
        loss = 1 + (2 * 8) + (6 * 0) + (3.5 * 2)
        return [loss]

    @pytest.fixture
    def output_tensors(self, model):
        return [model.loss]


class DummyEmbedder(Embedder):
    """Doesn't actually compute embeddings and vocabs dynamically, but sufficient for testing."""
    def __init__(self, tokens, embeds):
        """

        Args:
            tokens (list[unicode])
            embeds (np.array)
        """
        self.vocab = SimpleVocab(tokens)
        self._embeds = tf.constant(embeds, dtype=tf.float32)
        self._embed_dim = embeds.shape[1]

    @property
    def embeds(self):
        return self._embeds

    def dynamic_vocab(self, batch):
        return self.vocab

    def inputs_to_feed_dict(self, *args, **kwargs):
        return {}


class TestCombinedPredicateEmbedder(FeedableTester):
    @pytest.fixture
    def base_pred_embeddings(self):
        array = np.array([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [0, 2, 0, 8],
        ], dtype=np.float32)
        vocab = SimpleVocab('<unk> b0 b1'.split())
        return SimpleEmbeddings(array, vocab)

    @pytest.fixture
    def model(self, base_pred_embeddings):
        ent_embeds = np.array([
            [10, 20, 30, 40],
            [11, 21, 31, 41],
        ], dtype=np.float32)
        rel_embeds = ent_embeds

        ent_model = DummyEmbedder(['ent0', 'ent1'], ent_embeds)
        rel_model = DummyEmbedder(['rel0', 'rel1'], rel_embeds)
        return CombinedPredicateEmbedder(base_pred_embeddings, ent_model, rel_model)

    @pytest.fixture
    def inputs(self):
        return self.as_args_kwargs([])

    @pytest.fixture
    def feed_dict(self):
        return {}

    @pytest.fixture
    def outputs(self):
        embeds = np.array([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [0, 2, 0, 8],
            [10, 20, 30, 40],
            [11, 21, 31, 41],
            [10, 20, 30, 40],
            [11, 21, 31, 41],
        ], dtype=np.float32)
        return [embeds]

    @pytest.fixture
    def output_tensors(self, model):
        return [model.embeds]


# TODO: This test is obsolete
#class TestHistoryEmbedder(object):
#    @pytest.fixture
#    def model(self):
#        pred_embeds_tensor = tf.constant([
#            [1, 2, 3],
#            [4, 5, 6],
#        ], dtype=tf.float32)
#        class DummyPredicateEmbedder(object):
#            @property
#            def embeds(self):
#                return pred_embeds_tensor
#        pred_embeds = DummyPredicateEmbedder()
#        return HistoryEmbedder(pred_embeds, 3)
#
#    @pytest.fixture
#    def cases(self):
#        pred_names = [
#            ['a', 'b', 'c'],
#            ['c', 'b', 'c', 'd', 'e'],
#            [],
#        ]
#
#        preds = [[Bunch(name=name) for name in name_list] for name_list in pred_names]
#        cases = [Bunch(previous_decisions=pred_list) for pred_list in preds]
#        return cases
#
#    @pytest.mark.usefixtures('clean_test_session')
#    def test_cases_to_histories(self, model, cases):
#        histories = model._cases_to_histories(cases)
#        assert histories == {
#            0: ['a', 'b', 'c'],
#            1: ['c', 'd', 'e'],
#            2: [],
#        }


class TestPredicateScorer(object):
    @pytest.fixture
    def model(self):
        ninf = -float('inf')
        simple_scores = tf.constant([
            [1, 2, 3, ninf],
            [4, 5, ninf, ninf],
            [1, 1, 2, 2]
        ], dtype=tf.float32)
        soft_copy_scores = tf.constant([
            [8, -2, 10, 0],
            [0, 1, 0, 0],
            [11, 0.5, 1.4, -1.6],
        ], dtype=tf.float32)

        mask = tf.constant([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ], dtype=tf.float32)

        # scores don't actually depend on cases
        simple_scorer = Bunch(scores=SequenceBatch(simple_scores, mask), inputs_to_feed_dict=lambda cases: {})
        soft_copy_scorer = Bunch(scores=SequenceBatch(soft_copy_scores, mask), inputs_to_feed_dict=lambda cases: {})
        return PredicateScorer(simple_scorer, soft_copy_scorer)

    @pytest.fixture
    def cases(self):
        context = 'nothing'
        p = PredicateGenerator(context)
        c0 = ParseCase.initial(context, [p('a'), p('c'), p('d')])
        c1 = ParseCase.initial(context, [p('a'), p('b')])
        c2 = ParseCase.initial(context, [p('a'), p('b'), p('d'), p('c')])  # empty
        return [c0, c1, c2]

    @pytest.fixture
    def correct_scores(self):
        ninf = -float('inf')
        return np.array([
            [9, 0, 13, ninf],
            [4, 6, ninf, ninf],
            [12, 1.5, 3.4, 0.4]
        ], dtype=np.float32)

    @pytest.mark.usefixtures('clean_test_session')
    def test(self, model, cases, correct_scores):
        scores = model.compute(model.scores.values, cases)
        assert_array_almost_equal(correct_scores, scores)


class DummyRLongObject(RLongObject):
    pass


class TestExecutionStackEmbedder(object):
    @pytest.fixture
    def model(self):
        max_stack_size = 3
        max_list_size = 7
        primitive_embed_dim = 6
        object_embed_dim = 10
        primitive_embeddings = RLongPrimitiveEmbeddings(primitive_embed_dim)
        primitive_embedder = TokenEmbedder(primitive_embeddings, 'primitive_embeds', trainable=True)
        object_embedder = DummyStackObjectEmbedder(max_stack_size, object_embed_dim)
        return ExecutionStackEmbedder(primitive_embedder, object_embedder, max_stack_size, max_list_size,
                                      project_object_embeds=True, abstract_objects=False)

    @pytest.fixture
    def cases(self):
        make_case = lambda stack: Bunch(_previous_cases=[Bunch(denotation=Bunch(execution_stack=stack))])
        some_obj = DummyRLongObject()
        empty_list = []

        return [
            make_case(['r', -1]),
            make_case(['X1/1']),
            make_case(['b', some_obj, empty_list]),
        ]

    @pytest.mark.usefixtures('clean_test_session')
    def test_inputs_to_feed_dict(self, model, cases):
        feed = model.inputs_to_feed_dict(cases)
        assert_array_almost_equal(
            feed[model._primitive_indices.values],
            np.array([
                [0, 2, 19],
                [0, 0, 20],
                [7, 0, 1],
            ], dtype=np.float32)
        )
        assert_array_almost_equal(
            feed[model._primitive_indices.mask],
            np.array([
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
            ], dtype=np.float32)
        )

        assert_array_almost_equal(
            feed[model._is_object_feed.values],
            np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 1],
            ], dtype=np.float32)
        )

    @pytest.mark.usefixtures('clean_test_session')
    def test(self, model, cases):
        sess = tf.get_default_session()
        guarantee_initialized_variables(sess)
        embeds = model.compute(model.embeds, cases)
        primitive_embeddings = RLongPrimitiveEmbeddings(6)

        # compute object embedding after applying projection
        object_projection_layer = model._object_projection_layer
        W, b = object_projection_layer.get_weights()  # shapes [10, 6] and [6]
        object_embed = np.ones(10).dot(W) + b

        assert_array_almost_equal(embeds[0],
                                  np.concatenate((np.zeros(6), primitive_embeddings['r'], primitive_embeddings[-1]))
                                  )

        assert_array_almost_equal(embeds[1],
                                  np.concatenate((np.zeros(6), np.zeros(6), primitive_embeddings['X1/1']))
                                  )

        assert_array_almost_equal(embeds[2],
                                  np.concatenate((primitive_embeddings['b'], object_embed, object_embed))
                                  )


class TestHistoryEmbedder(object):
    def test_previous_decisions_for_this_utterance(self):
        prev_cases = [Bunch(current_utterance_idx=1, decision='a'), Bunch(current_utterance_idx=1, decision='b'),
                      Bunch(current_utterance_idx=2, decision='c'), Bunch(current_utterance_idx=2, decision='d')]

        case = Bunch(current_utterance_idx=2, _previous_cases=prev_cases)

        prev_decisions = HistoryEmbedder.previous_decisions_for_this_utterance(case)

        assert prev_decisions == ['c', 'd']

        bad_cases = [Bunch(current_utterance_idx=2, decision='a'), Bunch(current_utterance_idx=1, decision='b'),
                      Bunch(current_utterance_idx=2, decision='c'), Bunch(current_utterance_idx=2, decision='d')]

        bad_case = Bunch(current_utterance_idx=2, _previous_cases=bad_cases)
        with pytest.raises(AssertionError):
            _ = HistoryEmbedder.previous_decisions_for_this_utterance(bad_case)
