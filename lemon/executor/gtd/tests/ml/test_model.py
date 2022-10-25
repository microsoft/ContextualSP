import copy
import numpy as np
import pytest
import tensorflow as tf
from math import exp
from numpy.testing import assert_array_almost_equal

from gtd.ml.model import TokenEmbedder, MeanSequenceEmbedder, ConcatSequenceEmbedder, CandidateScorer, LSTMSequenceEmbedder, \
    SoftCopyScorer, Attention, BidiLSTMSequenceEmbedder
from gtd.ml.seq_batch import FeedSequenceBatch, SequenceBatch
from gtd.ml.utils import clean_session
from gtd.ml.vocab import SimpleVocab, SimpleEmbeddings
from gtd.tests.ml.test_framework import FeedableTester, clean_test_session
from gtd.utils import softmax


class VocabExample(SimpleVocab):
    def __init__(self, tokens, unk):
        if unk not in tokens:
            raise ValueError('unk must be in tokens')
        self.unk = unk
        super(VocabExample, self).__init__(tokens)

    def word2index(self, w):
        try:
            return self._word2index[w]
        except KeyError:
            return self._word2index[self.unk]


class TestTokenEmbedder(FeedableTester):
    @pytest.fixture
    def model(self):
        array = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 5, 7],
        ], dtype=np.float32)
        vocab = SimpleVocab('a b c'.split())
        embeddings = SimpleEmbeddings(array, vocab)
        return TokenEmbedder(embeddings, 'token_embeds')

    @pytest.fixture
    def inputs(self):
        return self.as_args_kwargs()

    @pytest.fixture
    def feed_dict(self):
        return {}

    @pytest.fixture
    def output_tensors(self, model):
        return [model.embeds]

    @pytest.fixture
    def outputs(self):
        array = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 5, 7],
        ], dtype=np.float32)
        return [array]


class TestSequenceEmbedder(FeedableTester):
    @pytest.fixture
    def model(self):
        token_embeds = tf.constant([
            [0, 0, 0],
            [1, 2, 3],
            [2, 4, 6],
            [3, 5, 7],
            [9, 9, 9],
        ], dtype=tf.float32)
        return MeanSequenceEmbedder(token_embeds)

    @pytest.fixture
    def inputs(self):
        token_vocab = SimpleVocab(['<pad>'] + 'a b c d'.split())
        sequences = [
            ['a', 'c'],
            ['b', 'c', 'c'],
            ['d', 'c', 'a'],
        ]
        return self.as_args_kwargs(sequences, token_vocab)

    @pytest.fixture
    def feed_dict(self, model):
        indices_tensor = model._sequence_batch.values
        mask_tensor = model._sequence_batch.mask

        pad = 0
        indices = [
            [1, 3, pad],
            [2, 3, 3],
            [4, 3, 1]
        ]

        mask = [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ]

        return {indices_tensor: np.array(indices), mask_tensor: np.array(mask)}

    @pytest.fixture
    def output_tensors(self, model):
        return [model.embeds]

    @pytest.fixture
    def outputs(self):
        npa = lambda arr: np.array(arr, dtype=np.float32)
        embeds = npa([
            npa([4, 7, 10]) / 2,
            npa([8, 14, 20]) / 3,
            npa([13, 16, 19]) / 3,
        ])
        return [embeds]


class TestConcatSequenceEmbedder(object):
    def test(self):
        token_vocab = SimpleVocab('a b c d'.split())
        sequences = [
            ['a', 'b', 'c', 'd'],
            ['c', 'd'],
        ]

        correct_embeds = np.array([
            [1, 2, 0, 3, 4, 1, 5, 6, 0, 7, 8, 1],
            [5, 6, 0, 7, 8, 1, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

        with clean_session():
            token_embeds = tf.constant([
                [1, 2, 0],
                [3, 4, 1],
                [5, 6, 0],
                [7, 8, 1],
            ], dtype=tf.float32)
            model = ConcatSequenceEmbedder(token_embeds)
            test_embeds = model.compute(model.embeds, sequences, token_vocab)

        assert_array_almost_equal(correct_embeds, test_embeds, decimal=5)


class TestFixedLengthConcatEmbedder(object):
    def test(self):
        token_vocab = SimpleVocab('a b c d'.split())
        sequences = [
            ['a', 'b', 'c', 'd'],
            ['c', 'd'],
        ]

        correct_embeds = np.array([
            [3, 4, 1, 5, 6, 0, 7, 8, 1],
            [0, 0, 0, 5, 6, 0, 7, 8, 1]
        ], dtype=np.float32)

        with clean_session():
            token_embeds = tf.constant([
                [1, 2, 0],
                [3, 4, 1],
                [5, 6, 0],
                [7, 8, 1],
            ], dtype=tf.float32)
            model = ConcatSequenceEmbedder(token_embeds, seq_length=3, align='right')
            test_embeds = model.compute(model.embeds, sequences, token_vocab)

            # check that static shape inference works
            assert model.embeds.get_shape().as_list() == [None, 3 * 3]

        assert_array_almost_equal(correct_embeds, test_embeds, decimal=5)


class TestCandidateScorer(FeedableTester):
    @pytest.fixture
    def query(self):
        # a batch size of three. Each row is a query vector
        return np.array([
            [2., 2., 4.],
            [1., 2., 0.],
            [1., 2., 8.],
        ], dtype=np.float32)

    @pytest.fixture
    def embeddings(self):
        array = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
        ], dtype=np.float32)

        vocab = SimpleVocab(['<pad>', 'a', 'b', 'c', 'd', 'e'])
        return SimpleEmbeddings(array, vocab)

    @pytest.fixture
    def projection_weights(self):
        W = np.random.normal(size=[3, 3])
        b = np.random.normal(size=[3])
        return W, b

    @pytest.fixture
    def model(self, query, embeddings, projection_weights):
        candidate_embedder = TokenEmbedder(embeddings, 'cand_embeds')
        scorer = CandidateScorer(tf.constant(query, dtype=tf.float32), candidate_embedder.embeds)
        scorer.projection_weights = projection_weights
        return scorer

    @pytest.fixture
    def inputs(self, embeddings):
        candidates = [
            ['a', 'c', 'd'],
            ['a', 'b', 'c', 'd', 'e'],
            []
        ]
        vocab = embeddings.vocab
        return self.as_args_kwargs(candidates, vocab)

    @pytest.fixture
    def feed_dict(self, model):
        values = model._cand_batch.values
        mask = model._cand_batch.mask

        choice_indices = np.array([
            [1, 3, 4, 0, 0],
            [1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0],
        ], dtype=np.int32)
        t, o = True, False
        choice_mask = np.array([
            [t, t, t, o, o],
            [t, t, t, t, t],
            [o, o, o, o, o],
        ])

        return {
            values: choice_indices,
            mask: choice_mask,
        }

    @pytest.fixture
    def output_tensors(self, model):
        return [model.scores.values, model._probs.values]

    @pytest.fixture
    def outputs(self, query, embeddings, model, feed_dict, projection_weights):
        # project the query tensor
        W, b = projection_weights
        query = query.dot(W) + b

        embeds = embeddings.array

        choice_embeds0 = embeds[[1, 3, 4]]
        query0 = query[0]
        logits0 = np.array(list(choice_embeds0.dot(query0)) + 2 * [float('-inf')])

        choice_embeds1 = embeds[[1, 2, 3, 4, 5]]
        query1 = query[1]
        logits1 = choice_embeds1.dot(query1)

        logits2 = np.array([1., 1., 1., 1., 1.]) * float('-inf')

        logits = [logits0, logits1, logits2]
        probs = [softmax(l) for l in logits]

        logits = np.array(logits)
        probs = np.array(probs)

        return [logits, probs]


class TestSoftCopyScorer(object):
    @pytest.fixture
    def model(self):
        attention_weights = tf.constant([
            [0.1, 0.5, 10., 0., 0],
            [0.1, 0.7, -10, 0., 1],
            [8.0, 0.3, 0.0, 11, 2],
        ], dtype=tf.float32)
        return SoftCopyScorer(attention_weights)

    @pytest.fixture
    def alignments(self):
        return [
            [[(0, 0.5), (2, 0.5)], [(2, 3.)], [(4, 10.), (0, 10.)]],
            [[(0, 0.), (1, 1.), (2, 2.), (4, 4.)]],
            [[(4, -1.), (3, -2.)]],
        ]

    @pytest.fixture
    def correct_scores(self):
        return np.array([
            [5.05, 30, 1],
            [-15.3, 0, 0],
            [-24, 0, 0],
        ], dtype=np.float32)

    @pytest.mark.usefixtures('clean_test_session')
    def test(self, model, alignments, correct_scores):
        scores = model.compute(model.scores.values, alignments)
        assert_array_almost_equal(correct_scores, scores)
        assert len(scores.shape) == 2

    @pytest.mark.usefixtures('clean_test_session')
    def test_out_of_bounds(self, model, alignments, correct_scores):
        bad_alignments = copy.deepcopy(alignments)
        bad_alignments[0][0][0] = (5, -1)  # one index beyond seq_length

        with pytest.raises(ValueError):
            scores = model.compute(model.scores.values, bad_alignments)


class TestLSTMSequenceEmbedder(object):

    def test_lstm(self):
        """Test whether the mask works properly for LSTM embedder."""
        token_vocab = SimpleVocab('a b c d'.split())
        sequences = [
            ['a', 'b', 'c', 'd'],
            ['c', 'd'],
            ['a', 'b', 'c', 'd'],
        ]
        sequences_alt = [
            ['a', 'b', 'c', 'd', 'a', 'b', 'd', 'c'],
            ['b', 'a', 'd'],
            ['c', 'd'],
        ]

        with clean_session():
            token_embeds = tf.constant([
                [1, 2, 0],
                [3, 4, 1],
                [5, 6, 0],
                [7, 8, 1],
            ], dtype=tf.float32)

            model = LSTMSequenceEmbedder(token_embeds, seq_length=4, hidden_size=7)
            test_embeds, test_hidden_states = model.compute(
                    [model.embeds, model.hidden_states.values],
                    sequences, token_vocab)
            assert test_embeds.shape == (3, 7)
            assert test_hidden_states.shape == (3, 4, 7)
            # Padded spaces should have the same hidden states
            assert_array_almost_equal(test_hidden_states[1,1,:], test_hidden_states[1,2,:], decimal=5)
            assert_array_almost_equal(test_hidden_states[1,1,:], test_hidden_states[1,3,:], decimal=5)

            # Try again but with different paddings
            # Should get the same result for ['c', 'd']
            big_model = LSTMSequenceEmbedder(token_embeds, seq_length=8, hidden_size=7)
            big_model.weights = model.weights  # match weights

            test_embeds_alt, test_hidden_states_alt = big_model.compute(
                    [big_model.embeds, big_model.hidden_states.values],
                    sequences_alt, token_vocab)
            assert test_embeds_alt.shape == (3, 7)
            assert test_hidden_states_alt.shape == (3, 8, 7)

        assert_array_almost_equal(test_embeds[1,:], test_embeds_alt[2,:], decimal=5)
        assert_array_almost_equal(test_hidden_states[1,:2,:],
                test_hidden_states_alt[2,:2,:], decimal=5)


class TestBidiLSTMSequenceEmbedder(object):

    def test_lstm(self):
        """Test whether the mask works properly for bidi LSTM embedder."""
        token_vocab = SimpleVocab('a b c d'.split())
        sequences = [
            ['a', 'b', 'c', 'd'],
            ['c', 'd'],
            ['a', 'b', 'c', 'd'],
        ]
        sequences_alt = [
            ['a', 'b', 'c', 'd', 'a', 'b', 'd', 'c'],
            ['b', 'a', 'd'],
            ['c', 'd'],
        ]

        with clean_session():
            token_embeds = tf.constant([
                [1, 2, 0],
                [3, 4, 1],
                [5, 6, 0],
                [7, 8, 1],
            ], dtype=tf.float32)

            model = BidiLSTMSequenceEmbedder(token_embeds, seq_length=4, hidden_size=7)
            test_embeds, test_hidden_states = model.compute(
                    [model.embeds, model.hidden_states.values],
                    sequences, token_vocab)
            assert test_embeds.shape == (3, 14)
            assert test_hidden_states.shape == (3, 4, 14)
            assert_array_almost_equal(test_embeds[1,:7], test_hidden_states[1,1,:7], decimal=5)
            assert_array_almost_equal(test_embeds[1,7:], test_hidden_states[1,0,7:], decimal=5)
            # Padded spaces should have the same forward embeddings
            assert_array_almost_equal(test_hidden_states[1,1,:7], test_hidden_states[1,2,:7], decimal=5)
            assert_array_almost_equal(test_hidden_states[1,1,:7], test_hidden_states[1,3,:7], decimal=5)
            # Padded spaces should have 0 backward embeddings
            assert_array_almost_equal(np.zeros((7,)), test_hidden_states[1,2,7:], decimal=5)
            assert_array_almost_equal(np.zeros((7,)), test_hidden_states[1,3,7:], decimal=5)
            # Other spaces should not have 0 embeddings with very high probability
            assert np.linalg.norm(test_hidden_states[1,0,:7]) > 1e-5
            assert np.linalg.norm(test_hidden_states[1,1,:7]) > 1e-5
            assert np.linalg.norm(test_hidden_states[1,0,7:]) > 1e-5
            assert np.linalg.norm(test_hidden_states[1,1,7:]) > 1e-5

            # Try again but with different paddings
            # Should get the same result for ['c', 'd']
            big_model = BidiLSTMSequenceEmbedder(token_embeds, seq_length=8, hidden_size=7)
            big_model.weights = model.weights  # match weights

            test_embeds_alt, test_hidden_states_alt = big_model.compute(
                    [big_model.embeds, big_model.hidden_states.values],
                    sequences_alt, token_vocab)
            assert test_embeds_alt.shape == (3, 14)
            assert test_hidden_states_alt.shape == (3, 8, 14)

        assert_array_almost_equal(test_embeds[1,:], test_embeds_alt[2,:], decimal=5)
        assert_array_almost_equal(test_hidden_states[1,:2,:],
                test_hidden_states_alt[2,:2,:], decimal=5)


class TestAttention(object):
    @pytest.fixture
    def memory_cells(self):
        # (batch_size, num_cells, cell_dim)
        values = tf.constant([  # (2, 2, 3)
            [
              [1., 2., 3.],
              [1., 1., 1.]
            ],
            [
              [1., 1.5, 0.],
              [-0.8, 1., -0.4]
            ]
        ], dtype=tf.float32)

        mask = tf.constant([  # (2, 2)
            [1, 0],
            [1, 1],
        ], dtype=tf.float32)

        return SequenceBatch(values, mask)

    @pytest.fixture
    def query(self):
        # (batch_size, cell_dim)
        return tf.constant([  # (2, 3)
            [1., 2., -1.5],
            [0., 0.3, 2.]
        ], dtype=tf.float32)

    @pytest.fixture
    def model(self, memory_cells, query):
        return Attention(memory_cells, query)

    @pytest.fixture
    def correct_logits(self):
        ninf = -float('inf')
        return np.array([
            [(1 + 4 + -4.5), ninf],
            [(0 + 0.45 + 0), (0 + 0.3 + -0.8)]
        ], dtype=np.float32)

    @pytest.fixture
    def correct_probs(self):
        normalizer = exp(0.45) + exp(-0.5)
        return np.array([
            [1.0, 0.0],
            [exp(0.45) / normalizer, exp(-0.5) / normalizer]
        ], dtype=np.float32)

    @pytest.fixture
    def correct_retrieved(self, correct_probs):
        a0 = correct_probs[1][0]
        a1 = correct_probs[1][1]
        weighted = a0 * np.array([1., 1.5, 0.]) + \
                   a1 * np.array([-0.8, 1., -0.4])

        return np.array([
            [1., 2., 3.],
            weighted,
        ], dtype=np.float32)

    @pytest.mark.usefixtures('clean_test_session')
    def test(self, model, correct_logits, correct_probs, correct_retrieved):
        sess = tf.get_default_session()
        logits, probs, retrieved = sess.run([model.logits, model.probs, model.retrieved])
        assert_array_almost_equal(correct_logits, logits)
        assert_array_almost_equal(correct_probs, probs)
        assert_array_almost_equal(correct_retrieved, retrieved)