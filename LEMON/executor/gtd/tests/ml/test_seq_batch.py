import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal
from tensorflow.python.framework.errors import InvalidArgumentError

from gtd.ml.seq_batch import SequenceBatch, FeedSequenceBatch, reduce_mean, reduce_max, reduce_sum
from gtd.ml.utils import clean_session
from gtd.ml.vocab import SimpleVocab
from gtd.tests.ml.test_framework import FeedableTester, assert_array_collections_equal, clean_test_session
from gtd.tests.ml.test_model import VocabExample


class TestSequenceBatch(object):
    def test(self):
        values = tf.constant([
            [1, -8, 5],
            [0, 2, 7],
            [2, -8, 6],
        ], dtype=tf.float32)

        float_mask = tf.constant([
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ], dtype=tf.float32)

        bool_mask = tf.constant([
            [True, True, True],
            [False, False, True],
            [True, True, False],
        ], dtype=tf.bool)

        ninf = float('-inf')
        correct = np.array([
            [1, -8, 5],
            [ninf, ninf, 7],
            [2, -8, ninf],
        ], dtype=np.float32)

        seq_batch0 = SequenceBatch(values, float_mask)
        seq_batch1 = SequenceBatch(values, bool_mask)

        with tf.Session():
            assert_almost_equal(seq_batch0.with_pad_value(ninf).values.eval(), correct)
            assert_almost_equal(seq_batch1.with_pad_value(ninf).values.eval(), correct)


class TestFeedSequenceBatch(FeedableTester):
    @pytest.fixture
    def model(self):
        return FeedSequenceBatch(align='left')

    @pytest.fixture
    def inputs(self):
        tokens = '<unk> a b c'.split()
        unk = '<unk>'
        vocab = VocabExample(tokens, unk)
        sequences = [
            'a a b b c'.split(),
            'a b'.split(),
            ['b'],
            ['c'],
        ]
        return self.as_args_kwargs(sequences, vocab)

    @pytest.fixture
    def feed_dict(self, model):
        indices = np.array([
            [1, 1, 2, 2, 3],
            [1, 2, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ], dtype=np.int32)

        mask = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ], dtype=np.float32)

        return {model.values: indices, model.mask: mask}

    def test_outputs(self):
        pass  # trivial to test placeholders

    def test_right_align(self, inputs):
        indices = np.array([
            [1, 1, 2, 2, 3],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 3],
        ], dtype=np.int32)

        mask = np.array([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ], dtype=np.float32)

        with clean_session():
            model = FeedSequenceBatch(align='right')
            correct = {model.values: indices, model.mask: mask}

            args, kwargs = inputs
            test = model.inputs_to_feed_dict(*args, **kwargs)
            assert_array_collections_equal(correct, test)

    def test_seq_length(self):
        tokens = '<unk> a b c'.split()
        unk = '<unk>'
        vocab = VocabExample(tokens, unk)
        sequences = [
            'a b a b c'.split(),  # more than length 4
            'a b'.split(),
            ['b'],
            ['c'],
        ]

        indices = np.array([
            [2, 1, 2, 3],
            [0, 0, 1, 2],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
        ], dtype=np.int32)

        mask = np.array([
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        with clean_session():
            model = FeedSequenceBatch(align='right', seq_length=4)
            test_feed = model.inputs_to_feed_dict(sequences, vocab)
            correct = {model.values: indices, model.mask: mask}
            assert_array_collections_equal(correct, test_feed)

            indices = tf.identity(model.values)
            mask = tf.identity(model.mask)
            assert indices.get_shape().as_list() == [None, 4]
            assert mask.get_shape().as_list() == [None, 4]

    def test_no_sequences(self):
        vocab = SimpleVocab('a b c'.split())
        sequences = []

        with clean_session():
            model = FeedSequenceBatch()
            indices = tf.identity(model.values)
            mask = tf.identity(model.mask)
            indices_val, mask_val = model.compute([indices, mask], sequences, vocab)
            assert indices_val.shape == mask_val.shape == (0, 0)


class TestReduceMean(object):
    def test_multidim(self):
        npa = lambda arr: np.array(arr, dtype=np.float32)
        correct = npa([
            npa([4, 7, 10]) / 2,
            npa([8, 14, 20]) / 3,
            npa([13, 16, 19]) / 3,
        ])

        with clean_session():
            array = tf.constant([[[1., 2., 3.],
                                  [3., 5., 7.],
                                  [0., 0., 0.]],
                                 [[2., 4., 6.],
                                  [3., 5., 7.],
                                  [3., 5., 7.]],
                                 [[9., 9., 9.],
                                  [3., 5., 7.],
                                  [1., 2., 3.]]], dtype=tf.float32)
            mask = tf.constant([
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
            ], dtype=tf.float32)

            bm = reduce_mean(SequenceBatch(array, mask))
            assert_almost_equal(bm.eval(), correct, decimal=5)

    def test_batch_mean(self):
        correct = np.array([-2. / 3, 1., 21. / 4])

        with clean_session():
            array = tf.constant([
                [1, -8, 5, 4, 9],
                [0, 2, 7, 8, 1],
                [2, -8, 6, 4, 9],
            ], dtype=tf.float32)

            mask = tf.constant([
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 1, 1],
            ], dtype=tf.float32)

            bad_mask = tf.constant([
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1],
            ], dtype=tf.float32)

            bm = reduce_mean(SequenceBatch(array, mask))
            assert_almost_equal(bm.eval(), correct, decimal=5)

            bm2 = reduce_mean(SequenceBatch(array, bad_mask))

            with pytest.raises(InvalidArgumentError):
                bm2.eval()

            # try allow_empty option
            bm3 = reduce_mean(SequenceBatch(array, bad_mask), allow_empty=True)
            assert_almost_equal(bm3.eval(), np.array([-2. / 3, 0., 21. / 4]))

    def test_empty(self):
        with clean_session():
            array = tf.constant(np.empty((0, 10, 20)))
            mask = tf.constant(np.empty((0, 10)))
            bm = reduce_mean(SequenceBatch(array, mask))
            assert bm.eval().shape == (0, 20)


class TestReduceMax(object):
    def test(self):
        npa = lambda arr: np.array(arr, dtype=np.float32)
        correct = npa([
            npa([3, 5, 7]),
            npa([3, 5, 7]),
            npa([9, 9, 9]),
        ])

        with clean_session():
            array = tf.constant([[[1., 2., 3.],
                                  [3., 5., 7.],
                                  [100., 200., 2000.]],
                                 [[2., 4., 6.],
                                  [3., 5., 7.],
                                  [3., 5., 7.]],
                                 [[9., 9., 9.],
                                  [3., 5., 7.],
                                  [1., 2., 3.]]], dtype=tf.float32)
            mask = tf.constant([
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
            ], dtype=tf.float32)

            bm = reduce_max(SequenceBatch(array, mask))
            assert_almost_equal(bm.eval(), correct, decimal=5)

            bad_mask = tf.constant([
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
            ], dtype=tf.float32)

            bm2 = reduce_mean(SequenceBatch(array, bad_mask))

            with pytest.raises(InvalidArgumentError):
                bm2.eval()


class TestReduceSum(object):
    def test(self):
        correct = np.array([-2, 2, 21])

        with clean_session():
            array = tf.constant([
                [1, -8, 5, 4, 9],
                [0, 2, 7, 8, 1],
                [2, -8, 6, 4, 9],
            ], dtype=tf.float32)

            mask = tf.constant([
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 1, 1],
            ], dtype=tf.float32)

            result = reduce_sum(SequenceBatch(array, mask))
            assert_almost_equal(result.eval(), correct, decimal=5)