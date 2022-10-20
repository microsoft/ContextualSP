from unittest import TestCase
import os

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tensorflow.python.framework.errors import InvalidArgumentError

from gtd.ml.utils import TensorDebugger, clean_session, expand_dims_for_broadcast, broadcast, Saver, \
    guarantee_initialized_variables, gather_2d
from gtd.tests.ml.test_framework import clean_test_session


class TestTensorDebugger(TestCase):
    def test_tensor_debugger_deps(self):
        tdb = TensorDebugger()

        x = tf.constant(3, name='x')
        z = tf.mul(x, 3, name='z')
        with tf.control_dependencies([x]):
            y = tf.constant(8, name='y')

        deps = tdb.dependency_graph

        # control dependencies depend on x's output
        self.assertEqual(deps['y'], {'x:0'})

        # each output depends on its op
        self.assertEqual(deps['y:0'], {'y'})

        # downstream ops depend on the output of earlier ops
        self.assertTrue('x:0' in deps['z'])

    def test_tensor_debugger_multiple(self):
        tdb = TensorDebugger()

        x = tf.constant([1, 2])
        tdb.register('x', x)
        zs = []
        for k in range(3):
            y = tf.constant(k)
            z = tf.reduce_sum(x * y)
            # register multiple nodes under the same name
            tdb.register('y', y)
            zs.append(z)

        # 0, (1 + 2), (2 + 4)
        final = tf.pack(zs)

        with tf.Session() as sess:
            results, bp_results = tdb.debug(sess, final, {})

            def test(a, b):
                self.assertTrue(np.array_equal(a, b))

            # result correctly passed back
            test(results, [0, 3, 6])
            # values in for loop accumulated as list
            test(bp_results['y'], [0, 1, 2])

    def test_tensor_debugger_exec_path(self):
        tdb = TensorDebugger()

        x = tf.constant(5, name='x')
        y = tf.placeholder(tf.int32, name='y')

        z = tf.mul(x, y, 'z')
        w = tf.constant(4, name='w')

        f = tf.mul(z, w, 'f')
        g = tf.constant(3, name='g')

        with tf.control_dependencies([f]):
            h = tf.constant(11, name='h')

        # don't register x
        tdb.register('y', y)
        tdb.register('z', z)
        tdb.register('w', w)
        tdb.register('f', f)
        tdb.register('g', g, force_run=True)
        tdb.register('h', h)

        with tf.Session() as sess:
            result, bp_results = tdb.debug(sess, f, {y: 2})
            # result is a single value, not a list
            self.assertEqual(result, 40)
            # excludes x, because not registered. excludes h, because not on execution path.
            # includes g, because of force_run
            self.assertEqual(bp_results, {'y': 2, 'z': 10, 'w': 4, 'g': 3})

            results, bp_results = tdb.debug(sess, [h, g], {y: 2})
            # returns a list
            self.assertEqual(results, [11, 3])
            # includes y, z, w and f because h depends on them through control_dependencies
            # includes g because of force_run
            self.assertEqual(bp_results, {'y': 2, 'z': 10, 'f': 40, 'w': 4, 'g': 3})


def test_expand_dims_for_broadcast():
    with clean_session():
        arr = tf.constant([
            [
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
            ],
        ], dtype=tf.float32)
        weights = tf.constant([1, 2], dtype=tf.float32)

        assert arr.get_shape().as_list() == [2, 3, 3]
        assert weights.get_shape().as_list() == [2]

        new_weights = expand_dims_for_broadcast(weights, arr)
        assert new_weights.eval().shape == (2, 1, 1)

        bad_weights = tf.constant([1, 2, 3], dtype=tf.float32)
        bad_new_weights = expand_dims_for_broadcast(bad_weights, arr)

        with pytest.raises(InvalidArgumentError):
            bad_new_weights.eval()


class TestGather2D(object):
    @pytest.fixture
    def x(self):
        x = tf.constant([
            [[1, 2], [2, 2], [3, 3]],
            [[4, 5], [5, 4], [6, 6]],
            [[7, 7], [8, 7], [9, 9]],
            [[0, 8], [1, 1], [2, 2]]
        ], dtype=tf.int32)
        return x

    @pytest.mark.usefixtures('clean_test_session')
    def test(self, x):
        i = tf.constant([[0, 2],
                         [3, 0]],
                        dtype=tf.int32)
        j = tf.constant([[1, 1],
                         [0, 2]],
                        dtype=tf.int32)
        vals = gather_2d(x, i, j)

        correct = np.array([
            [[2, 2], [8, 7]],
            [[0, 8], [3, 3]],
        ], dtype=np.int32)

        assert_array_almost_equal(correct, vals.eval())

        assert vals.get_shape().as_list() == [2, 2, 2]

    @pytest.mark.usefixtures('clean_test_session')
    def test_broadcast(self, x):
        i = tf.constant([[0, 2],
                         [3, 0]],
                        dtype=tf.int32)
        j = tf.constant([[1, 2]], dtype=tf.int32)  # needs to be broadcast up
        vals = gather_2d(x, i, j)

        correct = np.array([
            [[2, 2], [9, 9]],
            [[1, 1], [3, 3]],
        ], dtype=np.int32)

        assert_array_almost_equal(correct, vals.eval())


def test_broadcast():
    with clean_session():
        values = tf.constant([
            [
                [1, 2],
                [1, 2],
            ],
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ]
        ], dtype=tf.float32)

        mask = tf.constant([
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=tf.float32)

        correct = np.array([
            [
                [1, 1],
                [0, 0],
            ],
            [
                [1, 1],
                [1, 1],
            ],
            [
                [0, 0],
                [1, 1],
            ]
        ], dtype=np.float32)

        assert values.get_shape().as_list() == [3, 2, 2]
        assert mask.get_shape().as_list() == [3, 2]

        mask = expand_dims_for_broadcast(mask, values)
        assert mask.get_shape().as_list() == [3, 2, 1]

        mask = broadcast(mask, values)
        assert mask.get_shape().as_list() == [3, 2, 2]

        mask_val = mask.eval()

        assert_array_equal(mask_val, correct)


class TestSaver(object):
    @pytest.fixture
    def v(self):
        return tf.get_variable('v', shape=[], initializer=tf.constant_initializer(5))

    @pytest.mark.usefixtures('clean_test_session')
    def test_restore(self, tmpdir, v):
        save_100_path = str(tmpdir.join('weights-100'))
        save_10_path = str(tmpdir.join('weights-10'))

        saver = Saver(str(tmpdir))
        assign_op = tf.assign(v, 12)

        sess = tf.get_default_session()
        guarantee_initialized_variables(sess)

        assert v.eval() == 5
        saver.save(100)  # save as step 100

        sess.run(assign_op)
        assert v.eval() == 12
        saver.save(10)  # save as step 10

        saver.restore()  # restores from the larger step number by default (100)
        assert v.eval() == 5  # restored

        saver.restore(10)  # force restore number 10
        assert v.eval() == 12

        saver.restore(save_100_path)
        assert v.eval() == 5

        # latest should be the largest step number, not necessarily last saved
        assert saver.latest_checkpoint == save_100_path
        assert os.path.exists(save_100_path)

        assert saver.checkpoint_paths == {
            10: save_10_path,
            100: save_100_path,
        }
