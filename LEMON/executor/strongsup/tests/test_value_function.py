import numpy as np
import pytest
import tensorflow as tf

from gtd.ml.framework import Feedable
from gtd.ml.utils import guarantee_initialized_variables
from strongsup.value_function import LogisticValueFunction, ValueFunctionExample
from strongsup.utils import OptimizerOptions


class DummyParseModel(Feedable):
    def __init__(self, weights):
        self._weights = tf.Variable(weights, dtype=tf.float32)
        # Batch size x Embedding size
        self._placeholder = tf.placeholder(tf.float32, shape=[None, 2])
        self._case_encodings = tf.matmul(self._placeholder, self._weights)

    @property
    def case_encodings(self):
        return self._case_encodings

    def inputs_to_feed_dict(self, cases, ignore_previous_utterances, caching):
        # Ignore cases, ignore_previous_utterances, and caching
        dummy_parse_model_inputs = np.array([[1.0, 2.0], [2.0, 3.0]])
        return {self._placeholder: dummy_parse_model_inputs}


@pytest.fixture
def weights():
    return np.array([[0.0, 1.0], [1.0, 0.0]])


@pytest.fixture
def dummy_cases():
    # Never gets used
    return [1, 2]


@pytest.fixture
def rewards():
    return [1.0, 0.0]


@pytest.fixture
def value_function(weights):
    return LogisticValueFunction(
            DummyParseModel(weights), 0.01, OptimizerOptions("adam"))


def test_value_function(value_function, weights, dummy_cases, rewards):
    sess = tf.InteractiveSession()
    guarantee_initialized_variables(sess)

    fetch = {
        "loss": value_function._loss
    }

    feed_dict = value_function.inputs_to_feed_dict(dummy_cases, rewards)

    # Test that the loss decreases after taking a train step
    loss = sess.run(fetch, feed_dict=feed_dict)["loss"]
    values = value_function.values(dummy_cases)
    for i in range(10):
        vf_examples = [ValueFunctionExample(c, r) for c, r in zip(dummy_cases, rewards)]
        value_function.train_step(vf_examples)
    new_loss = sess.run(fetch, feed_dict=feed_dict)["loss"]
    new_values = value_function.values(dummy_cases)
    assert new_loss < loss

    # Test that the weights didn't propagate to the ParseModel
    fetch = {
        "weights": value_function._parse_model._weights
    }

    model_weights = sess.run(fetch, feed_dict=feed_dict)["weights"]
    assert np.array_equal(model_weights, weights)
