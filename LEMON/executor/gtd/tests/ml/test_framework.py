from abc import abstractmethod
from collections import Sequence, Mapping


import numpy as np
import pytest
import tensorflow as tf
from keras.engine import Input
from keras.layers import Dense
from numpy.testing import assert_array_almost_equal

from gtd.ml.framework import Feedable, KerasModel
from gtd.ml.utils import guarantee_initialized_variables, clean_session
from gtd.utils import Bunch


@pytest.yield_fixture
def clean_test_session():
    with clean_session() as sess:
        yield sess


def assert_array_collections_equal(correct, test, decimal=7):
    """Assert that two collections of numpy arrays have the same values.

    Collections can be either a Sequence or a Mapping.
    """
    if type(correct) != type(test):
        raise ValueError('correct ({}) and test ({}) must have the same type.'.format(type(correct), type(test)))

    assert_equal = lambda c, t: assert_array_almost_equal(c, t, decimal=decimal)

    if isinstance(correct, Sequence):
        assert len(correct) == len(test)
        for c, t in zip(correct, test):
            assert_equal(c, t)
    elif isinstance(correct, Mapping):
        # same keys
        assert set(test.keys()) == set(correct.keys())
        # same values
        for key in test:
            assert_equal(correct[key], test[key])
    else:
        raise TypeError('Inputs must be of type Mapping or Sequence, not {}.'.format(type(correct)))


class FeedableTester(object):
    """A template for testing Feedable classes.

    Subclass this class and implement all of its abstractmethods.

    NOTE:
        You must decorate the implementation of each abstractmethod with a @pytest.fixture decorator.
        See the `TestFeedable` class below for an example.
    """
    @abstractmethod
    def model(self):
        """The Model to be tested."""
        pass

    @abstractmethod
    def inputs(self):
        """Inputs to the model.

        Returns:
            (list, dict): an args, kwargs pair
        """
        pass

    @classmethod
    def as_args_kwargs(cls, *args, **kwargs):
        return args, kwargs

    @abstractmethod
    def feed_dict(self):
        """Return the correct result of the model's `feed_dict` method."""
        pass

    @abstractmethod
    def output_tensors(self):
        """Output tensors to be fetched.

        Returns:
            list[np.array]
        """
        pass

    @abstractmethod
    def outputs(self):
        """Return the correct results of running model.compute(fetch=output_tensors, ...)

        Returns:
            list[np.array]
        """
        pass

    @pytest.mark.usefixtures('clean_test_session')
    def test_inputs_to_feed_dict(self, model, inputs, feed_dict):
        """Test for correct feed_dict."""
        args, kwargs = inputs
        test_feed_dict = model.inputs_to_feed_dict(*args, **kwargs)
        assert_array_collections_equal(feed_dict, test_feed_dict)

    @pytest.mark.usefixtures('clean_test_session')
    def test_outputs(self, model, inputs, output_tensors, outputs):
        """Test for correct output."""
        sess = tf.get_default_session()
        guarantee_initialized_variables(sess)
        args, kwargs = inputs
        test_outputs = model.compute(output_tensors, *args, **kwargs)
        assert_array_collections_equal(outputs, test_outputs, decimal=4)


class KerasModelTester(FeedableTester):
    @pytest.fixture
    def output_tensors(self, model):
        return model.output_tensors

    @pytest.mark.usefixtures('clean_test_session')
    def test_placeholders(self, model, feed_dict):
        """Test that model.placeholders matches the keys of feed_dict."""
        assert set(model.placeholders) == set(feed_dict.keys())


class FeedableExample(Feedable):
    def __init__(self):
        x = tf.placeholder(tf.float32, shape=[], name='x')
        y = tf.get_variable('y', shape=[], initializer=tf.constant_initializer(2.0))
        z = x * y

        self.x = x
        self.y = y
        self.z = z

    def inputs_to_feed_dict(self, batch):
        return {self.x: batch.x}


class TestFeedableExample(FeedableTester):
    @pytest.fixture
    def model(self):
        return FeedableExample()

    @pytest.fixture
    def inputs(self):
        return self.as_args_kwargs(Bunch(x=5.0))

    @pytest.fixture
    def feed_dict(self, model):
        return {model.x: 5.0}

    @pytest.fixture
    def output_tensors(self, model):
        return [model.z]

    @pytest.fixture
    def outputs(self):
        return [10.0]


class KerasLayersModelExample(KerasModel):
    """A Model that is defined using Keras layers from beginning to end."""
    def __init__(self):
        x = Input([1])
        y = np.array([[2.0]])
        b = np.array([0.0])
        mult = Dense(1, weights=(y, b))
        z = mult(x)

        self.x = x
        self.mult = mult
        self.z = z

    @property
    def placeholders(self):
        return [self.x]

    def inputs_to_feed_dict(self, batch):
        return {self.x: np.array([[batch.x]])}

    @property
    def output_tensors(self):
        return [self.z]


class TestKerasLayersModel(KerasModelTester):
    @pytest.fixture
    def model(self):
        return KerasLayersModelExample()

    @pytest.fixture
    def inputs(self):
        return self.as_args_kwargs(Bunch(x=5.0))

    @pytest.fixture
    def feed_dict(self, model):
        return {model.x: 5.0}

    @pytest.fixture
    def outputs(self):
        return [10.0]