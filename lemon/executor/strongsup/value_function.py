import abc
import sys
from collections import namedtuple

import numpy as np
import tensorflow as tf

from gtd.ml.framework import Feedable, Model
from keras.layers import Dense
from strongsup.utils import OptimizerOptions, get_optimizer
from strongsup.value import check_denotation


class ValueFunctionExample(namedtuple('ValueFunctionExample', ['case', 'reward'])):
    """Represents a single training example for StateValueFunction.train_step.

    Attributes:
        case (ParseCase)
        reward (float): typically 0 or 1
    """
    __slots__ = ()

    @classmethod
    def examples_from_paths(cls, paths, example):
        """Return a list of ValueFunctionExamples derived from ParsePaths discovered during exploration.

        Args:
            paths (list[ParsePath])
            example (strongsup.example.Example)

        Returns:
            list[ValueFunctionExample]
        """
        vf_examples = []
        for path in paths:
            reward = 1 if check_denotation(example.answer, path.finalized_denotation) else 0
            vf_examples.extend(ValueFunctionExample(case, reward) for case in path)
        return vf_examples


class StateValueFunction(Model, metaclass=abc.ABCMeta):
    """Defines a value function that associates a value V to each state s as in RL"""

    @abc.abstractmethod
    def values(self, cases):
        """Returns the values for the states corresponding to a list of cases
        in the same order.

        Args:
            cases (list[ParseCase]): the cases

        Returns:
            values (list[float]): the values in same order as cases
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, vf_examples):
        """Compute the loss for which we are performing gradient descent upon.

        Args:
            vf_examples (list[ValueFunctionExample])

        Returns:
            float
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, vf_examples):
        """Takes a train step based on training examples

        Args:
            vf_examples (list[ValueFunctionExample])
        """
        raise NotImplementedError


class ConstantValueFunction(StateValueFunction):
    """Gives every state the same value"""
    def __init__(self, constant_value):
        self._constant_value = constant_value

    def values(self, cases):
        return [self._constant_value] * len(cases)

    @property
    def constant_value(self):
        return self._constant_value

    def loss(self, vf_examples):
        """Loss in terms of mean squared error."""
        if len(vf_examples) == 0:
            return 0.0

        c = self._constant_value
        diffs = [(c - ex.reward) for ex in vf_examples]
        return np.mean(np.power(diffs, 2))

    def train_step(self, vf_examples):
        """Is a no-op"""
        return


class LogisticValueFunction(StateValueFunction, Feedable):
    def __init__(self, parse_model, learning_rate, optimizer_opt):
        """
        Args:
            parse_model (ParseModel)
            learning_rate (float)
            optimizer_opt (OptimizerOptions)
        """
        with tf.name_scope("LogisticValueFunction"):
            self._rewards = tf.placeholder(
                    tf.float32, shape=[None], name="rewards")
            # Prevent gradient from updating the stuff that makes up encoding
            encodings = tf.stop_gradient(parse_model.case_encodings)
            self._values = tf.squeeze(
                    Dense(1, activation="sigmoid", bias=True)(encodings),
                    axis=[1])

        loss = tf.reduce_mean(tf.contrib.losses.log_loss(
            self._values, labels=self._rewards))

        optimizer = get_optimizer(optimizer_opt)(learning_rate)
        self._take_step = optimizer.minimize(loss)

        self._parse_model = parse_model
        # Hold it around for testing purposes
        self._loss = loss

    @classmethod
    def _unpack_vf_examples(cls, vf_examples):
        cases = [ex.case for ex in vf_examples]
        rewards = [ex.reward for ex in vf_examples]
        return cases, rewards

    def values(self, cases, ignore_previous_utterances=False):
        if len(cases) == 0:
            # Should only happen if everything gets pruned off beam.
            return []

        fetch = {"values": self._values}
        fetched = self.compute(
                fetch, cases, rewards=None,
                ignore_previous_utterances=ignore_previous_utterances)
        return fetched["values"]

    def loss(self, vf_examples):
        if len(vf_examples) == 0:
            return 0.0
        cases, rewards = self._unpack_vf_examples(vf_examples)
        return self.compute(self._loss, cases, rewards, ignore_previous_utterances=False)

    def train_step(self, vf_examples):
        # Make sure all rewards are between [0, 1] for log_loss
        for ex in vf_examples:
            assert 0 <= ex.reward <= 1

        if len(vf_examples) == 0:
            print(" WARNING: (ValueFunction) Zero cases   \033[F", file=sys.stderr)
        else:
            print(" Updating (ValueFunction) ({} cases)   \033[F".format(
                            len(vf_examples)), file=sys.stderr)

            cases, rewards = self._unpack_vf_examples(vf_examples)

            # Always acknowledge previous utterances on train steps
            self.compute(
                    self._take_step, cases, rewards,
                    ignore_previous_utterances=False)

    def inputs_to_feed_dict(self, cases, rewards=None,
                            ignore_previous_utterances=False):
        feed = {}
        if rewards:
            feed[self._rewards] = rewards

        if len(cases) == 0:
            raise ValueError("No cases")

        feed.update(self._parse_model.inputs_to_feed_dict(
            cases, ignore_previous_utterances, caching=False))
        return feed


def get_value_function(config, parse_model):
    """Needs to take the Config for ValueFunction"""
    if config.type == "constant":
        return ConstantValueFunction(config.constant_value)
    elif config.type == "logistic":
        return LogisticValueFunction(
                parse_model, config.learning_rate,
                OptimizerOptions(config.optimizer))
    else:
        raise ValueError(
                "ValueFunction {} not supported.".format(config.type))
