import copy
import random
from collections import MutableMapping

import numpy as np
import tensorflow as tf


# End of utterance token
EOU = '<EOU>'


def epsilon_greedy_sample(choices, num_to_sample, epsilon=0.05):
    """Samples without replacement num_to_sample choices from choices
    where the ith choice is choices[i] with prob 1 - epsilon, and
    uniformly at random with prob epsilon

    Args:
        choices (list[Object]): a list of choices
        num_to_sample (int): number of things to sample
        epsilon (float): probability to deviate

    Returns:
        list[Object]: list of size num_to_sample choices
    """
    assert(len(choices) >= num_to_sample)
    assert(0 <= epsilon <= 1)

    if (len(choices) == num_to_sample):
        return choices

    # Performance
    if epsilon == 0:
        return choices[:num_to_sample]

    sample = []
    index_choices = list(range(len(choices)))
    for i in range(num_to_sample):
        if random.random() <= epsilon or not i in index_choices:
            choice_index = random.choice(index_choices)
        else:
            choice_index = i
        index_choices.remove(choice_index)
        sample.append(choices[choice_index])
    return sample


def softmax(stuff):
    """Compute [exp(x) / S for x in stuff] where S = sum(exp(x) for x in stuff)"""
    stuff = np.array(stuff)
    stuff = np.exp(stuff - np.max(stuff))
    return stuff / np.sum(stuff)


def softmax_with_alpha_beta(stuff, alpha, beta):
    """Compute [exp(x*beta) / T * S^(1-alpha) for x in stuff]
    where S = sum(exp(x) for x in stuff)
    and   T = sum(exp(x*beta) for x in stuff)

    Assume that alpha >= 0 and beta >= 0.
    """
    stuff = np.array(stuff)
    stuff_times_beta = np.array([
        x * beta if x != float('-inf') else float('-inf')
        for x in stuff])
    m = np.max(stuff)
    return np.exp(
            stuff_times_beta
            - (m * beta + np.log(np.sum(np.exp(stuff_times_beta - m * beta))))
            + (1 - alpha) * (m + np.log(np.sum(np.exp(stuff - m)))))


def sample_with_replacement(stuff, probs, num_to_sample):
    """Samples num_to_sample total elements from stuff.

    Returns:
        list: list of elements
    """
    indices = np.random.choice(
                len(stuff), size=num_to_sample, replace=True, p=probs)
    return [stuff[index] for index in indices]


class PredicateList(object):
    """list[Predicate] but with fast index lookup"""

    def __init__(self, predicates):
        self.predicates = predicates
        self.predicate_to_index = {x.name: i for (i, x) in enumerate(predicates)}

    def index(self, x):
        return self.predicate_to_index[x.name]

    def __iter__(self):
        return iter(self.predicates)

    def __len__(self):
        return len(self.predicates)

    def __repr__(self):
        return repr(self.predicates)

    def __getitem__(self, i):
        return self.predicates[i]


class OptimizerOptions(object):
    SGD = "sgd"
    ADAM = "adam"
    VALID_OPTIONS = [SGD, ADAM]

    """Light-weight wrapper around options for Optimizers

    Args:
        opt_str (string): the string, needs to be in VALID_OPTIONS
    """

    def __init__(self, opt_str):
        if opt_str not in OptimizerOptions.VALID_OPTIONS:
            raise ValueError(
                    "{} not a valid optimizer option".format(opt_str))

        self._opt = opt_str

    @property
    def opt(self):
        return self._opt


def get_optimizer(optimizer_opt):
    assert type(optimizer_opt) is OptimizerOptions

    if optimizer_opt.opt == OptimizerOptions.SGD:
        return tf.train.GradientDescentOptimizer
    elif optimizer_opt.opt == OptimizerOptions.ADAM:
        return tf.train.AdamOptimizer
    else:
        raise ValueError("This should never happen")
