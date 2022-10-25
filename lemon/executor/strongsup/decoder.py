from collections import namedtuple

import numpy as np

from gtd.utils import flatten
from strongsup.case_weighter import get_case_weighter
from strongsup.value_function import get_value_function, ValueFunctionExample


class NormalizationOptions(object):
    """Constants for normalization options"""
    LOCAL = 'local'
    GLOBAL = 'global'


# used by the Decoder to compute gradients
WeightedCase = namedtuple('WeightedCase', ['case', 'weight'])


class Decoder(object):
    """A decoder does two things:
    - Given a batch of examples, produce a Beam (list of ParsePaths) for each example.
        Internally it uses an ExplorationPolicy to produce beams, and a ParseModel
        to score the ParseCases.
    - Given a batch of Beams, update the model parameters by passing appropriate
        ParseCases to the TrainParseModel.
    """

    def __init__(self, parse_model, config, domain):
        """Create a new decoder.

        Args:
            parse_model (TrainParseModel)
            config (Config): The decoder section of the config
            domain (Domain)
        """
        self._parse_model = parse_model
        self._value_function = get_value_function(
                config.value_function, parse_model.parse_model)
        self._case_weighter = get_case_weighter(
                config.case_weighter, parse_model.parse_model,
                self._value_function)
        self._config = config
        self._caching = config.inputs_caching
        self._domain = domain
        self._path_checker = domain.path_checker

        # Normalization and update policy
        self._normalization = config.normalization
        if config.normalization == NormalizationOptions.GLOBAL:
            raise ValueError('Global normalization is no longer supported.')

        # Exploration policy
        # TODO: Resolve this circular import differently
        from strongsup.exploration_policy import get_exploration_policy
        self._test_exploration_policy = get_exploration_policy(
                self, config.test_exploration_policy,
                self._normalization, train=False)
        self._train_exploration_policy = get_exploration_policy(
                self, config.train_exploration_policy,
                self._normalization, train=True)

    @property
    def parse_model(self):
        return self._parse_model

    @property
    def caching(self):
        return self._caching

    @property
    def domain(self):
        return self._domain

    def exploration_policy(self, train):
        """Returns the train or test exploration policy depending on
        train

        Args:
            train (bool)

        Returns:
            ExplorationPolicy
        """
        if train:
            return self._train_exploration_policy
        else:
            return self._test_exploration_policy

    def path_checker(self, path):
        """Return False if the ParsePath should be pruned away; True otherwise.

        Args:
            path (ParsePath)
        Returns:
            bool
        """
        return self._path_checker(path)

    def get_probs(self, beam):
        """Return a numpy array containing the probabilities of the paths
        in the given beam.

        The entries may not sum to 1 for local normalization since we have
        pruned away choices that are not executable.

        Args:
            beam (Beam)
        Returns:
            np.array of length len(beam) containing the probabilities.
        """
        if len(beam) == 0:
            return np.zeros(0)
        if self._normalization == NormalizationOptions.LOCAL:
            return np.exp(np.array([path.log_prob for path in beam]))
        else:
            stuff = np.array([path.score for path in beam])
            stuff = np.array(stuff - np.min(stuff))
            return stuff / np.sum(stuff)

    ################################
    # Prediction

    def predictions(self, examples, train, verbose=False):
        """Return the final beams for a batch of contexts.

        Args:
            contexts (list[Context]): a batch of Contexts
            verbose (bool)
            train (bool): If you're training or evaluating

        Returns:
            list[Beam]: a batch of Beams
        """
        exploration_policy = self.exploration_policy(train)
        beams = exploration_policy.get_beams(examples, verbose)
        return [beam.get_terminated() for beam in beams]

    def get_intermediate_beams(self, examples, train, verbose=False):
        exploration_policy = self.exploration_policy(train)
        return exploration_policy.get_intermediate_beams(examples, verbose)

    def score_breakdown(self, paths):
        """Return the logits for all (parse case, choice, scorer) tuples.

        Args:
            paths (list[ParsePath])
        Returns:
            grouped_attentions:
                a list of length(paths). Each entry is an np.array of shape
                (>= len(utterance)) containing the attention scores
            grouped_subscores:
                a list of length len(paths). Each entry is an np.array of shape
                (>= number of cases, len(choices), number of scorers)
                containing the logits of each scorer on each choice.
                By default there are 3 scorers: basic, attention, and soft copy.
        """
        if len(paths) == 0:
            return [], []
        cumul = [0]         # Used to group the results back
        cases = []
        for path in paths:
            for case in path:
                cases.append(case)
            cumul.append(len(cases))
        # Get the scores from the model
        attentions, subscores = self._parse_model.score_breakdown(cases, ignore_previous_utterances=False,
                                                                  caching=False)
        # Group the scores by paths
        grouped_attentions, grouped_subscores = [], []
        for i in range(len(paths)):
            grouped_attentions.append(attentions[cumul[i]:cumul[i+1]])
            grouped_subscores.append(subscores[cumul[i]:cumul[i+1]])
        return grouped_attentions, grouped_subscores

    ################################
    # Training

    def train_step(self, examples):
        # sample a beam of logical forms for each example
        beams = self.predictions(examples, train=True)

        all_cases = []  # a list of ParseCases to give to ParseModel
        all_case_weights = [] # the weights associated with the cases
        for example, paths in zip(examples, beams):
            case_weights = self._case_weighter(paths, example)
            case_weights = flatten(case_weights)
            cases = flatten(paths)
            assert len(case_weights) == sum(len(p) for p in paths)

            all_cases.extend(cases)
            all_case_weights.extend(case_weights)

        # for efficiency, prune cases with weight 0
        cases_to_reinforce = []
        weights_to_reinforce = []
        for case, weight in zip(all_cases, all_case_weights):
            if weight != 0:
                cases_to_reinforce.append(case)
                weights_to_reinforce.append(weight)

        # update value function
        vf_examples = []
        for example, paths in zip(examples, beams):
            vf_examples.extend(ValueFunctionExample.examples_from_paths(paths, example))
        self._value_function.train_step(vf_examples)

        # update parse model
        self._parse_model.train_step(
                cases_to_reinforce, weights_to_reinforce, caching=False)

    @property
    def step(self):
        return self._parse_model.step
