from abc import ABCMeta, abstractmethod


import numpy as np

from strongsup.utils import softmax_with_alpha_beta
from strongsup.value import check_denotation
from strongsup.value_function import ConstantValueFunction


class CaseWeighter(object, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, paths, example):
        """Compute update weights for all ParseCases in a batch of ParsePaths.

        Args:
            paths (list[ParsePath])
            example (Example): the Example for which the ParsePaths were sampled

        Returns:
            weights (list[list[float]]): one weight for each ParseCase
        """
        pass


class MMLCaseWeighter(CaseWeighter):
    def __init__(self, alpha, beta, parse_model):
        self._alpha = alpha
        self._beta = beta
        self._parse_model = parse_model

    def _destroy_path_scores(self, paths):
        # A bit of an information-hiding hack.
        # Now that the path weighter has used the path scores, prevent anyone else from using them by overwriting
        # them with None
        for path in paths:
            for case in path:
                case.choice_logits = None
                case.choice_log_probs = None

    def _weight_paths(self, paths, example):
        # paths may have incorrect scores, left there by some exploration policy
        self._parse_model.score_paths(
                paths, ignore_previous_utterances=False, caching=False)

        log_probs = []  # log p(z | x) + log p(y | z)
        for path in paths:
            z_given_x = path.log_prob
            y_given_z = 0 if check_denotation(example.answer, path.finalized_denotation) else float('-inf')
            lp = z_given_x + y_given_z
            log_probs.append(lp)
        log_probs = np.array(log_probs)

        self._destroy_path_scores(paths)  # destroy scores so no one else misuses them

        # if every probability is 0, the softmax downstream will compute 0/0 = NaN.
        # We will assume 0/0 = 0
        if np.all(log_probs == float('-inf')):
            return np.zeros(len(paths))

        # Reweight with alpha and beta
        weights_alpha = softmax_with_alpha_beta(log_probs, self._alpha, self._beta)

        assert np.all(np.isfinite(weights_alpha))
        return weights_alpha

    def __call__(self, paths, example):
        path_weights = self._weight_paths(paths, example)
        case_weights = []
        for path, path_wt in zip(paths, path_weights):
            case_weights.append([path_wt] * len(path))

        return case_weights


class REINFORCECaseWeighter(CaseWeighter):
    def __init__(self, correct_weight, incorrect_weight, value_function):
        """Weights the cases according to REINFORCE

        Args:
            correct_weight (float): the weight that each case should get if the
                                    denotation is correct
            incorrect_weight (float): weight for incorrect denotations
            value_function (StateValueFunction): assigns a value to each state to
                                                 be subtracted as a baseline
        """
        self._correct_weight = correct_weight
        self._incorrect_weight = incorrect_weight
        self._value_function = value_function

    def __call__(self, paths, example):
        path_weights = self._weight_paths(paths, example)
        cases = [case for path in paths for case in path]
        state_values = self._value_function.values(cases)

        case_weights = []
        index = 0
        for path, path_weight in zip(paths, path_weights):
            case_weights_for_path = []
            for case in path:
                case_weights_for_path.append(path_weight - state_values[index])
                index += 1
            case_weights.append(case_weights_for_path)
        return case_weights

    def _weight_paths(self, paths, example):
        # TODO: Destroy path scores?
        return [self._correct_weight
                if check_denotation(example.answer, path.finalized_denotation)
                else self._incorrect_weight for path in paths]


def get_case_weighter(config, parse_model, value_function):
    """Creates the correct CaseWeighter from the Config

    Args:
        config (Config): the config
        parse_model (ParseModel): the parse model that the case weighter
            will use
        value_function (ValueFunction): the value function that the case
            weighter will use

    Returns:
        CaseWeighter
    """
    if config.type == 'mml':
        # Make sure we're not using a ValueFunction if it's MML
        assert type(value_function) is ConstantValueFunction
        assert value_function.constant_value == 0
        return MMLCaseWeighter(config.alpha, config.beta, parse_model)
    elif config.type == 'reinforce':
        return REINFORCECaseWeighter(
                config.correct_weight, config.incorrect_weight, value_function)
    else:
        raise ValueError('CaseWeighter {} not supported.'.format(config.type))
