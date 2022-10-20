"""Store system evaluation results (e.g., accuracy)."""
from collections import OrderedDict
from codecs import open
from math import sqrt

import json
import numpy as np
import os
from scipy.stats import norm


class NumberSequenceStat(object):
    """Stores statistics of a sequence of numbers.
    This is a reimplementation of fig's StatFig.
    """

    def __init__(self):
        self.s_count = 0
        self.s_min = float('inf')
        self.s_max = float('-inf')
        self.s_min_key = None
        self.s_max_key = None
        self.s_sum = 0.
        self.s_sumsq = 0.

    def add(self, x, key=None):
        if isinstance(x, NumberSequenceStat):
            assert not key
            self.s_count += x.s_count
            self.s_sum += x.s_sum
            self.s_sumsq += x.s_sumsq
            if x.s_min < self.s_min:
                self.s_min = x.s_min
                self.s_min_key = x.s_min_key
            if x.s_max > self.s_max:
                self.s_max = x.s_max
                self.s_max_key = x.s_max_key
        elif isinstance(x, (list, tuple)):
            x = [float(u) for u in x]
            self.s_count += len(x)
            self.s_sum += sum(x)
            self.s_sumsq += sum(u*u for u in x)
            min_x = min(x)
            if min_x < self.s_min:
                self.s_min = min_x
                self.s_min_key = key
            max_x = max(x)
            if max_x > self.s_max:
                self.s_max = max_x
                self.s_max_key = key
        else:
            x = float(x)
            self.s_count += 1
            self.s_sum += x
            self.s_sumsq += x * x
            if x < self.s_min:
                self.s_min = x
                self.s_min_key = key
            if x > self.s_max:
                self.s_max = x
                self.s_max_key = key

    @property
    def count(self):
        return self.s_count

    @property
    def mean(self):
        return self.s_sum / self.s_count

    @property
    def sum(self):
        return self.s_sum

    @property
    def variance(self):
        return self.s_sumsq / self.s_count - self.mean ** 2

    @property
    def stddev(self):
        return self.variance ** .5

    @property
    def min(self):
        return self.s_min

    @property
    def max(self):
        return self.s_max

    @property
    def min_key(self):
        return self.s_min_key

    @property
    def max_key(self):
        return self.s_max_key

    @property
    def range(self):
        return self.s_max - self.s_min

    def __str__(self):
        if not self.s_count:
            return "NaN (0)"
        return "{min}{min_key} << {mean} >> {max}{max_key} ({std} std {count} count)".format(
                min=FmtD(self.s_min), min_key=('@' + self.s_min_key if self.s_min_key else ''),
                mean=FmtD(self.mean), std=FmtD(self.stddev),
                max=FmtD(self.s_max), max_key=('@' + self.s_max_key if self.s_max_key else ''),
                count=self.s_count)

    def as_dict(self):
        if not self.s_count:
            return {'count': 0}
        return {
                'count': self.s_count,
                'min': self.s_min,
                'mean': self.mean,
                'stddev': self.stddev,
                'max': self.s_max,
                'sum': self.s_sum,
                }


class BernoulliSequenceStat(NumberSequenceStat):
    """A NumberSequenceStat which assumes each value in the sequence is drawn i.i.d. from a Bernoulli."""
    def __init__(self, number_seq_stat=None):
        super(BernoulliSequenceStat, self).__init__()
        if number_seq_stat:
            self.add(number_seq_stat)

    def __str__(self):
        left, right = self.confidence_interval(0.05)
        ci_str = " 95% CI = [{} - {}]".format(left, right)
        s = super(BernoulliSequenceStat, self).__str__()
        return s + ci_str

    @classmethod
    def _confidence_interval_by_z_wald(cls, p_hat, n, z):
        increment = z * sqrt(p_hat * (1 - p_hat) / n)
        return p_hat - increment, p_hat + increment

    @classmethod
    def _confidence_interval_by_z_wilson(cls, p_hat, n, z):
        """Compute confidence interval for estimate of Bernoulli parameter p.

        Args:
            p_hat: maximum likelihood estimate of p
            n: samples observed
            z: if z = standard_normal_quantile(1 - alpha/2), then alpha is the probability that the
                true p falls outside the CI.

        Uses the Wilson score interval to compute a confidence interval
        for the true underlying Bernoulli parameter p.

        Should behave well even when p is close to 0 or 1 and when n is not too large.
        https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval

        Returns:
            left, right
        """
        z2 = z**2
        n2 = n**2
        numerator = lambda sign: p_hat + z2 / (2 * n) + \
                                 sign * z * sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n2))
        denominator = 1 + z2 / n
        left = numerator(-1.) / denominator
        right = numerator(1.) / denominator
        return left, right

    @classmethod
    def _confidence_interval_by_alpha(cls, p_hat, n, alpha, method='wald'):
        """Compute confidence interval for estimate of Bernoulli parameter p.

        Args:
            p_hat: maximum likelihood estimate of p
            n: samples observed
            alpha: the probability that the true p falls outside the CI

        Returns:
            left, right
        """
        prob = 1 - 0.5 * alpha
        z = norm.ppf(prob)

        compute_ci = cls._confidence_interval_by_z_wald if method == 'wald' else cls._confidence_interval_by_z_wilson

        return compute_ci(p_hat, n, z)

    def confidence_interval(self, alpha):
        p_hat = self.mean
        n = self.count
        return self._confidence_interval_by_alpha(p_hat, n, alpha)


def test_bernoulli_confidence_interval(method='wilson', trials=1000, ps=None):
    """Use this to compare performance of Wald vs Wilson CIs.

    You should see that Wilson does better for extreme values.

    Args:
        method: 'wilson' or 'wald'
        trials: # trials used to empirically estimate coverage probability
    """
    if ps is None:
        ps = np.arange(0.05, 0.95, 0.05)
    n = 200  # observations
    alpha = 0.1  # desired prob of CI not covering the true p

    # run simulations to see if the computed CI has the desired coverage prob
    alpha_hats = []
    for p in ps:
        misses = 0.
        for _ in range(int(trials)):
            samples = np.random.random(n) <= p  # draw n Bernoulli's
            p_hat = np.mean(samples)  # compute estimate
            left, right = BernoulliSequenceStat._confidence_interval_by_alpha(p_hat, n, alpha, method=method)
            if p < left or p > right:
                misses += 1

        alpha_hat = misses / trials
        alpha_hats.append(alpha_hat)

    import matplotlib.pyplot as plt
    plt.plot(ps, alpha_hats)  # this line should be close to the constant alpha for all values of p


def FmtD(x):
    """Return a nicely formatted string for number x."""
    if abs(x - round(x)) < 1e-40:
        return str(int(x))
    if abs(x) < 1e-3:
        return "{:.2e}".format(x)
    return "{:.3f}".format(x)


class Evaluation(object):
    """Stores various statistics."""

    def __init__(self):
        self.stats = OrderedDict()

    def add(self, name, value, key=None, stat_type=NumberSequenceStat):
        """Add a statistic.

        Args:
            name (string): Name of the metric
            value (bool, int, or float): The value
            key (any): (optional) ID of the object that achieves this value
        """
        if name not in self.stats:
            self.stats[name] = stat_type()

        stat = self.stats[name]
        assert isinstance(stat, stat_type)
        stat.add(value, key=key)

    def add_micro_macro(self, name, values, key=None):
        """Add two stats:
        - micro-averaging: average the values in each sequence first
        - macro-averaging: average all values together.
        """
        # Micro
        stat = NumberSequenceStat()
        stat.add(values)
        self.add(name + '_micro', stat, key=key)
        # Macro
        if stat.count:
            self.add(name + '_macro', stat.mean, key=key)

    def add_evaluation(self, evaluation):
        """Add all statistics from another Evaluation object."""
        for name, stat in evaluation.stats.items():
            self.add(name, stat)

    def line_summarize(self, prefix='EVAL', delim=' '):
        """Return a short one-line summary string."""
        stuff = []
        for name, stat in self.stats.items():
            if not stat.count:
                stuff.append(name + '=NaN')
            else:
                stuff.append(name + '=' + FmtD(stat.mean))
        return prefix + ': ' + delim.join(stuff)

    def summarize(self, buffer, prefix='EVAL'):
        """Print an extensive summary.

        Args:
            buffer: can be a file or a StringIO object
        """
        header = '===== SUMMARY for %s =====' % prefix
        buffer.write(header)
        buffer.write('\n')
        # Padding needed for aligning the key names
        pad = '{:' + str(max(len(x) for x in self.stats)) + '}'
        for name, stat in self.stats.items():
            buffer.write(('[{}] ' + pad + ' : {}').format(prefix, name, stat))
            buffer.write('\n')
        buffer.write('=' * len(header))
        buffer.write('\n')

    def json_summarize(self, json_filename, step, prefix=None):
        flags = 'r+' if os.path.exists(json_filename) else 'w+'
        with open(json_filename, flags) as json_file:
            text = json_file.read()
            json_file.seek(0)
            if len(text) == 0:
                log = {}
            else:
                log = json.loads(text)

            stats_dict = self.as_dict(prefix)
            for name, stat in stats_dict.items():
                if name in log:
                    log[name].append(stat['mean'])
                else:
                    log[name] = [stat['mean']]

            json.dump(log, json_file)
            json_file.truncate()

    def tboard_summarize(self, tb_logger, step, prefix=None):
        """Log evaluation to Tensorboard.

        Args:
            tb_logger (TensorBoardLogger)
            step (int)
            prefix (basestring)
        """
        for name, stat in self.stats.items():
            full_name = '{}_{}'.format(prefix, name) if prefix else name
            tb_logger.log(full_name, stat.mean, step)

    def as_dict(self, prefix=None):
        """Return a dict representation of the Evaluation."""
        result = {}
        for name, stat in self.stats.items():
            full_name = '{}_{}'.format(prefix, name) if prefix else name
            result[full_name] = stat.as_dict()
        return result
