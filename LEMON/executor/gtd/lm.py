from abc import ABCMeta, abstractmethod, abstractproperty
from collections import defaultdict, Counter

import numpy as np
from numpy.testing import assert_approx_equal


def last_k(tokens, k):
    """Get the last k elements of a list as a tuple."""
    if not (0 <= k <= len(tokens)):
        raise ValueError('k must be between 0 and len(tokens) = {}, got: {}'.format(len(tokens), k))
    return tuple(tokens[len(tokens) - k:])


def replace_parens(tokens):
    """Replace all instances of -LRB- and -RRB- with actual parentheses."""
    parens_map = {'-LRB-': '(', '-RRB-': ')'}
    return [parens_map.get(s, s) for s in tokens]  # return identity if not parens symbols


def normalize_counts(counts):
    """Return a normalized Counter object."""
    normed = Counter()
    total = sum(list(counts.values()), 0.0)
    assert total > 0  # cannot normalize empty Counter
    for key, ct in counts.items():
        normed[key] = ct / total
    normed.old_total = total  # document what the total was before normalization
    return normed


class LM(object, metaclass=ABCMeta):
    """Language model interface."""

    START = '<START>'
    END = '<END>'

    @abstractmethod
    def next_distribution(self, history):
        """Return a distribution over the next token.

        Args:
            history (List): a list of tokens generated so far

        Returns (Counter): a distribution
        """
        raise NotImplementedError

    @abstractproperty
    def max_context_size(self):
        """Return max allowed history context.

        Returns (int): maximum size of history context to keep
        """
        raise NotImplementedError


class CountLM(LM):
    """Naive language model.

    Uses empirical counts from the largest context it has observed. No sophisticated backoff strategy.

    Examples:
        lm = CountLM(4)

        # 'train' the language model
        for line in lines:
            tokens = line.split()
            lm.record_counts(tokens, append_end=True)
    """

    def __init__(self, max_context_size):
        """Construct a language model.

        Args:
            max_context_size (int): maximum # tokens to use as context
        """
        self._max_context_size = max_context_size
        self.contexts = defaultdict(Counter)

    @property
    def max_context_size(self):
        return self._max_context_size

    def _get_contexts(self, tokens):
        """List of contexts, from smallest to largest. Includes empty context.

        Returns:
            List[Tuple[str]]
        """
        contexts = []
        max_context = min(self._max_context_size, len(tokens))  # cannot go beyond max tokens
        for k in range(max_context + 1):
            contexts.append(last_k(tokens, k))
        return contexts

    def record_counts(self, tokens, append_end):
        """Record counts using `tokens` as a corpus.

        Args:
            tokens (List[str]): list of strings
        """
        history = [LM.START]
        if append_end:
            tokens = tokens + [LM.END]
        for tok in tokens:
            for context in self._get_contexts(history):
                self.contexts[context][tok] += 1
            history.append(tok)  # update history

    def _largest_context(self, history, contexts):
        """Find the largest context which matches history.

        Args:
            history (List[str]): a sequence of tokens
            contexts (Set[Tuple[str]]): a set of contexts, must include the empty context

        Returns:
            Tuple[str]: an item from contexts, which may be the empty context
        """
        assert tuple() in contexts  # empty context must be present

        for context in reversed(self._get_contexts(history)):
            if context in contexts:
                return context

    def _largest_known_context(self, history):
        """Find the largest recorded context which matches history."""
        return self._largest_context(history, self.contexts)

    def next_distribution(self, history):
        """Given a history, return a distribution (Counter) over the next token."""
        context = self._largest_known_context(history)
        counts = self.contexts[context]
        normed = normalize_counts(counts)
        normed.context = context
        return normed

    def sequence_probability(self, tokens):
        """Return the probability of each token in an article, based on the language model.

            Args:
                tokens (List): a list of tokens in the article

        Returns:
             List[Tuple[str, float]]: an ordered list of token-probability pairs"""
        history = [LM.START]
        probabilities = []

        for word in tokens:
            distr = self.next_distribution(history)
            if word in distr:
                probabilities.append((word, distr[word]))
            else:
                probabilities.append((word, 0.0))
            history.append(word)
        return probabilities


class KNNLM(LM):
    def __init__(self, article_embeddings, max_context_size, k_nearest):
        """Construct k-nearest-neighbor language model.

        Args:
            article_embeddings (ArticleEmbeddings): embeddings of each article
            max_context_size (int): max history to consider for CountLM
            k_nearest (int): # neighbors to consider
        """
        self.article_embeddings = article_embeddings
        self.k = k_nearest
        self.lm = CountLM(max_context_size)

    def record_nearest_counts(self, vec):
        name_score_pairs = self.article_embeddings.k_nearest_approx(vec, self.k)
        articles = [self.article_embeddings.name_to_article(name) for name, score in name_score_pairs]
        for art in articles:
            self.lm.record_counts(art.tokens, append_end=True)

    def next_distribution(self, history):
        return self.lm.next_distribution(history)

    @property
    def max_context_size(self):
        return self.lm.max_context_size

    def sequence_probability(self, tokens):
        return self.lm.sequence_probability(tokens)


class Generator(object, metaclass=ABCMeta):
    """Interface for language generator."""

    @abstractmethod
    def init_history(self):
        """Return a sequence of tokens to initialize the history."""
        pass

    @abstractmethod
    def get_next(self, history):
        """Get next token, given history."""
        pass

    @abstractmethod
    def stop_or_not(self, history):
        """Given what has been generated, decide whether to stop."""
        pass

    @abstractproperty
    def max_context_size(self):
        """Return max allowed history context.

        Returns (int): maximum size of history context to keep
        """
        raise NotImplementedError

    def truncate_history(self, history):
        """Truncate history when it grows much longer than max context size."""
        if len(history) > 2 * self.max_context_size:
            return list(last_k(history, self.max_context_size))
        return history

    def generate(self, history=None):
        """Generate a sequence of tokens."""
        if not history:
            history = self.init_history()
        return self.generate_custom(history, self.get_next, self.stop_or_not)

    def generate_custom(self, history, next_fxn, stop_fxn):
        """Generate a sequence using a custom next-token function and a custom stopping function.

        Args:
            history (List[T]): initial history
            next_fxn (Callable[[List[T]], T]): given a history, produce the next token
            stop_fxn (Callable[[List[T]], bool]): given a history, decide whether to stop
        """
        generated = []
        history = list(history)  # make a copy
        while True:
            next = next_fxn(history)
            history.append(next)
            history = self.truncate_history(history)
            if stop_fxn(history):
                break
            generated.append(next)
        return generated


class LMSampler(Generator):
    """Generation by sampling from a language model."""

    def __init__(self, lm):
        """Construct a LM sampler.

        Args:
            lm (LM): a language model
        """
        self.lm = lm

    @property
    def max_context_size(self):
        return self.lm.max_context_size

    def _sample_from_distribution(self, distr):
        """Sample from a categorical distribution.

        Args:
            distr (Counter): values must sum to 1
        Returns:
            one of the keys of distr
        """
        keys, probs = list(zip(*list(distr.items())))
        assert_approx_equal(sum(probs), 1.)
        return np.random.choice(keys, p=probs)

    def init_history(self):
        return [self.lm.START]

    def get_next(self, history):
        return self._sample_from_distribution(self.lm.next_distribution(history))

    def stop_or_not(self, history):
        return history[-1] == LM.END

    @staticmethod
    def format_generation(tokens):
        return ' '.join(replace_parens(tokens))


class DistributionStats(object):
    def __init__(self, distr):
        self.total = distr.old_total
        self.context = distr.context

        probs = list(distr.values())
        assert_approx_equal(sum(probs), 1.)
        self.entropy = -1. * sum([p * np.log(p) for p in probs])

    def __repr__(self):
        return '{}:{}:{}'.format(len(self.context), self.total, self.entropy)


class LMSamplerWithStats(LMSampler):
    def init_history(self):
        return [(LM.START, 0)]

    def get_next(self, history):
        token_history, _ = list(zip(*history))
        distr = self.lm.next_distribution(token_history)
        next_token = self._sample_from_distribution(distr)
        return next_token, DistributionStats(distr)

    def stop_or_not(self, history):
        word = lambda pair: pair[0]
        return word(history[-1]) == LM.END

    @staticmethod
    def format_generation(token_stat_pairs):
        tokens, stats = list(zip(*list(token_stat_pairs)))
        tokens = replace_parens(tokens)
        tokens = ['{:20}[{}]'.format(tok, stat) for tok, stat in zip(tokens, stats)]

        return '\n'.join(tokens)
