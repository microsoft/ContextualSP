import re
import logging
import numpy as np
from gtd.utils import memoize


@memoize
def get_spacy():
    """
    Loads the spaCy english processor.

    Tokenizing, Parsing, and NER are enabled. All other features are disabled.

    Returns:
        A spaCy Language object for English
    """
    logging.info('Loading spaCy...')
    import spacy.en
    nlp = spacy.en.English(tagger=False, parser=True, matcher=False)
    return nlp


class NER(object):
    def __init__(self):
        self.processor = get_spacy()

    def __call__(self, text):
        """Given a unicode string, return a tuple of the named entities found inside."""
        if not isinstance(text, str):
            text = str(text)
        doc = self.processor(text)
        return doc.ents


class Trie(object):

    def __init__(self, token, parent, sink=False):
        self.token = token
        self.parent = parent
        self.sink = sink
        self.children = {}

    def __contains__(self, phrase):
        if phrase[0] == self.token:
            if len(phrase) == 1:
                # On our last word. Must be a sink to match.
                return self.sink
        else:
            # doesn't match
            return False

        suffix = phrase[1:]
        for child in list(self.children.values()):
            if suffix in child:
                return True

    def ancestors(self):
        if self.parent is None:
            return []
        anc = self.parent.ancestors()
        anc.append(self.token)
        return anc


class PhraseMatcher(object):
    def __init__(self, phrases):
        """Construct a phrase matcher.

        Args:
            phrases (List[Tuple[str]]): a list of phrases to match, where each phrase is a tuple of strings
        """
        # construct Trie
        root = Trie('ROOT', None)
        for phrase in phrases:
            current = root
            for token in phrase:
                if token not in current.children:
                    current.children[token] = Trie(token, current)
                current = current.children[token]
            current.sink = True  # mark last node as a sink

        self.root = root
        self.phrases = phrases

    def has_phrase(self, phrase):
        """Check if a particular phrase is matched by the matcher.

        Args:
            phrase (tuple[str])
        """
        return ['ROOT'] + phrase in self.root

    def match(self, tokens):
        """A list of matches.

        Args:
            tokens (list[str]): a list of tokens

        Returns:
            list[tuple[str, int, int]]: A list of (match, start, end) triples. Each `match` is a tuple of tokens.
            `start` and `end` are word offsets.
        """
        root = self.root
        candidates = [root]

        matches = []
        for i, token in enumerate(tokens):

            # extend candidates or prune failed candidates
            new_candidates = []
            for cand in candidates:
                if token in cand.children:
                    new_candidates.append(cand.children[token])  # move to child
            candidates = new_candidates
            candidates.append(root)  # always add root

            for cand in candidates:
                if cand.sink:
                    match = tuple(cand.ancestors())
                    end = i + 1
                    start = end - len(match)
                    matches.append((match, start, end))

        return matches


# first_cap_re = re.compile('(.)([A-Z][a-z]+)')
first_cap_re = re.compile('([^-_])([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake_case(name):
    """Convert camelCase to snake_case (Python)."""
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()


def longest_common_subsequence(X, Y):
    # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Computing_the_length_of_the_LCS

    def LCS(X, Y):
        m = len(X)
        n = len(Y)
        # An (m+1) times (n+1) matrix
        C = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    C[i][j] = C[i - 1][j - 1] + 1
                else:
                    C[i][j] = max(C[i][j - 1], C[i - 1][j])
        return C

    def backTrack(C, X, Y, i, j):
        if i == 0 or j == 0:
            return []
        elif X[i - 1] == Y[j - 1]:
            return backTrack(C, X, Y, i - 1, j - 1) + [X[i - 1]]
        else:
            if C[i][j - 1] > C[i - 1][j]:
                return backTrack(C, X, Y, i, j - 1)
            else:
                return backTrack(C, X, Y, i - 1, j)

    m = len(X)
    n = len(Y)
    C = LCS(X, Y)
    return backTrack(C, X, Y, m, n)


def get_ngrams(s, n):
    """Get n-grams for s.

    >>> s = [1, 2, 3, 4]
    >>> get_ngrams(s, 2)
    [(1, 2), (2, 3), (3, 4)]
    >>> get_ngrams(s, 1)
    [(1,), (2,), (3,), (4,)]
    >>> get_ngrams(s, 4)
    [(1, 2, 3, 4)]
    """
    assert n <= len(s)
    assert n >= 1
    return [tuple(s[k:k + n]) for k in range(len(s) + 1 - n)]


def ngram_precision_recall(reference, candidate, n=None):
    if n is None:
        # Take the average over 1 through 4 grams.
        prs = []
        for m in [1, 2, 3, 4]:
            prs.append(ngram_precision_recall(reference, candidate, m))
        ps, rs = list(zip(*prs))
        return np.mean(ps), np.mean(rs)

    ref_set = set(get_ngrams(reference, n))
    can_set = set(get_ngrams(candidate, n))
    correct = float(len(ref_set & can_set))
    rec = correct / len(ref_set)
    prec = correct / len(can_set)
    return prec, rec