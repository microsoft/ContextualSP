from abc import ABCMeta, abstractmethod
from collections import Mapping

import numpy as np

from gtd.utils import EqualityMixin


class Vocab(object, metaclass=ABCMeta):
    @abstractmethod
    def word2index(self, w):
        pass

    @abstractmethod
    def index2word(self, i):
        pass


class SimpleVocab(Vocab, EqualityMixin):
    """A simple vocabulary object."""

    def __init__(self, tokens):
        """Create a vocab.

        Args:
            tokens (list[unicode]): a unique list of unicode tokens

        If t = tokens[i], this vocab will map token t to the integer i.
        """
        if not isinstance(tokens, list):
            raise ValueError('tokens must be a list')

        # build mapping
        word2index = {}
        for i, tok in enumerate(tokens):
            word2index[tok] = i

        if len(tokens) != len(word2index):
            raise ValueError('tokens must be unique')

        self._index2word = list(tokens)  # make a copy
        self._word2index = word2index

    @property
    def tokens(self):
        """Return the full list of tokens sorted by their index."""
        return self._index2word

    def __iter__(self):
        """Iterate through the full list of tokens."""
        return iter(self._index2word)

    def __len__(self):
        """Total number of tokens indexed."""
        return len(self._index2word)

    def __contains__(self, w):
        """Check if a token has been indexed by this vocab."""
        return w in self._word2index

    def word2index(self, w):
        return self._word2index[w]

    def index2word(self, i):
        return self._index2word[i]

    def words2indices(self, words):
        return list(map(self.word2index, words))

    def indices2words(self, indices):
        return [self.index2word(i) for i in indices]

    def save(self, path):
        """Save SimpleVocab to file path.

        Args:
            path (str)
        """
        with open(path, 'w') as f:
            for word in self._index2word:
                f.write(word)
                f.write('\n')

    @classmethod
    def load(cls, path):
        """Load SimpleVocab from file path.

        Args:
            path (str)

        Returns:
            SimpleVocab
        """
        strip_newline = lambda s: s[:-1]
        with open(path, 'r') as f:
            tokens = [strip_newline(line) for line in f]
        return cls(tokens)


class SimpleEmbeddings(Mapping):
    def __init__(self, array, vocab):
        """Create embeddings object.

        Args:
            array (np.array): has shape (vocab_size, embed_dim)
            vocab (SimpleVocab): a Vocab object
        """
        assert len(array.shape) == 2
        assert array.shape[0] == len(vocab)  # entries line up

        self.array = array
        self.vocab = vocab

    def __contains__(self, w):
        return w in self.vocab

    def __getitem__(self, w):
        idx = self.vocab.word2index(w)
        return np.copy(self.array[idx])

    def __iter__(self):
        return iter(self.vocab)

    def __len__(self):
        return len(self.vocab)

    @property
    def embed_dim(self):
        return self.array.shape[1]