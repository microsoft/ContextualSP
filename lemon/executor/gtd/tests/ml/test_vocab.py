import numpy as np
import pytest

from gtd.ml.vocab import SimpleVocab, SimpleEmbeddings


@pytest.fixture
def vocab():
    return SimpleVocab(['a', 'b', 'c'])


@pytest.fixture
def embeds(vocab):
    array = np.eye(len(vocab))
    return SimpleEmbeddings(array, vocab)


class TestSimpleVocab(object):
    def test_save_load(self, vocab, tmpdir):
        path = str(tmpdir.join('vocab.txt'))
        vocab.save(path)
        new_vocab = SimpleVocab.load(path)
        assert vocab == new_vocab