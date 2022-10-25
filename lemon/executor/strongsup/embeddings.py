import os
from collections import namedtuple
from os.path import join

import numpy as np

from dependency.data_directory import DataDirectory
from gtd.chrono import verboserate
from gtd.ml.vocab import SimpleVocab, SimpleEmbeddings
from gtd.utils import random_seed, cached_property, ComparableMixin

from strongsup.tables.predicate import WikiTablePredicateType, WikiTablePredicate
from strongsup.tables.world import TableWorld


def emulate_distribution(shape, target_samples, seed=None):
    m = np.mean(target_samples)
    s = np.std(target_samples)

    with random_seed(seed):
        samples = np.random.normal(m, s, size=shape)

    return samples


class StaticPredicateEmbeddings(SimpleEmbeddings):
    """All base predicate embeddings are initialized with zero vectors."""
    def __init__(self, embed_dim, fixed_predicates):
        vocab = ContextualPredicateVocab([ContextualPredicate(pred, None) for pred in fixed_predicates])
        array = emulate_distribution((len(vocab), embed_dim), GloveEmbeddings(5000).array, seed=0)
        super(StaticPredicateEmbeddings, self).__init__(array, vocab)


class TypeEmbeddings(SimpleEmbeddings):
    """All type embeddings are initialized with zero vectors."""
    def __init__(self, embed_dim, all_types):
        vocab = SimpleVocab(all_types)
        array = emulate_distribution((len(vocab), embed_dim), GloveEmbeddings(5000).array, seed=1)
        super(TypeEmbeddings, self).__init__(array, vocab)


class RLongPrimitiveEmbeddings(SimpleEmbeddings):
    def __init__(self, embed_dim):
        OBJECT = 'object'
        LIST = 'list'

        tokens = [
            OBJECT, LIST,
            'r', 'y', 'g', 'o', 'p', 'b', 'e',  # 7 colors
            'color-na',  # if an Alchemy beaker is empty or has multiple colors
            # TODO(kelvin): change the behavior of RLongAlchemyObject.color to return `color-na`
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # 0 index is used to represent things that are not visible
            -1,
            'X1/1',
            '0', '1', '2', '3', '4',  # Shapes!
        ]
        vocab = SimpleVocab(tokens)
        vocab.OBJECT = OBJECT
        vocab.LIST = LIST

        array = emulate_distribution((len(vocab), embed_dim), GloveEmbeddings(5000).array, seed=3)
        super(RLongPrimitiveEmbeddings, self).__init__(array, vocab)


class UtteranceVocab(SimpleVocab):
    """Vocab for input utterances.

    IMPORTANT NOTE: UtteranceVocab is blind to casing! All words are converted to lower-case.

    An UtteranceVocab is required to have the following special tokens: UNK, PAD
    See class attributes for more info.
    """
    UNK = "<unk>"
    PAD = "<pad>"
    SPECIAL_TOKENS = (UNK, PAD)

    def __init__(self, tokens):
        tokens = [t.lower() for t in tokens]
        super(UtteranceVocab, self).__init__(tokens)

        # check that all special tokens present
        for special in self.SPECIAL_TOKENS:
            if special not in self._word2index:
                raise ValueError('All special tokens must be present in tokens. Missing {}'.format(special))

    def word2index(self, w):
        """Map a word to an integer.

        If the word is not known to the vocab, return the index for UNK.
        """
        sup = super(UtteranceVocab, self)
        try:
            return sup.word2index(w.lower())
        except KeyError:
            return sup.word2index(self.UNK)


class GloveEmbeddings(SimpleEmbeddings):
    def __init__(self, vocab_size=400000):
        """Load GloveEmbeddings.

        Args:
            word_vocab_size (int): max # of words in the vocab. If not specified, uses all available GloVe vectors.

        Returns:
            (np.array, SemgenVocab)
        """
        embed_dim = 100
        if vocab_size < 5000:
            raise ValueError('Need to at least use 5000 words.')

        glove_path = join(DataDirectory.glove, 'glove.6B.100d.txt')
        download_path = 'http://nlp.stanford.edu/data/glove.6B.zip'
        if not os.path.exists(glove_path):
            raise RuntimeError('Missing file: {}. Download it here: {}'.format(glove_path, download_path))

        # embeddings for special words
        words = list(UtteranceVocab.SPECIAL_TOKENS)
        num_special = len(words)
        embeds = [np.zeros(embed_dim, dtype=np.float32) for _ in words]  # zeros are just placeholders for now

        with open(glove_path, 'r') as f:
            lines = verboserate(f, desc='Loading GloVe embeddings', total=vocab_size, initial=num_special)
            for i, line in enumerate(lines, start=num_special):
                if i == vocab_size: break
                tokens = line.split()
                word, embed = tokens[0], np.array([float(tok) for tok in tokens[1:]])
                words.append(word)
                embeds.append(embed)

        vocab = UtteranceVocab(words)
        embed_matrix = np.stack(embeds)

        special_embeds = emulate_distribution((num_special, embed_dim), embed_matrix[:5000, :], seed=2)
        embed_matrix[:num_special, :] = special_embeds
        assert embed_matrix.shape[1] == 100

        super(GloveEmbeddings, self).__init__(embed_matrix, vocab)


ContextualPredicate = namedtuple('ContextualPredicate', ['predicate', 'utterance'])
# A predicate paired with the utterance it may be mentioned in.
#
# Args:
#     predicate (Predicate)
#     utterance (Utterance)


class ContextualPredicateVocab(SimpleVocab):
    def __init__(self, tokens):
        """Create Vocab.

        Args:
            tokens (list[ContextualPredicate]): each token is a (Predicate, Context) pair.
        """
        for tok in tokens:
            if not isinstance(tok, ContextualPredicate):
                raise ValueError("Every token must be a ContextualPredicate.")
        super(ContextualPredicateVocab, self).__init__(tokens)


class Vocabs(object):
    def __init__(self, utterances, domain):
        """Construct Vocabs.

        Args:
            utterances (frozenset[Utterance]): a frozenset of Utterance objects
        """
        assert isinstance(utterances, frozenset)
        self._utterance_set = utterances
        self._fixed_predicates = domain.fixed_predicates
        self._fixed_predicates_set = set(self._fixed_predicates)

    def __hash__(self):
        return hash(self._utterance_set)

    def __eq__(self, other):
        if not isinstance(other, Vocabs):
            return False
        return self._utterance_set == other._utterance_set

    @cached_property
    def utterances(self):
        tokens = sorted(list(self._utterance_set))
        return SimpleVocab(tokens)

    def as_contextual_pred(self, pred, utterance):
        if self.is_static_pred(pred):
            utterance = None
        return ContextualPredicate(pred, utterance)

    def is_static_pred(self, pred):
        return pred in self._fixed_predicates_set

    @cached_property
    def static_preds(self):
        return ContextualPredicateVocab([self.as_contextual_pred(pred, None) for pred in self._fixed_predicates])

    @cached_property
    def dynamic_preds(self):
        tokens = set()
        for utterance in self._utterance_set:
            for pred in utterance.context.predicates:
                if not self.is_static_pred(pred):
                    tokens.add(self.as_contextual_pred(pred, utterance))

            # include all entities in the corresponding table
            # TODO(kelvin): improve this hack
            world = utterance.context.world
            if isinstance(world, TableWorld):
                graph = world.graph
                rows = graph.all_rows
                ent_strs = set()
                for col_str in graph.all_columns:
                    ent_strs.update(graph.reversed_join(col_str, rows))
                ents = [WikiTablePredicate(s) for s in ent_strs]
                tokens.update([self.as_contextual_pred(e, utterance) for e in ents])

        # necessary to ensure a deterministic result
        tokens = sorted(list(tokens))
        return ContextualPredicateVocab(tokens)

    @cached_property
    def all_preds(self):
        static = self.static_preds
        dynamic = self.dynamic_preds
        joint_tokens = []
        joint_tokens.extend(static.tokens)
        joint_tokens.extend(dynamic.tokens)
        return ContextualPredicateVocab(joint_tokens)
