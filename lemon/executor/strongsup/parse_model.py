import logging
import sys
from abc import abstractproperty, ABCMeta

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from numpy.testing import assert_array_almost_equal

from gtd.chrono import verboserate
from gtd.ml.framework import Feedable, Optimizable
from gtd.ml.model import Embedder, MeanSequenceEmbedder, ConcatSequenceEmbedder, \
    CandidateScorer, SoftCopyScorer, Scorer, \
    Attention, BidiLSTMSequenceEmbedder, TokenEmbedder
from gtd.ml.seq_batch import SequenceBatch, FeedSequenceBatch, reduce_mean, embed
from gtd.ml.utils import expand_dims_for_broadcast, gather_2d
from gtd.utils import DictMemoized
from strongsup.embeddings import Vocabs, ContextualPredicate
from strongsup.example import DelexicalizedUtterance
from strongsup.parse_case import PrettyCaseEmbedding
from strongsup.rlong.state import RLongObject
from strongsup.tables.predicate import WikiTablePredicate, WikiTablePredicateType
from strongsup.utils import get_optimizer


################################
# Embedders

class DelexicalizedDynamicPredicateEmbedder(Embedder):
    def __init__(self, rnn_states, type_embedder, name='DelexicalizedDynamicPredicateEmbedder'):
        """Construct DelexicalizedDynamicPredicateEmbedder.

        Args:
            rnn_states (SequenceBatch): of shape (num_contexts, seq_length, rnn_state_dim)
            type_embedder (TokenEmbedder)
            name (str)
        """
        self._type_embedder = type_embedder

        with tf.name_scope(name):
            # column indices of rnn_states (indexes time)
            self._col_indices = FeedSequenceBatch()  # (num_predicates, max_predicate_mentions)

            # row indices of rnn_states (indexes utterance)
            self._row_indices = tf.placeholder(dtype=tf.int32, shape=[None])  # (num_predicates,)
            row_indices_expanded = expand_dims_for_broadcast(self._row_indices, self._col_indices.values)

            # (num_predicates, max_predicate_mentions, rnn_state_dim)
            rnn_states_selected = SequenceBatch(
                gather_2d(rnn_states.values, row_indices_expanded, self._col_indices.values),
                self._col_indices.mask)

            # (num_predicates, rnn_state_dim)
            rnn_embeds = reduce_mean(rnn_states_selected, allow_empty=True)
            rnn_embeds = tf.verify_tensor_all_finite(rnn_embeds, "RNN-state-based embeddings")

            self._type_seq_embedder = MeanSequenceEmbedder(type_embedder.embeds, name='TypeEmbedder')
            self._embeds = tf.concat(1, [rnn_embeds, self._type_seq_embedder.embeds])

    def inputs_to_feed_dict(self, vocabs):
        """Feed.

        Args:
            vocabs (Vocabs)

        Returns:
            dict
        """
        utterance_vocab = vocabs.utterances
        pred_types = []
        row_indices = []
        col_indices = []

        for contextual_pred in vocabs.dynamic_preds.tokens:
            pred = contextual_pred.predicate
            utterance = contextual_pred.utterance
            pred_types.append(list(pred.types))

            if utterance is None:
                utterance_idx = 0
                positions = []
            else:
                # an int corresponding to a row index of rnn_states
                utterance_idx = utterance_vocab.word2index(utterance)
                try:
                    # the token offsets of where the predicate is mentioned in the delexicalized utterance
                    positions = utterance.placeholder_positions[pred]
                except KeyError:
                    # predicate doesn't appear in utterance
                    positions = []

            row_indices.append(utterance_idx)
            col_indices.append(positions)

        feed = {}
        feed[self._row_indices] = row_indices
        feed.update(self._col_indices.inputs_to_feed_dict(col_indices))
        feed.update(self._type_seq_embedder.inputs_to_feed_dict(pred_types, self._type_embedder.vocab))
        return feed

    @property
    def embeds(self):
        return self._embeds


class DynamicPredicateEmbedder(Embedder):
    def __init__(self, word_embedder, type_embedder, name='DynamicPredicateEmbedder'):
        """PredicateEmbedder.

        Embed a predicate as the average of its words, and the average of its types.

        Args:
            word_embedder (TokenEmbedder)
            type_embedder (TokenEmbedder)
            name (str): name scope for the sub-graph
        """
        self._word_embedder = word_embedder
        self._type_embedder = type_embedder

        with tf.name_scope(name):
            self._word_seq_embedder = MeanSequenceEmbedder(word_embedder.embeds, name='WordEmbedder')
            self._type_seq_embedder = MeanSequenceEmbedder(type_embedder.embeds, name='TypeEmbedder')
            self._embeds = tf.concat(1, [self._word_seq_embedder.embeds, self._type_seq_embedder.embeds])

    def inputs_to_feed_dict(self, vocabs):
        """Feed.

        Args:
            vocabs (Vocabs)

        Returns:
            dict
        """
        predicates = vocabs.dynamic_preds.tokens
        pred_words = [contextual_pred.predicate.words for contextual_pred in predicates]
        pred_types = [list(contextual_pred.predicate.types) for contextual_pred in predicates]

        feed = {}
        feed_words = self._word_seq_embedder.inputs_to_feed_dict(pred_words, self._word_embedder.vocab)
        feed_types = self._type_seq_embedder.inputs_to_feed_dict(pred_types, self._type_embedder.vocab)
        feed.update(feed_words)
        feed.update(feed_types)
        return feed

    @property
    def embeds(self):
        return self._embeds


class PositionalPredicateEmbedder(Embedder):
    def __init__(self, pred_embedder, name='PositionalPredicateEmbedder'):
        """Embed predicates using positional information.

        Args:
            pred_embedder (DynamicPredicateEmbedder): a dynamic predicate embedder, with no positional information
            name (str): name scope
        """
        with tf.name_scope(name):
            nbr_embedder = MeanSequenceEmbedder(pred_embedder.embeds,
                                                allow_empty=True)  # average of a predicate's neighbors
            self._embeds = tf.concat(1, [pred_embedder.embeds, nbr_embedder.embeds])

        self._nbr_embedder = nbr_embedder
        self._pred_embedder = pred_embedder

    @property
    def embeds(self):
        return self._embeds

    def _column_values(self, contextual_pred):
        pred = contextual_pred.predicate
        utterance = contextual_pred.utterance
        context = utterance.context
        pred_str = pred.name
        graph = context.world.graph
        ent_strs = list(graph.reversed_join(pred_str, graph.all_rows))
        return [ContextualPredicate(WikiTablePredicate(s), utterance) for s in ent_strs]

    def inputs_to_feed_dict(self, vocabs):
        """Feed.

        Args:
            vocabs (Vocabs)

        Returns:
            dict
        """
        dynamic_vocab = vocabs.dynamic_preds

        feed = {}
        feed.update(self._pred_embedder.inputs_to_feed_dict(vocabs))

        neighbors = []
        for contextual_pred in dynamic_vocab.tokens:
            if WikiTablePredicateType.is_relation(contextual_pred.predicate):
                nbrs = self._column_values(contextual_pred)  # a list of entity predicates
            else:
                nbrs = []
            neighbors.append(nbrs)

        feed.update(self._nbr_embedder.inputs_to_feed_dict(neighbors, dynamic_vocab))
        return feed


class CombinedPredicateEmbedder(Embedder):
    """Concatenates embeddings for static and dynamic predicates

        - static predicates: argmax, join, count, etc.
        - dynamic predicates: e.g. united_states, nation, num_gold_medals
    """

    def __init__(self, static_pred_embedder, dyn_pred_embedder):
        """Construct full predicate embedding model.

        Args:
            static_pred_embedder (TokenEmbedder): embeds for static predicates
            dyn_pred_embedder (DynamicPredicateEmbedder): embedder for dynamic predicates
        """
        with tf.name_scope('PredicateEmbedder'):
            self._embeds = tf.concat(0, [static_pred_embedder.embeds, dyn_pred_embedder.embeds])

        assert isinstance(static_pred_embedder, TokenEmbedder)

        self._static_pred_embedder = static_pred_embedder
        self._dyn_pred_embedder = dyn_pred_embedder

    @property
    def embeds(self):
        return self._embeds  # (vocab_size, embed_dim)

    def inputs_to_feed_dict(self, vocabs):
        # TODO(kelvin): these assert calls are slow
        assert vocabs.all_preds.tokens == vocabs.static_preds.tokens + vocabs.dynamic_preds.tokens
        assert vocabs.static_preds.tokens == self._static_pred_embedder.vocab.tokens
        return self._dyn_pred_embedder.inputs_to_feed_dict(vocabs)

    # WARNINGS:
    #     - The caching is only efficient if the same SETS of utterances are encountered across different mini-batches
    #     - The caching is only correct if the set of predicates only depends on the set of utterances in a mini-batch,
    #         and not the specific ParseCases.
    inputs_to_feed_dict_cached = DictMemoized(inputs_to_feed_dict)


class UtteranceEmbedder(Embedder):
    def __init__(self, word_embedder, lstm_dim, utterance_length):
        with tf.name_scope('UtteranceEmbedder'):
            self._word_vocab = word_embedder.vocab

            # A simpler embedder which is order-blind
            # self._seq_embedder = MeanSequenceEmbedder(word_embedder.embeds, seq_length=utterance_length)
            # self._seq_embedder.hidden_states = self._seq_embedder._embedded_sequence_batch

            self._seq_embedder = BidiLSTMSequenceEmbedder(word_embedder.embeds, seq_length=utterance_length, hidden_size=lstm_dim)

            self._gather_indices = tf.placeholder(tf.int32, shape=[None], name='gather_indices')
            self._gathered_embeds = tf.gather(self._seq_embedder.embeds, self._gather_indices)

            hidden_states = self._seq_embedder.hidden_states
            self._hidden_states_by_utterance = hidden_states
            self._gathered_hidden_states = SequenceBatch(tf.gather(hidden_states.values, self._gather_indices),
                                                         tf.gather(hidden_states.mask, self._gather_indices))

    @property
    def hidden_states(self):
        """A SequenceBatch."""
        return self._gathered_hidden_states

    @property
    def hidden_states_by_utterance(self):
        return self._hidden_states_by_utterance

    def inputs_to_feed_dict(self, cases, utterance_vocab):
        # Optimization: Multiple cases have the same context (same utterance)
        gather_indices = []
        for case in cases:
            gather_indices.append(utterance_vocab.word2index(case.current_utterance))

        feed = self._seq_embedder.inputs_to_feed_dict(utterance_vocab.tokens, self._word_vocab)
        feed[self._gather_indices] = gather_indices
        return feed

    @property
    def embeds(self):
        # return self._seq_embedder.embeds
        return self._gathered_embeds


class HistoryEmbedder(Embedder):
    def __init__(self, pred_embedder, history_length):
        pred_embeds = pred_embedder.embeds
        with tf.name_scope('HistoryEmbedder'):
            self._seq_embedder = ConcatSequenceEmbedder(pred_embeds, seq_length=history_length, align='right')

        self._history_length = history_length
        self._embeds = self._seq_embedder.embeds  # (batch_size, history_dim)
        self._pred_embedder = pred_embedder

        self._build_embeds_hash(self._embeds, history_length, pred_embedder.embed_dim)

    def _build_embeds_hash(self, embeds, history_length, embed_dim):
        # embeds is (batch_size, history_length * embed_dim)
        embeds_shape = tf.shape(embeds)
        batch_size = embeds_shape[0]
        reshaped_embeds = tf.reshape(embeds, [batch_size, history_length, embed_dim])

        # random vector, initialized once and never trained
        hash_vector = tf.get_variable('history_hash_vector', shape=[embed_dim], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(seed=0), trainable=False)

        # inner product every predicate embedding with the hash vector
        hash = tf.reshape(hash_vector, [1, 1, embed_dim])  # expand dims for broadcast
        self._embeds_hash = tf.reduce_sum(reshaped_embeds * hash, axis=2, keep_dims=False)  # (batch_size, max_stack_size)

    @property
    def embeds_hash(self):
        return self._embeds_hash

    @property
    def history_length(self):
        return self._history_length

    @classmethod
    def previous_decisions_for_this_utterance(cls, case):
        target_utterance_idx = case.current_utterance_idx
        previous_decisions = []
        biggest_idx_so_far = 0
        for prev_case in case._previous_cases:
            utterance_idx = prev_case.current_utterance_idx
            assert utterance_idx >= biggest_idx_so_far  # monotonicity check
            biggest_idx_so_far = max(biggest_idx_so_far, utterance_idx)

            if utterance_idx == target_utterance_idx:
                previous_decisions.append(prev_case.decision)
        return previous_decisions

    def inputs_to_feed_dict(self, cases, vocabs, ignore_previous_utterances):
        histories = []
        for case in cases:
            if ignore_previous_utterances:
                previous_decisions = self.previous_decisions_for_this_utterance(case)
            else:
                previous_decisions = case.previous_decisions

            utterance = case.current_utterance
            history = [vocabs.as_contextual_pred(pred, utterance) for pred in previous_decisions]
            histories.append(history)

        return self._seq_embedder.inputs_to_feed_dict(histories, vocabs.all_preds)

    @property
    def embeds(self):
        return self._embeds


################################
# Scorers

class SimplePredicateScorer(CandidateScorer):
    def __init__(self, query, predicate_embedder):
        """Given a query vector, compute logit scores for each predicate choice.

        Args:
            query (Tensor): the query tensor, of shape (batch_size, ?)
            predicate_embedder (CombinedPredicateEmbedder)
        """
        pred_embeds = predicate_embedder.embeds
        super(SimplePredicateScorer, self).__init__(query, pred_embeds, project_query=True)


class AttentionPredicateScorer(CandidateScorer):
    def __init__(self, query, predicate_embedder, utterance_embedder):
        attention = Attention(utterance_embedder.hidden_states,
                              query, project_query=True)
        # self._attention is already reserved for another purpose ...
        self._attention_on_utterance = attention
        pred_embeds = predicate_embedder.embeds
        super(AttentionPredicateScorer, self).__init__(attention.retrieved, pred_embeds, project_query=True)

    @property
    def attention_on_utterance(self):
        return self._attention_on_utterance


class SoftCopyPredicateScorer(Feedable, Scorer):
    def __init__(self, utterance_attention_weights, disable=False):
        self._disabled = disable

        if self._disabled:
            # just return all zeros
            self._batch_size = tf.placeholder(tf.int32, shape=[])
            self._num_candidates = tf.placeholder(tf.int32, shape=[])
            zeros = tf.zeros([self._batch_size, self._num_candidates], dtype=tf.float32)
            self._scores = SequenceBatch(values=zeros, mask=zeros)
        else:
            self._soft_copier = SoftCopyScorer(utterance_attention_weights)
            self._scores = self._soft_copier.scores

    @property
    def scores(self):
        return self._scores

    def _candidate_alignments(self, pred, utterance):
        aligns = utterance.predicate_alignment(pred)
        aligns = [(offset, s) for offset, s in aligns if offset < self._soft_copier.input_length]
        return aligns

    def inputs_to_feed_dict(self, utterances, choice_batch):
        """Feed inputs.

        Args:
            utterances: (list[Utterance])
            choice_batch (list[list[ContextualPredicate]])

        Returns:
            dict
        """
        if self._disabled:
            return {self._batch_size: len(utterances), self._num_candidates: max(len(c) for c in choice_batch)}

        alignments_batch = []
        for utterance, choices in zip(utterances, choice_batch):
            alignments = [self._candidate_alignments(contextual_pred.predicate, utterance) for contextual_pred in choices]
            alignments_batch.append(alignments)
        return self._soft_copier.inputs_to_feed_dict(alignments_batch)


class PredicateScorer(Feedable, Scorer):
    def __init__(self, simple_scorer, attention_scorer, soft_copy_scorer):
        """

        Args:
            simple_scorer (SimplePredicateScorer)
            attention_scorer (AttentionPredicateScorer)
            soft_copy_scorer (SoftCopyPredicateScorer)
        """
        assert isinstance(simple_scorer, SimplePredicateScorer)
        assert isinstance(attention_scorer, AttentionPredicateScorer)
        assert isinstance(soft_copy_scorer, SoftCopyPredicateScorer)

        simple_scores = simple_scorer.scores  # (batch_size, num_candidates)
        attention_scores = attention_scorer.scores  # (batch_size, num_candidates)
        soft_copy_scores = soft_copy_scorer.scores  # (batch_size, num_candidates)

        # check that Tensors are finite
        def verify_finite_inside_mask(scores, msg):
            finite_scores = scores.with_pad_value(0).values
            assert_op = tf.verify_tensor_all_finite(finite_scores, msg)
            return assert_op

        with tf.control_dependencies([
            verify_finite_inside_mask(simple_scores, 'simple_scores'),
            verify_finite_inside_mask(attention_scores, 'attention_scores'),
            verify_finite_inside_mask(soft_copy_scores, 'soft copy scores'),
        ]):
            scores = SequenceBatch(
                simple_scores.values + attention_scores.values + soft_copy_scores.values,
                simple_scores.mask)
            subscores = SequenceBatch(
                tf.pack(
                    [simple_scores.values, attention_scores.values, soft_copy_scores.values],
                    axis=2),
                simple_scores.mask)

        scores = scores.with_pad_value(-float('inf'))
        probs = SequenceBatch(tf.nn.softmax(scores.values), scores.mask)

        self._scores = scores
        self._subscores = subscores
        self._probs = probs

        self._simple_scorer = simple_scorer
        self._attention_scorer = attention_scorer
        self._soft_copy_scorer = soft_copy_scorer

    @property
    def scores(self):
        return self._scores

    @property
    def subscores(self):
        return self._subscores

    @property
    def probs(self):
        return self._probs

    @property
    def attention_on_utterance(self):
        return self._attention_scorer.attention_on_utterance

    def inputs_to_feed_dict(self, cases, vocabs):
        choice_batch = []
        for case in cases:
            utterance = case.current_utterance
            choices = [vocabs.as_contextual_pred(pred, utterance) for pred in case.choices]
            choice_batch.append(choices)

        utterances = [case.current_utterance for case in cases]
        pred_vocab = vocabs.all_preds

        feed_simple_scorer = self._simple_scorer.inputs_to_feed_dict(choice_batch, pred_vocab)
        feed_attention_scorer = self._attention_scorer.inputs_to_feed_dict(choice_batch, pred_vocab)
        feed_sc_scorer = self._soft_copy_scorer.inputs_to_feed_dict(utterances, choice_batch)
        feed = {}
        feed.update(feed_simple_scorer)
        feed.update(feed_attention_scorer)
        feed.update(feed_sc_scorer)
        return feed

    # WARNING:
    #     - The caching is only efficient if tuple(c.context for c in cases) is encountered frequently across batches
    #     - The caching is only correct if case.choices only depends on case.context
    inputs_to_feed_dict_cached = DictMemoized(inputs_to_feed_dict,
                                              custom_key_fxn=lambda self, cases, vocabs: tuple(c.current_utterance for c in cases))

    @property
    def simple_scorer(self):
        return self._simple_scorer

    @property
    def attention_scorer(self):
        return self._attention_scorer

    @property
    def soft_copy_scorer(self):
        return self._soft_copy_scorer


################################
# Full Models

class ParseModel(Feedable):
    """The NN responsible for scoring ParseCase choices.

    Given a ParseCase, it will return a logit score for every option in ParseCase.options.
    """

    def __init__(self, pred_embedder, history_embedder, stack_embedder, utterance_embedder, scorer_factory, h_dims,
                 domain, delexicalized):
        """ParseModel.

        Args:
            pred_embedder (CombinedPredicateEmbedder)
            history_embedder (HistoryEmbedder | None): if None, model won't condition on history of previous predictions
            stack_embedder (ExecutionStackEmbedder | None): if None, model won't condition on execution stack
            utterance_embedder (UtteranceEmbedder)
            scorer_factory (Callable[Tensor, PredicateScorer])
            h_dims (list[int])
            domain (str)
            delexicalized (bool)
        """
        # ReLU feedforward network
        with tf.name_scope('ParseModel'):
            state_embedders = [history_embedder, stack_embedder, utterance_embedder]
            state_embeds = []
            for state_embedder in state_embedders:
                if state_embedder:
                    state_embeds.append(state_embedder.embeds)

            self._input_layer = tf.concat(1, state_embeds)
            # (batch_size, hist_dim + stack_dim + utterance_dim)
            h = self._input_layer
            for h_dim in h_dims:
                h = Dense(h_dim, activation='relu')(h)  # (batch_size, h_dim)
            query = h

        self._case_encodings = query

        scorer = scorer_factory(query)

        self._domain = domain
        self._delexicalized = delexicalized
        self._pred_embedder = pred_embedder
        self._history_embedder = history_embedder
        self._stack_embedder = stack_embedder
        self._utterance_embedder = utterance_embedder
        self._scorer = scorer
        self._logits = scorer.scores.values
        self._attention_on_utterance = scorer.attention_on_utterance.logits
        self._sublogits = scorer.subscores.values
        self._log_probs = tf.nn.log_softmax(self._logits)
        self._probs = scorer.probs.values

        # track how many times we've called inputs_to_feed_dict with caching=True
        self._cache_calls = 0
        self._test_cache_every_k = 100

    @property
    def logits(self):
        return self._logits

    @property
    def log_probs(self):
        return self._log_probs

    @property
    def case_encodings(self):
        return self._case_encodings

    @property
    def probs(self):
        return self._probs

    def _compute_vocabs(self, utterances):
        """Compute Vocabs object.

        Args:
            utterances (frozenset[utterances])
        """
        return Vocabs(utterances, self._domain)

    _compute_vocabs_cached = DictMemoized(_compute_vocabs)

    def inputs_to_feed_dict(self, cases, ignore_previous_utterances, caching):
        feed = {}

        utterances = frozenset([case.current_utterance for case in cases])

        if self._delexicalized:
            for utterance in utterances:
                assert isinstance(utterance, DelexicalizedUtterance)

        if caching:
            vocabs = self._compute_vocabs_cached(utterances)
        else:
            vocabs = self._compute_vocabs(utterances)

        def feed_dict(model):
            if model is None:
                return lambda *args, **kwargs: {}  # nothing to be computed
            if caching:
                try:
                    return model.inputs_to_feed_dict_cached
                except AttributeError:
                    pass  # no cached version available
            return model.inputs_to_feed_dict

        feed.update(feed_dict(self._pred_embedder)(vocabs))
        feed.update(feed_dict(self._history_embedder)(cases, vocabs, ignore_previous_utterances))
        feed.update(feed_dict(self._stack_embedder)(cases))
        feed.update(feed_dict(self._utterance_embedder)(cases, vocabs.utterances))
        feed.update(feed_dict(self._scorer)(cases, vocabs))

        if caching:
            self._cache_calls += 1
            # every once in a while, check that the cache values are not stale
            if self._cache_calls % self._test_cache_every_k == 0:
                fresh_feed = self.inputs_to_feed_dict(cases, caching=False)
                for key in fresh_feed:
                    try:
                        assert_array_almost_equal(
                            fresh_feed[key], feed[key], decimal=3)
                    except Exception as e:
                        print('WTF', key)
                        print(cases)
                        print(fresh_feed[key])
                        print(feed[key])
                        raise e
                        # assert_array_collections_equal(fresh_feed, feed, decimal=4)

        return feed

    def score(self, cases, ignore_previous_utterances, caching):
        """Populate the choice_logits property for every ParseCase in the batch.

        Args:
            cases (list[ParseCase])
            ignore_previous_utterances (bool): if True, pretend like the previous utterances were not uttered
            caching (bool)
        """
        if len(cases) == 0:
            return

        # define variables to fetch
        fetch = {
            'logits': self._logits,
            'log_probs': self._log_probs,
        }
        if self._stack_embedder:
            fetch['stack_hash'] = self._stack_embedder.embeds_hash
        if self._history_embedder:
            fetch['history_hash'] = self._history_embedder.embeds_hash

        # fetch variables
        fetched = self.compute(fetch, cases, ignore_previous_utterances, caching)

        # unpack fetched values
        logits, log_probs = fetched['logits'], fetched['log_probs']  # numpy arrays with shape (batch_size, max_choices)
        stack_hash = fetched['stack_hash'] if self._stack_embedder else [None] * len(cases)
        history_hash = fetched['history_hash'] if self._history_embedder else [None] * len(cases)

        num_nans = lambda arr: np.sum(np.logical_not(np.isfinite(arr)))

        # cut to actual number of choices
        for i, case in enumerate(cases):
            case.choice_logits = logits[i, :len(case.choices)]
            case.choice_log_probs = log_probs[i, :len(case.choices)]
            case.pretty_embed = PrettyCaseEmbedding(history_hash[i], stack_hash[i])

            logit_nans = num_nans(case.choice_logits)
            log_prob_nans = num_nans(case.choice_log_probs)

            # Tracking NaN
            if logit_nans > 0:
                logging.error("logit NaNs: %d/%d", logit_nans, case.choice_logits.size)
            if log_prob_nans > 0:
                logging.error("log_prob NaNs: %d/%d", log_prob_nans, case.choice_log_probs.size)

    def score_paths(self, paths, ignore_previous_utterances, caching):
        cases_to_be_scored = []
        used_case_ids = set()
        for path in paths:
            for case in path:
                if id(case) not in used_case_ids:
                    cases_to_be_scored.append(case)
                    used_case_ids.add(id(case))
        self.score(cases_to_be_scored, ignore_previous_utterances, caching)

    def score_breakdown(self, cases, ignore_previous_utterances, caching):
        """Return the logits for all (parse case, choice, scorer) tuples.

        Args:
            cases (list[ParseCase])
            ignore_previous_utterances (bool): if True, pretend like the previous utterances were not uttered
            caching (bool)
        Returns:
            attention_on_utterance:
                np.array of shape (len(cases), max len(utterance))
                containing the attention score of each token.
            sublogits:
                np.array of shape (len(cases), max len(choices), number of scorers)
                containing the logits of each scorer on each choice.
                By default there are 3 scorers: basic, attention, and soft copy.
        """
        if len(cases) == 0:
            return []
        return self.compute([self._attention_on_utterance, self._sublogits], cases, ignore_previous_utterances, caching)


class CrossEntropyLossModel(Feedable):
    """Defines a standard cross entropy loss on the decision of a ParseCase."""

    def __init__(self, logits):
        """Define the loss model.

        Args:
            logits (Tensor): a tensor of shape (batch_size, max_choices)
        """
        with tf.name_scope('LossModel'):
            self._labels = tf.placeholder(tf.int32, shape=[None], name='labels')
            self._losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._labels, name='losses')

    def inputs_to_feed_dict(self, cases):
        """For each ParseCase, map case.decision to the appropriate placeholder.

        Args:
            cases (list[ParseCase])
        """
        labels = [c.choices.index(c.decision) for c in cases]
        return {self._labels: np.array(labels)}

    @property
    def losses(self):
        return self._losses


class LogitLossModel(Feedable):
    """Defines a loss based on the logit."""

    def __init__(self, logits):
        """Define the loss model.

        Args:
            logits (Tensor): a tensor of shape (batch_size, max_choices)
        """
        with tf.name_scope('LossModel'):
            self._labels = tf.placeholder(tf.int32, shape=[None], name='labels')
            # Pick out the correct logit terms using gather
            shape = tf.shape(logits)
            flattened_logits = tf.reshape(logits, [-1])
            self._losses = - tf.gather(flattened_logits,
                                       tf.range(shape[0]) * shape[1] + self._labels)

    def inputs_to_feed_dict(self, cases):
        """For each ParseCase, map case.decision to the appropriate placeholder.

        Args:
            cases (list[ParseCase])
        """
        labels = [c.choices.index(c.decision) for c in cases]
        return {self._labels: np.array(labels)}

    @property
    def losses(self):
        return self._losses


class TrainParseModel(Optimizable, Feedable):
    """A wrapper around a ParseModel for training."""

    def __init__(self, parse_model, loss_model_factory, learning_rate,
                 optimizer_opt, max_batch_size=None):
        loss_model = loss_model_factory(parse_model.logits)
        losses = loss_model.losses

        with tf.name_scope('TrainParseModel'):
            weights = tf.placeholder(tf.float32, [None])
            weighted_losses = losses * weights
            loss = tf.reduce_sum(weighted_losses)

            step = tf.get_variable('step', shape=[], dtype=tf.int32,
                                   initializer=tf.constant_initializer(0), trainable=False)

            increment_step_op = tf.assign_add(step, 1)

        optimizer = get_optimizer(optimizer_opt)(learning_rate)

        take_step = optimizer.minimize(loss, global_step=step)

        self._weights = weights
        self._loss = loss
        self._parse_model = parse_model
        self._loss_model = loss_model
        self._step = step
        self._increment_step = increment_step_op
        self._take_step = take_step

        # For batched computation
        self._max_batch_size = max_batch_size
        if max_batch_size is not None:
            self._grads_and_vars = optimizer.compute_gradients(loss)
            self._grad_tensors = []
            self._combined_grad_placeholders = []
            for grad, var in self._grads_and_vars:
                self._grad_tensors.append(tf.convert_to_tensor(grad))
                self._combined_grad_placeholders.append(tf.placeholder(tf.float32))
            self._apply_gradients = optimizer.apply_gradients(
                list(zip(self._combined_grad_placeholders,
                    [var for (_, var) in self._grads_and_vars])))

    @property
    def loss(self):
        return self._loss

    @property
    def parse_model(self):
        return self._parse_model

    @property
    def logits(self):
        return self.parse_model.logits

    def score(self, cases, ignore_previous_utterances, caching):
        return self.parse_model.score(cases, ignore_previous_utterances, caching)

    def score_breakdown(self, cases, ignore_previous_utterances, caching):
        return self.parse_model.score_breakdown(cases, ignore_previous_utterances, caching)

    def inputs_to_feed_dict(self, cases, weights, caching):
        """Convert a batch of ParseCases and their corresponding weights into a feed_dict.

        Args:
            cases (list[ParseCase])
            weights (list[float])

        Returns:
            feed_dict
        """
        feed = {}
        feed.update(self._loss_model.inputs_to_feed_dict(cases))
        feed.update(self._parse_model.inputs_to_feed_dict(cases, ignore_previous_utterances=False,
                                                          caching=caching))
        # when updating the model, we always acknowledge previous utterances

        feed[self._weights] = np.array(weights)
        return feed

    def train_step(self, cases, weights, caching):
        if len(cases) != len(weights):
            raise ValueError('cases and weights must have the same length.')

        if len(cases) == 0:
            #logging.warn('Training on zero cases.')
            print(" WARNING: Zero cases   \033[F", file=sys.stderr)
            # still increment the step
            sess = tf.get_default_session()
            sess.run(self._increment_step)
        elif not self._max_batch_size or len(cases) <= self._max_batch_size:
            print(" Updating ({} cases)   \033[F".format(len(cases)), file=sys.stderr)
            self.compute(self._take_step, cases, weights, caching)
        else:
            print(" Updating ({} cases)   \033[F".format(len(cases)), file=sys.stderr)
            assert not caching
            grads = None
            slices = list(range(0, len(cases), self._max_batch_size))
            for i in verboserate(slices, desc='Computing gradients ({} cases)'.format(len(cases))):
                cases_slice = cases[i:i + self._max_batch_size]
                weights_slice = weights[i:i + self._max_batch_size]
                grads_slice = self.compute(self._grad_tensors,
                                           cases_slice, weights_slice, False)
                if grads is None:
                    grads = grads_slice
                else:
                    for i in range(len(self._grad_tensors)):
                        grads[i] += grads_slice[i]
            sess = tf.get_default_session()
            feed_dict = dict(list(zip(self._combined_grad_placeholders, grads)))
            sess.run(self._apply_gradients, feed_dict)
            sess.run(self._increment_step)

    @property
    def step(self):
        return self._step.eval()

    @property
    def objective_tensor(self):
        return self.loss


def stack_element_category(elem):
    if isinstance(elem, (str, int)):
        return 'PRIMITIVE'
    elif isinstance(elem, RLongObject):
        return 'OBJECT'
    elif isinstance(elem, list):
        return 'LIST'
    else:
        raise ValueError('Stack element of unknown category: {}'.format(elem))


class StackObjectEmbedder(Feedable, metaclass=ABCMeta):
    @abstractproperty
    def embeds(self):
        """A Tensor of shape [batch_size, max_stack_size, object_embed_dim].

        Elements of each stack MUST be right-aligned!

        i.e. zero-padding should be on the left side.
        We want the topmost (rightmost) element of the stack to always appear in the same position.
        """
        pass

    @property
    def embed_dim(self):
        return self.embeds.get_shape().as_list()[2]


class StackOfAttributesEmbedder(Feedable):
    """Embed a batch of stacks, where each stack element is a list of attributes.

    Lists of attributes are embedded as the concatenation of their elements, with right padding.
    Lists that exceed max_list_size are truncated on the right.
    """

    def __init__(self, attribute_embedder, extract_attribute, max_stack_size, max_list_size):
        """

        Args:
            attribute_embedder (TokenEmbedder)
            extract_attribute (Callable[RLongObject, str]): extract a particular attribute from an RLongObject
            max_stack_size (int)
            max_list_size (int)
        """
        list_embedder = ConcatSequenceEmbedder(attribute_embedder.embeds, seq_length=max_list_size)
        list_embeds_flat = list_embedder.embeds  # (batch_size * max_stack_size, list_embed_dim)
                                                 # where list_embed_dim = attribute_embed_dim * max_list_size

        list_embeds_flat_shape = tf.shape(list_embeds_flat)
        batch_size = list_embeds_flat_shape[0] / max_stack_size  # a scalar Tensor, dynamically determined
        list_embed_dim = list_embeds_flat.get_shape().as_list()[1]  # a Python int, statically known

        # (batch_size, max_stack_size, list_embed_dim)
        self._embeds = tf.reshape(list_embeds_flat, shape=[batch_size, max_stack_size, list_embed_dim])
        self._attribute_embedder = attribute_embedder
        self._extract_attribute = extract_attribute
        self._list_embedder = list_embedder
        self._max_stack_size = max_stack_size

    @property
    def embeds(self):
        return self._embeds  # (batch_size, max_stack_size, list_embed_dim)

    def _pad_stack(self, stack):
        extra = self._max_stack_size - len(stack)
        assert extra >= 0  # stack should never exceed max_stack-size

        # always apply left-padding of the stack
        empty_list = []
        return [empty_list] * extra + stack

    def convert_to_attribute_stacks(self, exec_stacks):
        """Convert a batch of execution stacks into a batch of attribute stacks.

        Stack elements are converted as follows:
            A list of RLongObjects is converted into a list of attributes.
            A single RLongObject is converted into a single-item list.
            A primitive stack element is converted into an empty list.

        Args:
            exec_stacks (list[list[basestring|int|RLongObject|list[RLongObject]]]): a batch of execution stacks,
                where each stack element is either a primitive, RLongObject, or list of RLongObjects.

        Returns:
            attribute_stacks (list[list[list[str]]]): a batch of stacks, where each stack element is a list of
                attributes (as strings).
        """
        extract_attribute = self._extract_attribute

        attribute_stacks = []
        for stack in exec_stacks:
            attribute_stack = []
            for elem in stack:
                category = stack_element_category(elem)
                if category == 'PRIMITIVE':
                    attribute_list = []
                elif category == 'OBJECT':
                    attribute_list = [extract_attribute(elem)]
                elif category == 'LIST':
                    attribute_list = [extract_attribute(o) for o in elem]  # assume that list is a list of objects
                else:
                    raise ValueError('Cannot embed: {}'.format(elem))
                attribute_stack.append(attribute_list)
            attribute_stacks.append(attribute_stack)

        return attribute_stacks

    def inputs_to_feed_dict(self, exec_stacks):
        """Feed inputs.

        Args:
            attribute_stacks (list[list[list[str]]]): a batch of stacks, where each stack element is a list of
                attributes (as strings).

        Returns:
            feed_dict
        """
        attribute_stacks = self.convert_to_attribute_stacks(exec_stacks)

        sequences = []
        for stack in attribute_stacks:
            padded_stack = self._pad_stack(stack)
            for attribute_list in padded_stack:
                sequences.append(attribute_list)

        assert len(sequences) == len(exec_stacks) * self._max_stack_size
        return self._list_embedder.inputs_to_feed_dict(sequences, self._attribute_embedder.vocab)


class RLongObjectEmbedder(StackObjectEmbedder):
    def __init__(self, attribute_extractors, primitive_embedder, max_stack_size, max_list_size):
        embedders = [StackOfAttributesEmbedder(primitive_embedder, attribute_extractor, max_stack_size, max_list_size)
                     for attribute_extractor in attribute_extractors]

        # (batch_size, max_stack_size, max_list_size * primitive_embed_dim * len(embedders))
        self._embeds = tf.concat(2, [embedder.embeds for embedder in embedders])
        self._embedders = embedders

    @property
    def embeds(self):
        return self._embeds

    def inputs_to_feed_dict(self, exec_stacks):
        feed = {}
        for embedder in self._embedders:
            feed.update(embedder.inputs_to_feed_dict(exec_stacks))
        return feed


class ExecutionStackEmbedder(Embedder):
    def __init__(self, primitive_embedder, object_embedder, max_stack_size, max_list_size,
                 project_object_embeds=True, abstract_objects=False):
        """ExecutionStackEmbedder.

        Args:
            primitive_embedder (TokenEmbedder)
            object_embedder (StackObjectEmbedder)
            max_stack_size (int)
            max_list_size (int)
            project_object_embeds (bool): defaults to True. If True, project object embeddings into
                dimension of primitive embeddings.
            abstract_objects (bool): defaults to False. If True, just embed all objects using the
                same generic "object" token.
        """
        # get primitive and object embeds
        primitive_indices = FeedSequenceBatch(align='right', seq_length=max_stack_size)  # (batch_size, max_stack_size)
        primitive_embeds = embed(primitive_indices, primitive_embedder.embeds).values  # (batch_size, max_stack_size, embed_dim)
        object_embeds = object_embedder.embeds  # (batch_size, max_stack_size, object_embed_dim)

        # get Tensor shapes
        primitive_embed_dim = primitive_embedder.embed_dim
        object_embed_dim = object_embedder.embed_dim
        batch_size = tf.shape(primitive_indices.values)[0]

        # project object embeds into same dimension as primitive embeds
        if project_object_embeds:
            object_projection_layer = Dense(primitive_embed_dim, activation='linear')
            object_embeds_flat = tf.reshape(object_embeds, [batch_size * max_stack_size, object_embed_dim])
            projected_object_embeds_flat = object_projection_layer(object_embeds_flat)
            projected_object_embeds = tf.reshape(projected_object_embeds_flat, [batch_size, max_stack_size, primitive_embed_dim])
        else:
            object_projection_layer = None
            projected_object_embeds = object_embeds

        # combine primitive and object embeds
        is_object_feed = FeedSequenceBatch(align='right', seq_length=max_stack_size, dtype=tf.float32)
        is_object = is_object_feed.values  # (batch_size, max_stack_size)
        is_object = expand_dims_for_broadcast(is_object, primitive_embeds)  # (batch_size, max_stack_size, embed_dim)
        stack_embeds = is_object * projected_object_embeds + (1 - is_object) * primitive_embeds  # (batch_size, max_stack_size, embed_dim)

        # make sure to mask out empty stack positions
        stack_embeds = stack_embeds * expand_dims_for_broadcast(primitive_indices.mask, stack_embeds)

        flat_stack_embeds = tf.reshape(stack_embeds, [batch_size, max_stack_size * primitive_embed_dim])

        self._build_embeds_hash(stack_embeds, primitive_embed_dim)

        self._primitive_embedder = primitive_embedder
        self._object_embedder = object_embedder
        self._max_stack_size = max_stack_size
        self._max_list_size = max_list_size
        self._abstract_objects = abstract_objects

        self._object_projection_layer = object_projection_layer

        self._embeds = flat_stack_embeds
        self._primitive_indices = primitive_indices
        self._is_object_feed = is_object_feed

    def _build_embeds_hash(self, stack_embeds, embed_dim):
        # stack_embeds is (batch_size, max_stack_size, embed_dim)

        # random vector, initialized once and never trained
        hash_vector = tf.get_variable('exec_stack_hash_vector', shape=[embed_dim], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(seed=0), trainable=False)

        # inner product every stack embedding with the hash vector
        hash = tf.reshape(hash_vector, [1, 1, embed_dim])  # expand dims for broadcast
        self._embeds_hash = tf.reduce_sum(stack_embeds * hash, axis=2, keep_dims=False)  # (batch_size, max_stack_size)

    @property
    def embeds_hash(self):
        return self._embeds_hash

    def inputs_to_feed_dict(self, cases):
        OBJECT = self._primitive_embedder.vocab.OBJECT
        LIST = self._primitive_embedder.vocab.LIST
        abstract_objects = self._abstract_objects

        # collect batch of execution stacks
        exec_stacks = []
        for case in cases:
            previous_cases = case._previous_cases
            if len(previous_cases) == 0:
                exec_stack = []  # TODO(kelvin): always safe to assume stack starts out empty?
            else:
                latest_case = previous_cases[-1]
                exec_stack = latest_case.denotation.execution_stack  # use the denotation up until this point
            exec_stacks.append(exec_stack)

        is_object_batch = []
        primitive_stack_batch = []
        for exec_stack in exec_stacks:
            primitive_stack = []
            is_object = []
            for elem in exec_stack:
                category = stack_element_category(elem)
                if category == 'PRIMITIVE':
                    is_obj = 0.
                    primitive_val = elem
                elif category == 'OBJECT':
                    is_obj = 0. if abstract_objects else 1.  # if abstract_objects, embed it as a primitive instead
                    primitive_val = OBJECT
                elif category == 'LIST':
                    is_obj = 0. if abstract_objects else 1.  # if abstract_objects, embed it as a primitive instead
                    primitive_val = LIST if len(elem) != 1 else OBJECT  # singleton list treated as object
                else:
                    raise ValueError('Cannot embed: {}'.format(elem))

                is_object.append(is_obj)
                primitive_stack.append(primitive_val)

            primitive_stack_batch.append(primitive_stack)
            is_object_batch.append(is_object)

        primitive_feed = self._primitive_indices.inputs_to_feed_dict(primitive_stack_batch,
                                                                     self._primitive_embedder.vocab)
        object_feed = self._object_embedder.inputs_to_feed_dict(exec_stacks)
        is_object_feed = self._is_object_feed.inputs_to_feed_dict(is_object_batch)

        feed = {}
        feed.update(primitive_feed)
        feed.update(object_feed)
        feed.update(is_object_feed)
        return feed

    @property
    def embeds(self):
        """Tensor of shape [batch_size, max_stack_size * primitive_embed_dim]."""
        return self._embeds


class DummyStackObjectEmbedder(StackObjectEmbedder):
    """Just embeds every object as a vector of all ones.

    This is really just used as a placeholder model when we set abstract_objects=True in ExecutionStackEmbedder.
    In that scenario, the outputs of this embedder do not actually get used in the final embedding of the stack.
    """

    def __init__(self, max_stack_size, object_embed_dim):
        self._batch_size = tf.placeholder(tf.int32, shape=[])
        self._embeds = tf.ones([self._batch_size, max_stack_size, object_embed_dim])

    @property
    def embeds(self):
        return self._embeds

    def inputs_to_feed_dict(self, exec_stacks):
        return {self._batch_size: len(exec_stacks)}
