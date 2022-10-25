from abc import abstractproperty, ABCMeta, abstractmethod

import tensorflow as tf
from keras.layers import Dense, LSTM

from gtd.ml.framework import Feedable, Model
from gtd.ml.seq_batch import FeedSequenceBatch, embed, reduce_mean, SequenceBatch, reduce_sum, weighted_sum, reduce_max
from gtd.ml.vocab import Vocab


class Embedder(Feedable, metaclass=ABCMeta):
    """A map from objects to embeddings."""

    @abstractproperty
    def embeds(self):
        """A Tensor of shape [vocab_size, :, ..., :]."""
        pass

    @property
    def embed_dim(self):
        return self.embeds.get_shape().as_list()[1]


class TokenEmbedder(Embedder):
    """An embedding model for simple token-like objects (such as words).

    The embedding matrix is a TensorFlow Variable, with one row for each token.
    """

    def __init__(self, simple_embeddings, var_name, trainable=True):
        """Create VariableEmbeddings.

        Args:
            simple_embeddings (SimpleEmbeddings): a gtd.vocab.SimpleEmbeddings object
            var_name (str): name for the Variable
            trainable (bool): whether the embedding matrix is trainable or not
        """
        vocab = simple_embeddings.vocab
        vocab_size = len(vocab)
        embed_dim = simple_embeddings.embed_dim
        embeds = tf.get_variable(var_name, shape=[vocab_size, embed_dim],
                                 initializer=tf.constant_initializer(simple_embeddings.array), trainable=trainable)

        self._embeds = embeds
        self._embed_dim = embed_dim
        self._vocab = vocab

    @property
    def vocab(self):
        return self._vocab

    @property
    def embeds(self):
        return self._embeds

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def vocab_size(self):
        return len(self.vocab)

    def inputs_to_feed_dict(self, *args, **kwargs):
        return {}


class SequenceEmbedder(Embedder, metaclass=ABCMeta):
    """An embedding matrix for objects that can be represented as sequences (such as sentences)."""

    def __init__(self, token_embeds, align='left', seq_length=None, name='SequenceEmbedder'):
        """Create a SequenceEmbeddings object.

        Args:
            token_embeds (Tensor): a Tensor of shape (token_vocab_size, token_dim)
            align (str): see FeedSequenceBatch
            seq_length (int): see FeedSequenceBatch
        """
        with tf.name_scope(name):
            sequence_batch = FeedSequenceBatch(align=align, seq_length=seq_length)  # (sequence_vocab_size, seq_length)
            embedded_sequence_batch = embed(sequence_batch, token_embeds)
            embeds = self.embed_sequences(embedded_sequence_batch)

        self._sequence_batch = sequence_batch
        self._embedded_sequence_batch = embedded_sequence_batch
        self._embeds = embeds

    @abstractmethod
    def embed_sequences(self, embedded_sequence_batch):
        """Convert an embedded SequenceBatch into a Tensor of sequence embeddings.

        Args:
            embedded_sequence_batch (gtd.ml.seq_batch.SequenceBatch): a SequenceBatch of shape
                [seq_vocab_size, seq_length, token_dim]

        Returns:
            sequence_embeds (Tensor): of shape [seq_vocab_size, seq_dim]
        """
        pass

    def inputs_to_feed_dict(self, sequences, token_vocab):
        """Feed sequences.

        Args:
            sequences (list[list[unicode]]): a list of sequences
            token_vocab (SimpleVocab): a map from token names to integers

        Returns:
            feed_dict
        """
        return self._sequence_batch.inputs_to_feed_dict(sequences, token_vocab)

    @property
    def embeds(self):
        return self._embeds


class MeanSequenceEmbedder(SequenceEmbedder):
    def __init__(self, token_embeds, align='left', seq_length=None, allow_empty=False, name='MeanSequenceEmbedder'):
        """MeanSequenceEmbedder.

        Args:
            allow_empty (bool): allow computing the average of an empty sequence. In this case, we assume 0/0 == 0,
            rather than NaN. Default is False, causing an error to be thrown.
            (see SequenceEmbedder for other args)
        """
        self._allow_empty = allow_empty
        super(MeanSequenceEmbedder, self).__init__(token_embeds, align=align, seq_length=seq_length, name=name)

    def embed_sequences(self, embedded_sequence_batch):
        return reduce_mean(embedded_sequence_batch, allow_empty=self._allow_empty)


class MaxSequenceEmbedder(SequenceEmbedder):
    def embed_sequences(self, embedded_sequence_batch):
        return reduce_max(embedded_sequence_batch)


class ConcatSequenceEmbedder(SequenceEmbedder):
    def embed_sequences(self, embedded_sequence_batch):
        values = embedded_sequence_batch.values
        shape = tf.shape(values)
        nrows, ncols = shape[0], shape[1] * shape[2]
        new_shape = tf.pack([nrows, ncols])
        result = tf.reshape(values, new_shape)  # (batch_size, seq_length * embed_dim)

        # add static shape info
        batch_dim, seq_length_dim, token_dim = values.get_shape()
        concat_dim = token_dim * seq_length_dim
        result.set_shape(tf.TensorShape([batch_dim, concat_dim]))

        return result


class Attention(Model):
    """Implements standard attention.

    Given some memory, a memory mask and a query, outputs the weighted memory cells.
    """

    def __init__(self, memory_cells, query, project_query=False):
        """Define Attention.

        Args:
            memory_cells (SequenceBatch): a SequenceBatch containing a Tensor of shape (batch_size, num_cells, cell_dim)
            query (Tensor): a tensor of shape (batch_size, query_dim).
            project_query (bool): defaults to False. If True, the query goes through an extra projection layer to
                coerce it to cell_dim.
        """
        cell_dim = memory_cells.values.get_shape().as_list()[2]
        if project_query:
            # project the query up/down to cell_dim
            self._projection_layer = Dense(cell_dim, activation='linear')
            query = self._projection_layer(query)  # (batch_size, cand_dim)

        memory_values, memory_mask = memory_cells.values, memory_cells.mask

        # batch matrix multiply to compute logit scores for all choices in all batches
        query = tf.expand_dims(query, 2)  # (batch_size, cell_dim, 1)
        logit_values = tf.batch_matmul(memory_values, query)  # (batch_size, num_cells, 1)
        logit_values = tf.squeeze(logit_values, [2])  # (batch_size, num_cells)

        # set all pad logits to negative infinity
        logits = SequenceBatch(logit_values, memory_mask)
        logits = logits.with_pad_value(-float('inf'))

        # normalize to get probs
        probs = tf.nn.softmax(logits.values)  # (batch_size, num_cells)

        retrieved = tf.batch_matmul(tf.expand_dims(probs, 1), memory_values)  # (batch_size, 1, cell_dim)
        retrieved = tf.squeeze(retrieved, [1])  # (batch_size, cell_dim)

        self._logits = logits.values
        self._probs = probs
        self._retrieved = retrieved

    @property
    def logits(self):
        return self._logits  # (batch_size, num_cells)

    @property
    def probs(self):
        return self._probs  # (batch_size, num_cells)

    @property
    def retrieved(self):
        return self._retrieved  # (batch_size, cell_dim)

    @property
    def projection_weights(self):
        """Get projection weights.

        Returns:
            (np.array, np.array): a pair of numpy arrays, (W, b) used to project the query tensor to
                match the predicate embedding dimension.
        """
        return self._projection_layer.get_weights()

    @projection_weights.setter
    def projection_weights(self, value):
        W, b = value
        self._projection_layer.set_weights([W, b])


class Scorer(Model, metaclass=ABCMeta):
    @abstractproperty
    def scores(self):
        """Return a SequenceBatch."""
        pass


class CandidateScorer(Feedable, Scorer):
    def __init__(self, query, cand_embeds, project_query=False):
        """Create a CandidateScorer.

        Args:
            query (Tensor): of shape (batch_size, query_dim)
            cand_embeds (Tensor): of shape (cand_vocab_size, cand_dim)
            project_query (bool): whether to project the query tensor to match the dimension of the cand_embeds
        """
        with tf.name_scope("CandidateScorer"):
            cand_batch = FeedSequenceBatch()
            embedded_cand_batch = embed(cand_batch, cand_embeds)  # (batch_size, num_candidates, cand_dim)
            attention = Attention(embedded_cand_batch, query, project_query=project_query)

        self._attention = attention
        self._cand_batch = cand_batch
        self._scores = SequenceBatch(attention.logits, cand_batch.mask)
        self._probs = SequenceBatch(attention.probs, cand_batch.mask)

    @property
    def probs(self):
        return self._probs

    @property
    def scores(self):
        return self._scores

    @property
    def projection_weights(self):
        return self._attention.projection_weights

    @projection_weights.setter
    def projection_weights(self, value):
        self._attention.projection_weights = value

    def inputs_to_feed_dict(self, candidates, cand_vocab):
        """Feed inputs.

        Args:
            candidates (list[list[unicode]]): a batch of sequences, where each sequence is a unique set of candidates.
            cand_vocab (Vocab): a map from a candidate string to an int

        Returns:
            feed_dict
        """
        return self._cand_batch.inputs_to_feed_dict(candidates, cand_vocab)


class SoftCopyScorer(Feedable, Scorer):
    def __init__(self, input_scores):
        """Align a candidate with elements of the input, and define its score to be the summed score of aligned inputs.

        Args:
            input_scores (Tensor): of shape (batch_size, input_length)
        """
        input_scores_flat = tf.reshape(input_scores, shape=[-1])  # (batch_size * input_length,)
        self._input_length = input_scores.get_shape().as_list()[1]

        alignments_flat = FeedSequenceBatch()  # (total_candidates, max_alignments)
        alignment_weights_flat = FeedSequenceBatch(dtype=tf.float32)  # (total_candidates, max_alignments)

        aligned_attention_weights = embed(alignments_flat, input_scores_flat)  # (total_candidates, max_alignments)
        scores_flat = weighted_sum(aligned_attention_weights, alignment_weights_flat.with_pad_value(0).values)  # (total_candidates,)

        unflatten = FeedSequenceBatch()  # (batch_size, num_candidates)
        scores = embed(unflatten, scores_flat).with_pad_value(0)  # (batch_size, num_candidates)

        self._alignments_flat = alignments_flat
        self._alignment_weights_flat = alignment_weights_flat
        self._unflatten = unflatten
        self._scores = scores

    @property
    def input_length(self):
        return self._input_length

    @property
    def scores(self):
        """A SequenceBatch."""
        return self._scores

    def inputs_to_feed_dict(self, alignments):
        """Feed inputs.

        Args:
            alignments (list[list[list[(int, float)]]]): alignments[i][j] is a list of alignments for candidate j
                of example i. Each alignment is an (idx, strength) pair. `idx` corresponds to a position in the input
                sequence. `strength` is a float.

        Returns:
            a feed_dict
        """
        alignments_flat = []
        alignment_weights_flat = []
        unflatten = []
        flat_idx = 0
        for ex_idx, ex_alignments in enumerate(alignments):  # loop over examples
            uf = []
            for aligns in ex_alignments:  # loop over candidates
                if len(aligns) > 0:
                    positions, strengths = [list(l) for l in zip(*aligns)]
                    if max(positions) > (self._input_length - 1):
                        raise ValueError("alignment positions must not exceed input length")
                else:
                    positions, strengths = [], []

                offset = ex_idx * self.input_length
                positions_flat = [offset + i for i in positions]

                alignments_flat.append(positions_flat)
                alignment_weights_flat.append(strengths)

                uf.append(flat_idx)
                flat_idx += 1

            unflatten.append(uf)

        feed = {}
        feed.update(self._alignments_flat.inputs_to_feed_dict(alignments_flat))
        feed.update(self._alignment_weights_flat.inputs_to_feed_dict(alignment_weights_flat))
        feed.update(self._unflatten.inputs_to_feed_dict(unflatten))
        return feed


class LSTMSequenceEmbedder(SequenceEmbedder):
    """Forward LSTM Sequence Embedder

    Also provide attention states.
    """
    def __init__(self, token_embeds, seq_length, align='left', name='LSTMSequenceEmbedder', hidden_size=50):
        self.hidden_size = hidden_size
        super(LSTMSequenceEmbedder, self).__init__(token_embeds, align=align, seq_length=seq_length, name=name)

    def embed_sequences(self, embed_sequence_batch):
        self._forward_lstm = LSTM(self.hidden_size, return_sequences=True)
        # Pass input through the LSTMs
        # Shape: (batch_size, seq_length, hidden_size)
        hidden_state_values = self._forward_lstm(embed_sequence_batch.values, embed_sequence_batch.mask)
        self._hidden_states = SequenceBatch(hidden_state_values, embed_sequence_batch.mask)

        # Embedding dimension: (batch_size, hidden_size)
        shape = tf.shape(embed_sequence_batch.values)
        forward_final = tf.slice(hidden_state_values, [0, shape[1] - 1, 0], [-1, 1, self.hidden_size])
        return tf.squeeze(forward_final, [1])

    @property
    def weights(self):
        return self._forward_lstm.get_weights()

    @weights.setter
    def weights(self, w):
        self._forward_lstm.set_weights(w)

    @property
    def hidden_states(self):
        return self._hidden_states


class BidiLSTMSequenceEmbedder(SequenceEmbedder):
    """Bidirectional LSTM Sequence Embedder

    Also provide attention states.
    """
    def __init__(self, token_embeds, seq_length, align='left', name='BidiLSTMSequenceEmbedder', hidden_size=50):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        super(BidiLSTMSequenceEmbedder, self).__init__(token_embeds, align=align, seq_length=seq_length, name=name)

    def embed_sequences(self, embed_sequence_batch):
        """Return sentence embeddings as a tensor with with shape
        [batch_size, hidden_size * 2]
        """
        forward_values = embed_sequence_batch.values
        forward_mask = embed_sequence_batch.mask
        backward_values = tf.reverse(forward_values, [False, True, False])
        backward_mask = tf.reverse(forward_mask, [False, True])
        # Initialize LSTMs
        self._forward_lstm = LSTM(self.hidden_size, return_sequences=True)
        self._backward_lstm = LSTM(self.hidden_size, return_sequences=True)
        # Pass input through the LSTMs
        # Shape: (batch_size, seq_length, hidden_size)
        forward_seq = self._forward_lstm(forward_values, forward_mask)
        forward_seq.set_shape((None, self.seq_length, self.hidden_size))
        backward_seq = self._backward_lstm(backward_values, backward_mask)
        backward_seq.set_shape((None, self.seq_length, self.hidden_size))
        # Stitch the outputs together --> hidden states (for computing attention)
        # Final dimension: (batch_size, seq_length, hidden_size * 2)
        lstm_states = tf.concat(2, [forward_seq, tf.reverse(backward_seq, [False, True, False])])
        self._hidden_states = SequenceBatch(lstm_states, forward_mask)
        # Stitch the final outputs together --> sequence embedding
        # Final dimension: (batch_size, hidden_size * 2)
        seq_length = tf.shape(forward_values)[1]
        forward_final = tf.slice(forward_seq, [0, seq_length - 1, 0], [-1, 1, self.hidden_size])
        backward_final = tf.slice(backward_seq, [0, seq_length - 1, 0], [-1, 1, self.hidden_size])
        return tf.squeeze(tf.concat(2, [forward_final, backward_final]), [1])

    @property
    def weights(self):
        return (self._forward_lstm.get_weights(), self._backward_lstm.get_weights())

    @weights.setter
    def weights(self, w):
        forward_weights, backward_weights = w
        self._forward_lstm.set_weights(forward_weights)
        self._backward_lstm.set_weights(backward_weights)

    @property
    def hidden_states(self):
        """Return a SequenceBatch whose value has shape
        [batch_size, max_seq_length, hidden_size * 2]
        """
        return self._hidden_states
