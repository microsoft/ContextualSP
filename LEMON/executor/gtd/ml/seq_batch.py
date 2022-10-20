import numpy as np
import tensorflow as tf

from gtd.ml.framework import Feedable
from gtd.ml.utils import expand_dims_for_broadcast, broadcast


class SequenceBatch(object):
    """Represent a batch of sequences as a Tensor."""

    def __init__(self, values, mask, name='SequenceBatch'):
        with tf.name_scope(name):
            # check that dimensions are correct
            values_shape = tf.shape(values)
            mask_shape = tf.shape(mask)
            values_shape_prefix = tf.slice(values_shape, [0], [2])
            max_rank = max(values.get_shape().ndims, mask.get_shape().ndims)

            assert_op = tf.assert_equal(values_shape_prefix, mask_shape,
                                        data=[values_shape_prefix, mask_shape], summarize=max_rank,
                                        name="assert_shape_prefix")

            with tf.control_dependencies([assert_op]):
                    self._values = tf.identity(values, name='values')
                    self._mask = tf.identity(mask, name='mask')

    @property
    def values(self):
        """A Tensor holding the values of the sequence batch, of shape [batch_size, seq_length, :, ..., :].

        Each row represents one sequence.
        """
        return self._values

    @property
    def mask(self):
        """A boolean mask of shape [batch_size, seq_length], indicating which entries of self.values are padding.

        mask[i, j] = 0 if the entry is padding, 1 otherwise.

        Returns:
            A Tensor of shape (batch_size, seq_length)
        """
        return self._mask

    def with_pad_value(self, val):
        """Return a new SequenceBatch, with pad values set to the specified value."""
        return SequenceBatch(change_pad_value(self.values, self.mask, val), self.mask)


def change_pad_value(values, mask, pad_val):
    """Given a set of values and a pad mask, change the value of all pad entries.

    Args:
        values (Tensor): of shape [batch_size, seq_length, :, ..., :].
        mask (Tensor): binary float tensor of shape [batch_size, seq_length]
        pad_val (float): value to set all pad entries to

    Returns:
        Tensor: a new Tensor of same shape as values
    """
    # broadcast the mask to match shape of values
    mask = expand_dims_for_broadcast(mask, values)  # (batch_size, seq_length, 1, ..., 1)
    mask = broadcast(mask, values)
    mask = tf.cast(mask, tf.bool)  # cast to bool

    # broadcast val
    broadcast_val = pad_val * tf.ones(tf.shape(values))

    new_values = tf.select(mask, values, broadcast_val)
    return new_values


class FeedSequenceBatch(Feedable, SequenceBatch):
    """A SequenceBatch that is fed into TensorFlow from the outside.

    The SequenceBatch is represented by a Tensor of shape [batch_size, seq_length]
        - batch_size is dynamically determined by the # sequences fed
        - seq_length is dynamically set to the length of the longest sequence fed, or the statically specified value.
    """
    def __init__(self, align='left', seq_length=None, dtype=tf.int32, name='FeedSequenceBatch'):
        """Create a Feedable SequenceBatch.

        Args:
            align (str): can be 'left' or 'right'. If 'left', values will be left-aligned, with padding on the right.
                If 'right', values will be right-aligned, with padding on the left. Default is 'left'.
            seq_length (int): the Tensor representing the SequenceBatch will have exactly this many columns. Default
                is None. If None, seq_length will be dynamically determined.
            dtype: data type of the SequenceBatch values array. Defaults to int32.
            name (str): namescope for the Tensors created inside this Model.
        """
        if align not in ('left', 'right'):
            raise ValueError("align must be either 'left' or 'right'.")
        self._align_right = (align == 'right')
        self._seq_length = seq_length

        with tf.name_scope(name):
            values = tf.placeholder(dtype, shape=[None, None], name='values')  # (batch_size, seq_length)
            mask = tf.placeholder(tf.float32, shape=[None, None], name='mask')  # (batch_size, seq_length)

        if self._seq_length is not None:
            # add static shape information
            batch_dim, _ = values.get_shape()
            new_shape = tf.TensorShape([batch_dim, tf.Dimension(seq_length)])
            values.set_shape(new_shape)
            mask.set_shape(new_shape)

        super(FeedSequenceBatch, self).__init__(values, mask)

    def inputs_to_feed_dict(self, sequences, vocab=None):
        """Convert sequences into a feed_dict.

        Args:
            sequences (list[list[unicode]]): a list of unicode sequences
            vocab (Vocab): a vocab mapping tokens to integers. If vocab is None, sequences are directly passed
                into TensorFlow, without performing any token-to-integer lookup.

        Returns:
            a feed_dict
        """
        batch_size = len(sequences)
        if batch_size == 0:
            seq_length = 0 if self._seq_length is None else self._seq_length
            empty = np.empty((0, seq_length))
            return {self.values: empty, self.mask: empty}

        # dynamic seq_length if none specified
        if self._seq_length is None:
            seq_length = max(len(tokens) for tokens in sequences)
        else:
            seq_length = self._seq_length

        # if no vocab, just pass the raw value
        if vocab is None:
            tokens_to_values = lambda words: words
        else:
            tokens_to_values = vocab.words2indices

        if self._align_right:
            truncate = lambda tokens: tokens[-seq_length:]
            indices = [[(seq_length - n) + i for i in range(n)] for n in range(seq_length + 1)]
        else:
            truncate = lambda tokens: tokens[:seq_length]
            indices = list(map(range, list(range(seq_length + 1))))

        values_arr = np.zeros((batch_size, seq_length), dtype=np.float32)
        mask_arr = np.zeros((batch_size, seq_length), dtype=np.float32)

        for row_idx, tokens in enumerate(sequences):
            num_tokens = len(tokens)
            if num_tokens == 0:
                continue

            if num_tokens > seq_length:
                truncated_tokens = truncate(tokens)
            else:
                truncated_tokens = tokens

            inds = indices[len(truncated_tokens)]
            vals = tokens_to_values(truncated_tokens)
            values_arr[row_idx][inds] = vals
            mask_arr[row_idx][inds] = 1.0

        return {self.values: values_arr, self.mask: mask_arr}


def embed(sequence_batch, embeds):
    mask = sequence_batch.mask
    embedded_values = tf.gather(embeds, sequence_batch.values)
    embedded_values = tf.verify_tensor_all_finite(embedded_values, 'embedded_values')

    # set all pad embeddings to zero
    broadcasted_mask = expand_dims_for_broadcast(mask, embedded_values)
    embedded_values *= broadcasted_mask

    return SequenceBatch(embedded_values, mask)


def reduce_mean(seq_batch, allow_empty=False):
    """Compute the mean of each sequence in a SequenceBatch.

    Args:
        seq_batch (SequenceBatch): a SequenceBatch with the following attributes:
            values (Tensor): a Tensor of shape (batch_size, seq_length, :, ..., :)
            mask (Tensor): if the mask values are arbitrary floats (rather than binary), the mean will be
            a weighted average.
        allow_empty (bool): allow computing the average of an empty sequence. In this case, we assume 0/0 == 0, rather
            than NaN. Default is False, causing an error to be thrown.

    Returns:
        Tensor: of shape (batch_size, :, ..., :)
    """
    values, mask = seq_batch.values, seq_batch.mask
    # compute weights for the average
    sums = tf.reduce_sum(mask, 1, keep_dims=True)  # (batch_size, 1)

    if allow_empty:
        asserts = []  # no assertion
        sums = tf.select(tf.equal(sums, 0), tf.ones(tf.shape(sums)), sums)  # replace 0's with 1's
    else:
        asserts = [tf.assert_positive(sums)]  # throw error if 0's exist

    with tf.control_dependencies(asserts):
        weights = mask / sums  # (batch_size, seq_length)
    return weighted_sum(seq_batch, weights)


def reduce_sum(seq_batch):
    weights = tf.ones(shape=tf.shape(seq_batch.mask))
    return weighted_sum(seq_batch, weights)


def weighted_sum(seq_batch, weights):
    """Compute the weighted sum of each sequence in a SequenceBatch.

    Args:
        seq_batch (SequenceBatch): a SequenceBatch.
        weights (Tensor): a Tensor of shape (batch_size, seq_length). Determines the weights. Weights outside the
            seq_batch's mask are ignored.

    Returns:
        Tensor: of shape (batch_size, :, ..., :)
    """
    values, mask = seq_batch.values, seq_batch.mask
    weights = weights * mask  # ignore weights outside the mask
    weights = expand_dims_for_broadcast(weights, values)
    weighted_array = values * weights  # (batch_size, seq_length, X)
    return tf.reduce_sum(weighted_array, 1)  # (batch_size, X)


def reduce_max(seq_batch):
    sums = tf.reduce_sum(seq_batch.mask, 1, keep_dims=True)  # (batch_size, 1)
    with tf.control_dependencies([tf.assert_positive(sums)]):  # avoid dividing by zero
        seq_batch = seq_batch.with_pad_value(float('-inf'))  # set pad values to -inf
        result = tf.reduce_max(seq_batch.values, 1)
    return result