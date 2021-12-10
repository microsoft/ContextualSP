import torch


def get_attn_mask(seq_lengths, padding_at_front=False):
    # Given seq_lengths like [3, 1, 2], when padding_at_front=False, this will produce
    # [[[1, 1, 1],
    #   [1, 1, 1],
    #   [1, 1, 1]],
    #  [[1, 0, 0],
    #   [0, 0, 0],
    #   [0, 0, 0]],
    #  [[1, 1, 0],
    #   [1, 1, 0],
    #   [0, 0, 0]]]
    # int(max(...)) so that it has type 'int instead of numpy.int64
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.LongTensor(batch_size, max_length, max_length).fill_(0)
    for batch_idx, seq_length in enumerate(seq_lengths):
        if padding_at_front:
            attn_mask[batch_idx, max_length - seq_length:, max_length - seq_length:] = 1
        else:  # padding at the end of the sentence
            attn_mask[batch_idx, :seq_length, :seq_length] = 1
    return attn_mask
