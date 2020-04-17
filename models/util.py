# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
from typing import Tuple


def get_span_representation(forward_encoder_out, backward_encoder_out, span_start, span_end):
    """
    Given a span start/end position, fetch the subtraction representation of the span from LSTM.
    """
    # span end is always larger than actual value
    span_end -= 1
    forward_span_repr = get_forward_span_repr(forward_encoder_out, span_start, span_end)
    backward_span_repr = get_backward_span_repr(backward_encoder_out, span_start, span_end)
    # cat two representations
    span_repr = torch.cat((forward_span_repr, backward_span_repr))
    return span_repr


def get_forward_span_repr(forward_encoder_out, span_start, span_end):
    """
    Get forward span representation
    """
    if span_end >= len(forward_encoder_out):
        span_end = len(forward_encoder_out) - 1

    assert span_start <= span_end
    if span_start == 0:
        forward_span_repr = forward_encoder_out[span_end]
    else:
        forward_span_repr = forward_encoder_out[span_end] - forward_encoder_out[span_start - 1]
    return forward_span_repr


def get_backward_span_repr(backward_encoder_out, span_start, span_end):
    """
    Get backward span representation
    """
    assert span_start <= span_end

    if span_end >= len(backward_encoder_out) - 1:
        backward_span_repr = backward_encoder_out[span_start]
    else:
        backward_span_repr = backward_encoder_out[span_start] - backward_encoder_out[span_end + 1]
    return backward_span_repr


def find_start_end(cus_list, pattern) -> Tuple:
    """
    Find the start & end of pattern in cus_list. If none, return 0,0.
    :param cus_list:
    :param pattern:
    :return:
    """
    for i in range(len(cus_list)):
        if cus_list[i] == pattern[0] and cus_list[i:i + len(pattern)] == pattern:
            return i, i + len(pattern)
    return 0, 0
