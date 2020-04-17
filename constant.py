# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class SpecialSymbol:
    copy_delimiter = ' [COPY] '


class CacheMethod:
    pick = "pickle"
    dil = "dill"


class CacheMode:
    all = "all"
    single = "single"


class ContextMode:
    turn_model = "turn"
    concat_history = "concat"
    context_independent = "none"
    # not support now
    concat_previous = "prev"
    copy_hard_token = "hard_token"
    concat_hard_token = "concat_hard_token"


class CopyMode:
    # WARNING: CopyMode actually does not work in the code
    # it corresponds to the setting `use_copy_segment` or `use_copy_token` though
    # here we make it behave as a constant to better reading
    no_copy = "none"
    copy_tree = "seg"
    copy_token = "token"
    copy_segment_with_context = "conseg"
    copy_token_with_context = "contoken"
    copy_mix_segment = "mixseg"


class BERTMode:
    no_bert = "v0"
    bert_with_table = "v3"
