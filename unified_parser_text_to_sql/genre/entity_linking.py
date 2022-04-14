# Copyright (c) Facebook, Inc. and Microsoft Corporation.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List

import torch
from genre.trie import Trie

keyword = ['select', 'distinct', 'from', 'join', 'on', 'where', 'group', 'by', 'order', 'asc', 'desc', 'limit',
           'having',
           'and', 'not', 'or', 'like', 'between', 'in',
           'sum', 'count', 'max', 'min', 'avg',
           '(', ')', ',', '>', '<', '=', '>=', '!=', '<=',
           'union', 'except', 'intersect',
           '1', '2', '3', '4', '5']


def get_end_to_end_prefix_allowed_tokens_fn_hf(
        model,
        sentences: List[str],
        start_mention_token="{",
        end_mention_token="}",
        start_entity_token="[",
        end_entity_token="]",
        mention_trie: Trie = None,
        candidates_trie: Trie = None,
        mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.tokenizer.encode(x),
        lambda x: model.tokenizer.decode(torch.tensor(x)),
        model.tokenizer.bos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.eos_token_id,
        len(model.tokenizer) - 1,
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def get_end_to_end_prefix_allowed_tokens_fn_fairseq(
        model,
        sentences: List[str],
        start_mention_token="{",
        end_mention_token="}",
        start_entity_token="[",
        end_entity_token="]",
        mention_trie: Trie = None,
        candidates_trie: Trie = None,
        mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
        encode_fn,
        decode_fn,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        vocabulary_length,
        sentences: List[str],
        start_mention_token="{",
        end_mention_token="}",
        start_entity_token="[",
        end_entity_token="]",
        mention_trie: Trie = None,
        candidates_trie: Trie = None,
        mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    assert not (
            candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    codes = {}
    codes["EOS"] = eos_token_id
    codes["BOS"] = bos_token_id

    keyword_codes = {k: encode_fn(" {}".format(k))[1] for k in keyword}
    keyword_codes['wselect'] = encode_fn("{}".format('select'))[1]

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        trie_out = get_trie_schema(sent)
        return trie_out

    def get_trie_schema(sent):
        pointer_start = get_keyword_mention(sent)
        keyword_rnt = list(keyword_codes.values())

        if pointer_start + 1 < len(sent) and pointer_start != -1:
            ment_next = mention_trie.get(sent[pointer_start + 1:])
            if codes["EOS"] in ment_next:
                return ment_next + keyword_rnt
            else:
                return ment_next
        else:
            ment_next = mention_trie.get([])
            return ment_next + keyword_rnt + [codes["EOS"]]

    def get_keyword_mention(sent):
        pointer_start = -1
        for i, e in enumerate(sent):
            if e in keyword_codes.values():
                pointer_start = i
        return pointer_start

    return prefix_allowed_tokens_fn
