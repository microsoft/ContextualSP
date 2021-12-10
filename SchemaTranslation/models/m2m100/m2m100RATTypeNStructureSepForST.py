# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch M2M100 model. """
import math
import random
from typing import Optional

import models.modules.rat.layers.rat_sep as relational_aware_transformer
import torch
import torch.nn.functional as F
from torch import nn
from transformers import M2M100Config, M2M100Model
from transformers.file_utils import (
    add_start_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput)
from transformers.models.m2m_100 import M2M100PreTrainedModel
from transformers.models.m2m_100.modeling_m2m_100 import M2M_100_START_DOCSTRING, M2M100Decoder, \
    M2M100SinusoidalPositionalEmbedding, M2M100EncoderLayer, \
    _expand_mask
from transformers.utils import logging

from models.m2m100.m2m100ForST import M2M100ForST
from models.modules.rat.mask_utils import get_attn_mask
from models.modules.rat.relations.relation_extractor import RelationExtractor

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "M2M100Config"
_TOKENIZER_FOR_DOC = "M2M100Tokenizer"


@add_start_docstrings(
    "The M2M100 Model with a language modeling head. Can be used for summarization.", M2M_100_START_DOCSTRING
)
class M2M100RATTypeNStructureSepForST(M2M100ForST):

    def __init__(self, config: M2M100Config):
        super().__init__(config)
        print(type(self).__name__)
        self.model = M2M100RATModel(config)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()


@add_start_docstrings(
    "The bare M2M100 Model outputting raw hidden-states without any specific head on top.",
    M2M_100_START_DOCSTRING,
)
class M2M100RATModel(M2M100Model):
    def __init__(self, config: M2M100Config):
        super().__init__(config)
        print(type(self).__name__)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = M2M100RATEncoder(config, self.shared)
        self.decoder = M2M100Decoder(config, self.shared)

        self.init_weights()


class M2M100RATEncoder(M2M100PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`M2M100EncoderLayer`.

    Args:
        config: M2M100Config
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        print(type(self).__name__)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([M2M100EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Conroy
        self.structure_extractor = RelationExtractor(  # structure
                include_types=False,
                # include_types=True,
                include_structure=True,
                # include_structure=False,
                pad_id=config.relation_tokens["pad"],
                eos_id=config.relation_tokens["eos"],
                context_sep_id=config.relation_tokens["context_sep"],
                item_sep_id=config.relation_tokens["item_sep"],
                language_token_src_id=config.relation_tokens["language_token_en"])
        self.type_extractor = RelationExtractor(  # type
                # include_types=False,
                include_types=True,
                # include_structure=True,
                include_structure=False,
                pad_id=config.relation_tokens["pad"],
                eos_id=config.relation_tokens["eos"],
                context_sep_id=config.relation_tokens["context_sep"],
                item_sep_id=config.relation_tokens["item_sep"],
                language_token_src_id=config.relation_tokens["language_token_en"])
        self.relation_tokens = config.relation_tokens
        hidden_size = 1024
        num_heads = 8
        dropout = 0.1
        num_layers = config.num_rat_layers

        tie_layers = False
        ff_size = hidden_size * 4
        print(f"initializing {num_layers} layers RAT")
        self.relation_aware_encoder = relational_aware_transformer.Encoder(
            lambda: relational_aware_transformer.EncoderLayer(
                hidden_size,
                relational_aware_transformer.MultiHeadedAttentionWithRelations(
                    num_heads,
                    hidden_size,
                    dropout),
                relational_aware_transformer.PositionwiseFeedForward(
                    hidden_size,
                    ff_size,
                    dropout),
                self.structure_extractor.num_relations,
                self.type_extractor.num_relations,
                dropout),
            hidden_size,
            num_layers,
            tie_layers)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.M2M100Tokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids, inputs_embeds)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Conroy:
        # mask: 0 for tokens that are **masked** / same as attention_mask
        #       evidence can be found in models/modules/rat/rat.py line 103 masked_fill
        assert len(input_ids.shape) == 2  # batch, src_length
        sequence_lengths = torch.sum(~input_ids.eq(self.relation_tokens["pad"]), dim=1).tolist()  # {<pad>: 1}
        relation_mask = get_attn_mask(sequence_lengths, padding_at_front=False).type_as(input_ids)
        structures = self.structure_extractor.build_relations(input_ids).type_as(input_ids)  # extract structure
        types = self.type_extractor.build_relations(input_ids).type_as(input_ids)  # extract type-aware self-relations
        hidden_states = self.relation_aware_encoder(hidden_states, structures, types, mask=relation_mask)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



