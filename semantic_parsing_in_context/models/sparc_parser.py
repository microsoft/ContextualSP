# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import sys
import traceback
from collections import namedtuple
from typing import Dict, List, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import Vocabulary
from context.copy_production_rule_field import CopyProductionRule
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, Attention
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from torch.nn.modules import Dropout
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.training.metrics import Average
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import util
from allennlp.state_machines import BeamSearch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from allennlp.common.checks import check_dimensions_match
from context.converter import ActionConverter
from context.grammar import Statement, Action, Segment
from context.world import SparcWorld
from models.decode_trainer import MaximumMarginalLikelihood
from models.metrics import MetricUtil, TurnAverage
from models.states_machine.grammar_based_state import GrammarStatelet, GrammarBasedState, RnnStatelet, ConditionStatelet
from models.transition_functions.linking_transition_function import LinkingTransitionFunction
from models.util import get_span_representation, find_start_end
from copy import deepcopy

VALIDATE_SIZE = 422


@Model.register('sparc')
class SparcParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_encoder: Seq2SeqEncoder,
                 decoder_beam_search: BeamSearch,
                 input_attention: Attention,
                 text_embedder: TextFieldEmbedder,
                 max_decoding_steps: int,
                 action_embedding_dim: int,
                 entity_embedding_dim: int,
                 training_beam_size: int,
                 dropout_rate: float,
                 gate_dropout_rate: float = 0.2,
                 loss_mask: int = 6,
                 serialization_dir: str = 'checkpoints\\basic_model',
                 dataset_path: str = 'dataset',
                 decoder_num_layers: int = 1,
                 rule_namespace: str = 'rule_labels',
                 # parser linking and schema encoding setting
                 use_feature_score: bool = False,
                 use_schema_encoder: bool = False,
                 use_linking_embedding: bool = False,
                 # turn-level encoder setting
                 use_discourse_encoder: bool = False,
                 discourse_output_dim: int = 100,
                 use_attend_over_history: bool = False,
                 use_turn_position: bool = False,
                 # gate setting
                 use_context_gate: bool = False,
                 use_sigmoid_gate: bool = False,
                 use_unk_candidate: bool = False,
                 attn_on_self: bool = False,
                 gate_attn_size: int = 100,
                 # copy setting
                 use_sql_attention: bool = False,
                 link_to_precedent_sql: bool = False,
                 sql_hidden_size: int = 100,
                 use_copy_tree: bool = False,
                 use_copy_token: bool = False,
                 use_hard_token_as_seg: bool = False,
                 copy_encode_anon: bool = False,
                 copy_encode_with_context: bool = False,
                 # bert setting
                 bert_mode: str = "v0",
                 debug_parsing: bool = False):
        super().__init__(vocab)

        self.vocab = vocab
        self.max_decoding_steps = max_decoding_steps

        """
        Loss mask means i-th round where i is less than loss_mask will be used, otherwise will be masked.
        """
        self.loss_mask = loss_mask

        # padding for invalid action
        self.action_padding_index = -1

        # dropout inside/outside lstm
        if dropout_rate > 0:
            self.var_dropout = InputVariationalDropout(p=dropout_rate)
        else:
            self.var_dropout = lambda x: x

        self.dropout = Dropout(p=dropout_rate)

        # embedding layer of action like `Statement -> Select` and etc.
        self.rule_namespace = rule_namespace
        num_actions = vocab.get_vocab_size(self.rule_namespace)

        """
        Define encoder layer
        """
        self.text_embedder = text_embedder

        # for bert/non-bert, we use the same text encoder
        self.text_encoder = text_encoder

        self.encoder_output_dim = text_encoder.get_output_dim()
        self.embedding_dim = self.text_embedder.get_output_dim()
        self.scale = int(math.sqrt(self.embedding_dim))

        self.decoder_num_layers = decoder_num_layers

        """
        Define embedding layer
        """
        # used for scoring the action selection
        self.output_action_embedder = Embedding(num_embeddings=num_actions,
                                                embedding_dim=action_embedding_dim)
        # used for sequence generation input
        self.action_embedder = Embedding(num_embeddings=num_actions,
                                         embedding_dim=action_embedding_dim)

        # entity type embedding layer such as text/number/date/boolean/primary/foreign and etc. 0 for padding
        # TODO: entity type embedding will add in the text embedding, so it should keep the same dimension
        self.num_entity_types = 9 + 1
        self.link_entity_type_embedder = Embedding(num_embeddings=self.num_entity_types,
                                                   embedding_dim=entity_embedding_dim,
                                                   padding_index=0)

        self.output_entity_type_embedder = Embedding(num_embeddings=self.num_entity_types,
                                                     embedding_dim=action_embedding_dim,
                                                     padding_index=0)

        # Note: the dimension is highly related to the knowledge graph field.
        # please go there to see the dimensions of this linking feature.
        self.linking_layer = torch.nn.Linear(14, 1)
        torch.nn.init.uniform_(self.linking_layer.weight)
        torch.nn.init.zeros_(self.linking_layer.bias)

        self.beam_search = decoder_beam_search

        """
        Define discourse/turn level encoder
        """
        self.use_discourse_encoder = use_discourse_encoder
        self.use_attend_over_history = use_attend_over_history

        if use_discourse_encoder:
            discourse_input_dim = self.text_encoder.get_output_dim()
            self.discourse_level_encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=discourse_input_dim,
                                                                         hidden_size=discourse_output_dim,
                                                                         batch_first=True),
                                                                 stateful=True)
            if bert_mode == "v0":
                check_dimensions_match(self.embedding_dim * 2 + discourse_output_dim,
                                       text_encoder.get_input_dim(),
                                       "[Text Embedding; Linking Embedding; Discourse State]",
                                       "Text Encoder Input")
            else:
                check_dimensions_match(self.embedding_dim + discourse_output_dim,
                                       text_encoder.get_input_dim(),
                                       "[Text Embedding; Linking Embedding; Discourse State]",
                                       "Text Encoder Input")

            check_dimensions_match(discourse_input_dim,
                                   text_encoder.get_output_dim(),
                                   "Discourse Input",
                                   "Text Encoder Output")
        else:
            self.discourse_level_encoder = lambda x: x

        self.use_turn_position = use_turn_position

        # Note: turn attention means the mechanism which will concat extra positional embedding
        # with the original encoder output hidden state (whether it is encoded by discourse lstm)
        if use_turn_position:
            # turn position needs extra positional embedding (which is equal to word embedding)
            self.encoder_output_dim += self.embedding_dim

        # turn position embedding, the same dimension with embedding dim, maximum is 5

        # TODO: 7(sparc) or 50 (cosql)
        self.turn_embedder = Embedding(num_embeddings=50,
                                       embedding_dim=text_embedder.get_output_dim())

        # whether to use the context modeling
        self.use_context_gate = use_context_gate

        self.use_sigmoid_gate = use_sigmoid_gate
        self.use_unk_candidate = use_unk_candidate

        # if true, calculate attention between i and 1,2, ...i;
        # otherwise, only calculate i and 1,2,...i-1; i-th is always set as 1.0;

        self.attn_on_self = attn_on_self

        context_hidden_size = text_encoder.get_output_dim()
        if self.use_turn_position:
            context_hidden_size += self.turn_embedder.get_output_dim()

        if self.use_context_gate:

            if self.use_unk_candidate:
                self.first_unk_context = nn.Parameter(torch.FloatTensor(context_hidden_size))
                torch.nn.init.uniform_(self.first_unk_context, -0.1, 0.1)

            self.context_w = nn.Linear(context_hidden_size, gate_attn_size, bias=False)
            self.context_u = nn.Linear(context_hidden_size, gate_attn_size, bias=False)
            self.context_v = nn.Linear(gate_attn_size * 2, 1)

            torch.nn.init.uniform_(self.context_w.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.context_u.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.context_v.weight, -0.1, 0.1)

        # embedding of the first special action
        self.first_action_embedding = nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self.first_attended_output = nn.Parameter(torch.FloatTensor(self.encoder_output_dim))

        # initialize parameters
        torch.nn.init.uniform_(self.first_action_embedding, -0.1, 0.1)
        torch.nn.init.uniform_(self.first_attended_output, -0.1, 0.1)

        """
        Define sql query related network
        """
        # if anon, we will encode schema using its type; otherwise, using its schema information
        self.copy_encode_anon = copy_encode_anon

        if use_copy_token:
            # encoder_output_dim is the decoder_input_dim
            copy_gate = FeedForward(self.encoder_output_dim,
                                    num_layers=2,
                                    hidden_dims=[int(self.encoder_output_dim / 2), 1],
                                    # keep the last layer
                                    activations=[torch.tanh, lambda x: x],
                                    dropout=gate_dropout_rate)
        else:
            copy_gate = None

        if self.copy_encode_anon:
            """
            Use token type to represent its meaning
            """
            self.sql_context_encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=action_embedding_dim,
                                                                     hidden_size=sql_hidden_size,
                                                                     batch_first=True,
                                                                     bidirectional=True))
            self.sql_segment_encoder = PytorchSeq2VecWrapper(nn.LSTM(input_size=action_embedding_dim,
                                                                     hidden_size=sql_hidden_size,
                                                                     batch_first=True,
                                                                     bidirectional=True))
            self.sql_global_embedder = Embedding(num_embeddings=num_actions,
                                                 embedding_dim=action_embedding_dim)
        else:
            self.sql_context_encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=self.embedding_dim,
                                                                     hidden_size=sql_hidden_size,
                                                                     batch_first=True,
                                                                     bidirectional=True))
            self.sql_segment_encoder = PytorchSeq2VecWrapper(nn.LSTM(input_size=self.embedding_dim,
                                                                     hidden_size=sql_hidden_size,
                                                                     batch_first=True,
                                                                     bidirectional=True))
            self.sql_global_embedder = Embedding(num_embeddings=num_actions,
                                                 embedding_dim=self.embedding_dim)
        # FIXME: if tied, you should assign these two the same as the above embeddings
        #  self.output_action_embedder / self.output_entity_type_embedder
        self.sql_schema_embedder = Embedding(num_embeddings=self.num_entity_types,
                                             embedding_dim=action_embedding_dim,
                                             padding_index=0)

        self.sql_hidden_size = sql_hidden_size
        self.sql_output_size = self.sql_segment_encoder.get_output_dim()
        # add bias (equal to add bias to action embedding)
        self.copy_sql_output = nn.Linear(self.sql_output_size, action_embedding_dim)
        self.copy_sql_input = nn.Linear(self.sql_output_size, action_embedding_dim, bias=True)

        # attentional reading from precedent sql query
        self.use_sql_attention = use_sql_attention
        self.link_to_precedent_sql = link_to_precedent_sql

        # link to precedent sql means the final pointer network(linking score) will also point to precedent SQL
        if self.link_to_precedent_sql:
            assert self.use_sql_attention is True

        sql_attention = DotProductAttention()

        self.use_copy_segment = use_copy_tree
        self.use_hard_token_as_seg = use_hard_token_as_seg
        self.use_copy_token = use_copy_token

        self.use_last_sql = self.use_sql_attention or self.use_copy_token or self.use_copy_segment
        assert not (self.use_copy_token & self.use_copy_segment), "Cannot use both segment copy/token token !"

        # for segment-level copy, encode sql with context means acquiring the span representation;
        # for token-level copy, encode sql with context means using an encoder to get its hidden state;
        self.copy_encode_with_context = copy_encode_with_context
        # transform embedding into the same dimension with sql_context_encoder output
        self.sql_embedder_transform = nn.Linear(action_embedding_dim, self.sql_output_size, bias=False)

        torch.nn.init.zeros_(self.copy_sql_input.bias)
        """
        Define parsing related variants
        """
        self.use_schema_encoder = use_schema_encoder
        self.use_linking_embedding = use_linking_embedding
        self.use_feature_score = use_feature_score

        # responsible for column encoding and table encoding respectively
        if use_schema_encoder:
            self.schema_encoder = PytorchSeq2VecWrapper(nn.LSTM(input_size=self.embedding_dim,
                                                                hidden_size=int(self.embedding_dim / 2),
                                                                bidirectional=True,
                                                                batch_first=True))
        else:
            self.schema_encoder = None
        """
        Define bert mode, now we support two kinds of mode:
        "v0": IRNet
        "v3": IRNet + BERT
        """
        self.bert_mode = bert_mode

        decoder_input_dim = self.encoder_output_dim

        # extra attention concat input
        if self.use_sql_attention:
            decoder_input_dim += self.sql_output_size

        if self.use_sql_attention:
            self.transition_function = LinkingTransitionFunction(encoder_output_dim=self.encoder_output_dim,
                                                                 decoder_input_dim=decoder_input_dim,
                                                                 action_embedding_dim=action_embedding_dim,
                                                                 input_attention=input_attention,
                                                                 sql_attention=sql_attention,
                                                                 sql_output_dim=self.sql_output_size,
                                                                 predict_start_type_separately=False,
                                                                 add_action_bias=False,
                                                                 copy_gate=copy_gate,
                                                                 dropout=dropout_rate,
                                                                 num_layers=self.decoder_num_layers)
        else:
            self.transition_function = LinkingTransitionFunction(encoder_output_dim=self.encoder_output_dim,
                                                                 decoder_input_dim=decoder_input_dim,
                                                                 action_embedding_dim=action_embedding_dim,
                                                                 input_attention=input_attention,
                                                                 predict_start_type_separately=False,
                                                                 add_action_bias=False,
                                                                 copy_gate=copy_gate,
                                                                 dropout=dropout_rate,
                                                                 num_layers=self.decoder_num_layers)

        """
        Define the linear layer convert matching feature into score
        """

        """
        Define metrics to measure
        """
        self.sql_metric_util = MetricUtil(dataset_path=dataset_path)
        self.action_metric_util = MetricUtil()

        self.sql_metric = TurnAverage('sql')
        self.action_metric = TurnAverage('action')

        self.gate_metrics = {
            '_copy': Average(),
            'info': Average()
        }

        """
        Debugging setting
        """
        self.debug_parsing = debug_parsing

        if self.debug_parsing:
            try:
                from models.visualizer import Visualizer
                # get serialization_dir
                summary_dir = os.path.join("sql_log", os.path.split(serialization_dir)[-1])
                self.visualizer = Visualizer(summary_dir=summary_dir,
                                             validation_size=VALIDATE_SIZE,
                                             vocab=self.vocab)
            except ImportError:
                print("Please install tensorboardX to enable debugging in parsing.")

        self.performance_history = {}

        """
        Define transition function
        """

        self.decoder_trainer = MaximumMarginalLikelihood(training_beam_size,
                                                         re_weight=False,
                                                         loss_mask=self.loss_mask)

        self.dev_step = 0

    @overrides
    def forward(self,
                inter_utterance: Dict[str, torch.LongTensor],
                inter_segment: torch.LongTensor,
                inter_nonterminal: List,
                valid_actions_list: List[List[List[CopyProductionRule]]],
                action_sequence: torch.LongTensor,
                worlds: List[List[SparcWorld]],
                inter_schema: Dict[str, torch.LongTensor],
                entity_type: torch.FloatTensor,
                entity_mask: torch.LongTensor,
                # Action sequence with copy is built for copy segment.
                # Towards the first turn, it is equal to action_sequence
                action_sequence_with_copy: torch.LongTensor = None,
                schema_position: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        device = entity_type.device

        if 'tokens' in inter_utterance:
            assert self.bert_mode == "v0"
            # batch_size x inter_size x utter_size
            utterance_tokens = inter_utterance['tokens']
        else:
            assert self.bert_mode != "v0"
            utterance_tokens = inter_utterance['bert']

        batch_size, inter_size, _ = utterance_tokens.size()

        entity_type = entity_type * entity_mask
        # batch_size x col_size (we should expand it into inter_size then)
        entity_type = entity_type.long().view(batch_size * inter_size, -1)

        # make ids
        inter_segment = inter_segment.view(batch_size * inter_size, -1).long()

        encoder_input, encoder_mask, linked_scores, encoding_schema = self._init_parser_input(
            inter_utterance=inter_utterance,
            inter_schema=inter_schema,
            entity_type=entity_type,
            inter_segment=inter_segment,
            schema_position=schema_position)

        world_flatten = []
        for inter_world in worlds:
            for inter_ind, world in enumerate(inter_world):
                # only remove them when training & not pretrain
                world_flatten.append(world)

        valid_action_flatten = []
        for inter_valid_action in valid_actions_list:
            for valid_action in inter_valid_action:
                valid_action_flatten.append(valid_action)

        # batch_size x col_size (we should expand it into inter_size then)
        entity_type = entity_type.long().view(batch_size * inter_size, -1)

        if self.training:
            initial_state, penalty_term = self._init_grammar_state(encoder_input,
                                                                   encoder_mask,
                                                                   batch_size,
                                                                   inter_size,
                                                                   world_flatten,
                                                                   linked_scores,
                                                                   valid_action_flatten,
                                                                   entity_type,
                                                                   encoding_schema)

            if self.use_copy_segment:
                # segment-level copy will change the supervision
                action_copy_mask = torch.ne(action_sequence_with_copy, self.action_padding_index)
                decode_output = self.decoder_trainer.decode(initial_state,
                                                            self.transition_function,
                                                            (action_sequence_with_copy,
                                                             action_copy_mask))
            else:
                action_mask = torch.ne(action_sequence, self.action_padding_index)
                decode_output = self.decoder_trainer.decode(initial_state,
                                                            self.transition_function,
                                                            (action_sequence,
                                                             action_mask))
            return {'loss': decode_output['loss'] + 0 * penalty_term}
        else:
            assert batch_size == 1, "Now we only support batch_size = 1 on evaluation"
            self.dev_step += batch_size
            loss = torch.tensor([0]).float().to(device)

            initial_state, _ = self._init_grammar_state(encoder_input,
                                                        encoder_mask,
                                                        batch_size,
                                                        inter_size,
                                                        world_flatten,
                                                        linked_scores,
                                                        valid_action_flatten,
                                                        entity_type,
                                                        encoding_schema)

            if action_sequence is not None and action_sequence.size(1) > 1:
                try:
                    with torch.no_grad():
                        if self.use_copy_segment:
                            action_copy_mask = torch.ne(action_sequence_with_copy, self.action_padding_index)
                            loss = self.decoder_trainer.decode(initial_state,
                                                               self.transition_function,
                                                               (action_sequence_with_copy,
                                                                action_copy_mask))['loss']
                        else:
                            action_mask = torch.ne(action_sequence, self.action_padding_index)
                            loss = self.decoder_trainer.decode(initial_state,
                                                               self.transition_function,
                                                               (action_sequence,
                                                                action_mask))['loss']
                except ZeroDivisionError:
                    # reached a dead-end during beam search
                    pass

            outputs: Dict[str, Any] = {
                'loss': loss
            }

            # In evaluation, segment-level copy will lead to two concerns:
            # 1. the evaluation of turn $t$ can only be done after the turn $t-1$, so we need dynamically update
            #    precedent action sequence stored in the world.
            # 2. the generating results should be reformulated as non-copy existing (e.g. expand the copy action).
            num_steps = self.max_decoding_steps
            # construct db_contexts
            db_contexts = [world.db_context if world is not None else None
                           for world in world_flatten]

            # Get the rules out of the instance
            index_to_rule = [production_rule_field[0]
                             for production_rule_field in
                             valid_action_flatten[0]]

            if self.use_last_sql:
                assert batch_size == 1
                outputs['best_predict_idx'] = []
                outputs['best_predict_action_copy'] = []
                outputs['debug_info'] = []

                # CAUTION: we will change the world & valid_action_flatten, so we need deepcopy
                world_flatten = deepcopy(world_flatten)
                valid_action_flatten = deepcopy(valid_action_flatten)

                # clear all copy-based actions in valid_action_flatten(if any)
                for i in range(inter_size):
                    # no padding
                    copy_action_ids = []
                    for j in reversed(range(len(valid_action_flatten[i]))):
                        action = valid_action_flatten[i][j]
                        if action.rule == '':
                            del valid_action_flatten[i][j]
                        elif action.is_copy_rule:
                            copy_action_ids.append(j)
                            del valid_action_flatten[i][j]

                    valid_action_flatten[i] = [action for action in valid_action_flatten[i]
                                               if not action.is_copy_rule and action.rule]
                    world_flatten[i].clear_precedent_state(copy_action_ids)
                # to easily handle it, we assume batch_size = 1
                for i in range(inter_size):
                    # WARNING: if we use both discourse encoder & segment copy, the discourse encoder
                    #  assumes the input & mask are batchwise-interaction, however, here we pass by
                    #  the only turn itself. Therefore, for discourse encoder scenario, we should
                    #  pass by the encoder[:i + 1]
                    if self.use_discourse_encoder:
                        initial_state, _ = self._init_grammar_state(encoder_input[: i + 1],
                                                                    encoder_mask[: i + 1],
                                                                    batch_size,
                                                                    # inter_size is i + 1
                                                                    i + 1,
                                                                    world_flatten[: i + 1],
                                                                    linked_scores[: i + 1],
                                                                    valid_action_flatten[: i + 1],
                                                                    entity_type[: i + 1],
                                                                    encoding_schema[: i + 1])
                        initial_state.debug_info = [[] for _ in range(i + 1)]
                        temp_predict_results = self.beam_search.search(num_steps,
                                                                       initial_state,
                                                                       self.transition_function,
                                                                       keep_final_unfinished_states=True)
                    else:
                        # refresh initial_state. unsqueeze to fake the batch_size dimension
                        initial_state, _ = self._init_grammar_state(encoder_input[i].unsqueeze(dim=0),
                                                                    encoder_mask[i].unsqueeze(dim=0),
                                                                    batch_size,
                                                                    # inter_size is set as 1
                                                                    1,
                                                                    [world_flatten[i]],
                                                                    linked_scores[i].unsqueeze(dim=0),
                                                                    [valid_action_flatten[i]],
                                                                    entity_type[i].unsqueeze(dim=0),
                                                                    encoding_schema[i].unsqueeze(dim=0))
                        initial_state.debug_info = [[]]
                        temp_predict_results = self.beam_search.search(num_steps,
                                                                       initial_state,
                                                                       self.transition_function,
                                                                       keep_final_unfinished_states=True)

                    # if use discourse, take the last one; else, take the default prediction 0
                    take_away_key = i if self.use_discourse_encoder else 0
                    # keep the same structure
                    if take_away_key in temp_predict_results and i != 0:
                        best_action_sequence_with_copy = temp_predict_results[take_away_key][0].action_history[0]
                        best_action_sequence = []
                        for action_id in best_action_sequence_with_copy:
                            if valid_action_flatten[i][action_id].is_copy_rule:
                                # extend copy's related action ids
                                copy_action: Segment = world_flatten[i].valid_actions_flat[action_id]
                                valid_ins_idx = [idx for idx in copy_action.copy_ins_idx
                                                 if idx != self.action_padding_index]
                                best_action_sequence.extend(valid_ins_idx)
                            else:
                                best_action_sequence.append(action_id)
                        # Get the rules out of the instance
                        index_to_rule_copy = [production_rule_field[0]
                                              for production_rule_field in
                                              valid_action_flatten[i]]
                        outputs['best_predict_action_copy'].append(",".join([str(index_to_rule_copy[action_idx])
                                                                             for action_idx in
                                                                             best_action_sequence_with_copy]))
                    elif i == 0:
                        # no copy action
                        best_action_sequence = temp_predict_results[0][0].action_history[0]
                        outputs['best_predict_action_copy'].append(",".join([str(index_to_rule[action_idx])
                                                                             for action_idx in
                                                                             best_action_sequence]))
                    else:
                        best_action_sequence = []
                        outputs['best_predict_action_copy'].append("[EMPTY]")

                    outputs['best_predict_idx'].append(best_action_sequence)
                    outputs['debug_info'].append(temp_predict_results[0][0].debug_info[0])

                    if i != inter_size - 1:
                        # update next world's precedent action sequence.
                        # note the update is both for COPY & TOKEN.
                        # for token-level copy, it will ignore the segment actions.
                        world_flatten[i + 1].update_precedent_state(best_action_sequence,
                                                                    extract_tree=not self.use_hard_token_as_seg)
                        world_flatten[i + 1].update_copy_valid_action()
                        # manually construct CopyRule for valid_action_flatten
                        for local_ind, prod_rule in enumerate(world_flatten[i + 1].precedent_segment_seq):
                            # get nonterminal name
                            nonterminal = prod_rule.nonterminal
                            rule_repr = str(prod_rule)
                            copy_rule = CopyProductionRule(rule=rule_repr,
                                                           # the copy rule is appended dynamically
                                                           is_global_rule=False,
                                                           is_copy_rule=True,
                                                           nonterminal=nonterminal)
                            valid_action_flatten[i + 1].append(copy_rule)
                        assert len(valid_action_flatten[i + 1]) == len(world_flatten[i + 1].valid_actions_flat)
                # to fake the batch_size scope
                outputs['best_predict_action_copy'] = [outputs['best_predict_action_copy']]
                outputs['debug_info'] = [outputs['debug_info']]
            else:
                # This tells the state to start keeping track of debug info, which we'll pass along in
                # our output dictionary.
                initial_state.debug_info = [[] for _ in range(batch_size * inter_size)]

                best_final_states = self.beam_search.search(num_steps,
                                                            initial_state,
                                                            self.transition_function,
                                                            keep_final_unfinished_states=True)

                outputs['best_predict_idx'] = [
                    best_final_states[i][0].action_history[0] if i in best_final_states else [0]
                    for i in range(batch_size * inter_size)]
                outputs['debug_info'] = [[best_final_states[i][0].debug_info[0] if i in best_final_states else []
                                          for i in range(batch_size * inter_size)]]

            # in test mode, predict the actual SQL
            if worlds[0][0].sql_query == '':
                outputs['best_predict_action'] = [[",".join([str(index_to_rule[action_idx])
                                                             for action_idx in action_seq])
                                                   for action_seq in outputs['best_predict_idx']]]

                predict_sql = self.predict_sql(iter_size=batch_size * inter_size,
                                               index_to_rule=index_to_rule,
                                               predict_result=outputs['best_predict_idx'],
                                               db_contexts=db_contexts)
                outputs['best_predict_sql'] = predict_sql

                # add utterance for better reading
                if self.bert_mode == "v0":
                    # if under the v3 bert mode, the utterance idx cannot be recovered as
                    # the `BertIndexer` will re-index these tokens.
                    utterance_strs = [[self.vocab.get_token_from_index(int(token_id)) if token_id != 0
                                       else ''
                                       for token_id in token_seq]
                                      for token_seq in utterance_tokens.view(batch_size * inter_size, -1)]
                    outputs['utterance'] = [utterance_strs]
                for debug_sample in outputs['debug_info'][0]:
                    # every sample is a list
                    for info_dict in debug_sample:
                        info_dict['question_attention'] = ["{0:.2f}".format(float(num))
                                                           for num in info_dict['question_attention']]
                        info_dict['probabilities'] = ["{0:.2f}".format(float(num))
                                                      for num in info_dict['probabilities']]
            else:
                # reshape sequence and mask for convenient
                action_sequence = action_sequence.reshape(batch_size * inter_size, -1)
                action_mask = torch.ne(action_sequence, self.action_padding_index)

                action_correct_mat, action_mask_mat = self.action_metric_util(outputs['best_predict_idx'],
                                                                              action_sequence,
                                                                              batch_size,
                                                                              action_mask)
                self.action_metric(action_correct_mat, action_mask_mat)

                # construct action mapping
                action_mapping: List[List[str]] = [[production_rule[0] for production_rule in valid_action]
                                                   for batch_valid_actions in valid_actions_list
                                                   for valid_action in batch_valid_actions]
                sql_ground_truth: List[str] = [world.sql_query
                                               for batch_world in worlds
                                               for world in batch_world]
                # calculate SQL matching
                sql_correct_mat, sql_mask_mat = self.sql_metric_util(outputs['best_predict_idx'],
                                                                     sql_ground_truth,
                                                                     batch_size,
                                                                     action_mask,
                                                                     db_contexts,
                                                                     action_mapping,
                                                                     with_sql=True)
                self.sql_metric(sql_correct_mat, sql_mask_mat)

                if self.debug_parsing:
                    self.visualizer.update_global_step()
                    self.visualizer.log_sql(inter_utterance,
                                            sql_correct_mat[0],
                                            sql_ground_truth,
                                            encoder_mask,
                                            [[index_to_rule[ind] for ind in inter]
                                             for inter in outputs['best_predict_idx']])

            return outputs

    @staticmethod
    def predict_sql(iter_size, index_to_rule, predict_result, db_contexts) -> List[str]:
        predict_sql = []
        for i in range(iter_size):
            action_seq = [index_to_rule[ind] for ind in predict_result[i]]
            converter = ActionConverter(db_context=db_contexts[i])
            try:
                generated_sql = converter.translate_to_sql(action_seq)
            except:
                # if fail, return one valid SQL
                generated_sql = f'SELECT * from {list(db_contexts[i].schema.keys())[0]}'
                exec_info = sys.exc_info()
                traceback.print_exception(*exec_info)
            predict_sql.append(generated_sql)
        # fake the batch_size dimension
        return [predict_sql]

    def _init_parser_input(self, inter_utterance: Dict[str, torch.LongTensor],
                           inter_schema: Dict[str, torch.LongTensor],
                           entity_type: torch.LongTensor,
                           inter_segment: torch.LongTensor = None,
                           schema_position: torch.LongTensor = None):
        device = entity_type.device
        # {'text': {'token': tensor}, 'linking': tensor }
        # batch_size x inter_size x col_size x col_token_size

        if 'tokens' in inter_utterance:
            assert self.bert_mode == "v0"
            utterance_tokens = inter_utterance['tokens']
        else:
            assert self.bert_mode != "v0"
            utterance_tokens = inter_utterance['bert']

        batch_size, inter_size, _ = utterance_tokens.size()

        if self.bert_mode == "v0":
            schema_text = inter_schema['text']
            batch_size, inter_size, col_size, col_token_size = schema_text['tokens'].size()
            # batch_size * inter_size x col_size x col_token_size (e.g. hospital station is 2)
            token_dict = {
                'tokens': schema_text['tokens'].view(batch_size * inter_size, col_size, col_token_size)
            }
            inter_utterance['tokens'] = inter_utterance['tokens'].view(batch_size * inter_size, -1)
            # batch_size * inter_size x col_size x col_token_size x embedding_size
            embedded_schema = self.text_embedder.forward(token_dict, num_wrapping_dims=1)
            # get input mask
            encoder_mask = util.get_text_field_mask(inter_utterance).float()
            embedded_utterance = self.text_embedder.forward(inter_utterance)

            # Compute entity and question word similarity.  We tried using cosine distance here, but
            # because this similarity is the main mechanism that the model can use to push apart logit
            # scores for certain actions (like "n -> 1" and "n -> -1"), this needs to have a larger
            # output range than [-1, 1].
            if self.use_schema_encoder:
                # resize schema and others
                encoder_schema_mask = (token_dict['tokens'] != 0).long()
                embedded_schema = embedded_schema.view(batch_size * inter_size * col_size, col_token_size, -1)
                encoder_schema_mask = encoder_schema_mask.view(batch_size * inter_size * col_size, col_token_size)

                # get the results, note the result is actually the final result of every column
                encoding_schema = self.schema_encoder.forward(embedded_schema, encoder_schema_mask)
                encoding_schema = encoding_schema.view(batch_size * inter_size, col_size, -1)

                # encode table & column
                linking_scores = torch.bmm(encoding_schema,
                                           torch.transpose(embedded_utterance, 1, 2))
            else:
                encoding_schema = embedded_schema.view(batch_size * inter_size,
                                                       col_size * col_token_size,
                                                       self.embedding_dim)
                question_entity_similarity = torch.bmm(encoding_schema,
                                                       torch.transpose(embedded_utterance, 1, 2)) / self.scale

                # BOW representation
                # eps for nan loss
                encoder_sum = (token_dict['tokens'] != 0).view(batch_size * inter_size * col_size,
                                                               col_token_size).sum(dim=1).float() + 1e-2
                encoding_schema = encoding_schema.view(batch_size * inter_size * col_size,
                                                       col_token_size, self.embedding_dim).sum(dim=1)
                encoding_schema = encoding_schema / encoder_sum.unsqueeze(dim=1).expand_as(encoding_schema)
                encoding_schema = encoding_schema.view(batch_size * inter_size, col_size, self.embedding_dim)

                # batch_size * inter_size x col_size x col_token_size x utt_token_size
                question_entity_similarity = question_entity_similarity.view(batch_size * inter_size,
                                                                             col_size,
                                                                             col_token_size,
                                                                             -1)
                # batch_size * inter_size x col_size x utt_token_size
                question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)
                linking_scores = question_entity_similarity_max_score

            # calculate linking scores and probabilities
            if self.use_feature_score:
                linking_features = inter_schema['linking']
                feature_size = linking_features.size(-1)
                linking_features = linking_features.view(batch_size * inter_size, col_size, -1, feature_size)
                # batch_size * inter_size x col_size x utt_token_siz`e
                feature_scores = self.linking_layer(linking_features).squeeze(3)
                # batch_size * inter_size x col_size x utt_token_size
                linking_scores = linking_scores + feature_scores

            entity_size = self.num_entity_types

            # concat word embedding and type embedding
            # batch_size * inter_size x utt_token_size x (link_embedded_size + utt_embedded_size)
            # encoder_input = linking_embedding + embedded_with_segment
            # encoder_input = embedded_with_segment
            if self.use_turn_position and not self.use_discourse_encoder:
                embedded_segment = self.turn_embedder.forward(inter_segment)
                embedded_utterance = embedded_utterance + embedded_segment

            if self.use_linking_embedding:
                # batch_size * inter_size x col_size x entity_size (10 now)
                entity_type_mat = torch.zeros((batch_size * inter_size, col_size, entity_size), dtype=torch.float32,
                                              device=device)

                # create one hot vector
                expand_entity_type = entity_type.unsqueeze(dim=2)
                entity_type_mat.scatter_(dim=2, index=expand_entity_type, value=1)

                # add 1e-8 as epsilon
                entity_type_mat = entity_type_mat + 1e-8

                # batch_size * inter_size x utt_token_size x entity_size
                linking_probabilities = self._get_linking_probabilities(linking_scores.transpose(1, 2),
                                                                        entity_type_mat)

                linking_probabilities = encoder_mask.unsqueeze(dim=-1).repeat(1, 1, entity_size) * linking_probabilities

                # batch_size * inter_size x entity_size x entity_embedding_size
                entity_ids = torch.arange(0, entity_size, 1, dtype=torch.long, device=device).unsqueeze(dim=0). \
                    repeat(batch_size * inter_size, 1)
                entity_type_embeddings = self.link_entity_type_embedder.forward(entity_ids)

                # non linear layer for embedding
                # TODO: why tanh ?
                entity_type_embeddings = torch.tanh(entity_type_embeddings)

                # calculate the weighted entity embeddings
                # batch_size * inter_size x utt_token_size x entity_embedding_size
                linking_embedding = torch.bmm(linking_probabilities, entity_type_embeddings)
                parser_input = torch.cat([embedded_utterance, linking_embedding], dim=-1)
            else:
                parser_input = embedded_utterance
        elif self.bert_mode == "v3":
            assert inter_segment is not None
            assert schema_position is not None

            batch_size, inter_size, col_size, _ = schema_position.size()
            schema_position = schema_position.long()
            # batch_size * inter_size x col_size x 2
            schema_position = schema_position.view(batch_size * inter_size, col_size, -1)

            max_col_token_size = (schema_position[:, :, 1] - schema_position[:, :, 0]).max()
            # we do not use any schema
            utter_end_indices = inter_segment.ne(0).sum(dim=1)

            for key, value in inter_utterance.items():
                inter_utterance[key] = inter_utterance[key].view(batch_size * inter_size, -1)
                if 'type-ids' in key:
                    for i in range(batch_size * inter_size):
                        inter_utterance[key][i, :utter_end_indices[i]] = 0
                        inter_utterance[key][i, utter_end_indices[i]:] = 1

            embedded_mix = self.text_embedder.forward(inter_utterance)
            mask_mix = inter_utterance['mask']
            embedded_mix = embedded_mix * mask_mix.unsqueeze(dim=2).float()

            # split embedded mix into two parts: utterance & schema
            embedded_utterance = []
            encoder_mask = []
            embedded_schema = []
            encoder_schema_mask = []

            for ind, end_ind in enumerate(utter_end_indices):
                embedded_utterance.append(embedded_mix[ind, :end_ind, :])
                encoder_mask.append(mask_mix[ind, :end_ind])

                cur_embedded_schema = []
                cur_schema_mask = []
                for col_ind in range(col_size):
                    entity_start_ind = schema_position[ind, col_ind, 0]
                    entity_end_ind = schema_position[ind, col_ind, 1]
                    pad_len = max_col_token_size - (entity_end_ind - entity_start_ind)
                    # padding for concat
                    cur_embedded_schema.append(F.pad(embedded_mix[ind, entity_start_ind: entity_end_ind, :],
                                                     pad=[0, 0, 0, pad_len],
                                                     mode='constant'))
                    cur_schema_mask.append(F.pad(mask_mix[ind, entity_start_ind: entity_end_ind],
                                                 pad=[0, pad_len]))
                cur_embedded_schema = torch.stack(cur_embedded_schema, dim=0)
                embedded_schema.append(cur_embedded_schema)
                cur_schema_mask = torch.stack(cur_schema_mask, dim=0)
                encoder_schema_mask.append(cur_schema_mask)

            embedded_utterance = pad_sequence(embedded_utterance, batch_first=True)
            embedded_schema = pad_sequence(embedded_schema, batch_first=True)
            # according to length of segment to identify which one is utterance/schema
            encoder_mask = pad_sequence(encoder_mask, batch_first=True)
            encoder_schema_mask = pad_sequence(encoder_schema_mask, batch_first=True)

            if self.use_schema_encoder:
                # resize schema and others
                embedded_schema = embedded_schema.view(batch_size * inter_size * col_size, max_col_token_size, -1)
                encoder_schema_mask = encoder_schema_mask.view(batch_size * inter_size * col_size, max_col_token_size)

                # get the results, note the result is actually the final result of every column
                encoding_schema = self.schema_encoder.forward(embedded_schema, encoder_schema_mask)
                encoding_schema = encoding_schema.view(batch_size * inter_size, col_size, -1)

                # encode table & column
                linking_scores = torch.bmm(encoding_schema,
                                           torch.transpose(embedded_utterance, 1, 2)) / self.scale
            else:
                # encode table & column
                encoding_schema = embedded_schema.view(batch_size * inter_size,
                                                       col_size * max_col_token_size,
                                                       self.embedding_dim)
                question_entity_similarity = torch.bmm(encoding_schema,
                                                       torch.transpose(embedded_utterance, 1, 2)) / self.scale

                # eps for nan loss
                encoder_sum = encoder_schema_mask.view(batch_size * inter_size * col_size,
                                                       max_col_token_size).sum(dim=1).float() + 1e-2
                encoding_schema = encoding_schema.view(batch_size * inter_size * col_size,
                                                       max_col_token_size, self.embedding_dim).sum(dim=1)
                encoding_schema = encoding_schema / encoder_sum.unsqueeze(dim=1).expand_as(encoding_schema)
                encoding_schema = encoding_schema.view(batch_size * inter_size, col_size, self.embedding_dim)

                # batch_size * inter_size x col_size x col_token_size x utt_token_size
                question_entity_similarity = question_entity_similarity.view(batch_size * inter_size,
                                                                             col_size,
                                                                             max_col_token_size,
                                                                             -1)
                # batch_size * inter_size x col_size x utt_token_size
                question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)
                linking_scores = question_entity_similarity_max_score

            if self.use_feature_score:
                linking_features = inter_schema['linking']
                feature_size = linking_features.size(-1)
                linking_features = linking_features.view(batch_size * inter_size, col_size, -1, feature_size)
                # batch_size * inter_size x col_size x utt_token_siz`e
                feature_scores = self.linking_layer.forward(linking_features).squeeze(3)
                linking_scores = linking_scores + feature_scores

            parser_input = embedded_utterance

            # calculate linking scores with utterance
            if self.use_turn_position and not self.use_discourse_encoder:
                embedded_segment = self.turn_embedder.forward(inter_segment)
                parser_input = parser_input + embedded_segment

            if self.use_linking_embedding:
                entity_size = self.num_entity_types

                # batch_size * inter_size x col_size x entity_size (10 now)
                entity_type_mat = torch.zeros((batch_size * inter_size, col_size, entity_size), dtype=torch.float32,
                                              device=device)

                # create one hot vector
                expand_entity_type = entity_type.unsqueeze(dim=2)
                entity_type_mat.scatter_(dim=2, index=expand_entity_type, value=1)

                # add 1e-8 as epsilon
                entity_type_mat = entity_type_mat + 1e-8

                # batch_size * inter_size x utt_token_size x entity_size
                linking_probabilities = self._get_linking_probabilities(linking_scores.transpose(1, 2),
                                                                        entity_type_mat)

                linking_probabilities = encoder_mask.unsqueeze(dim=-1).repeat(1, 1,
                                                                              entity_size).float() * linking_probabilities

                # batch_size * inter_size x entity_size x entity_embedding_size
                entity_ids = torch.arange(0, entity_size, 1, dtype=torch.long, device=device).unsqueeze(dim=0). \
                    repeat(batch_size * inter_size, 1)
                entity_type_embeddings = self.link_entity_type_embedder.forward(entity_ids)
                entity_type_embeddings = torch.tanh(entity_type_embeddings)

                # calculate the weighted entity embeddings
                # batch_size * inter_size x utt_token_size x entity_embedding_size
                linking_embedding = torch.bmm(linking_probabilities, entity_type_embeddings)
                parser_input = parser_input + linking_embedding
        else:
            raise Exception("DO NOT SUPPORT BERT MODE :{}".format(self.bert_mode))

        return parser_input, encoder_mask, linking_scores, encoding_schema

    def _init_grammar_state(self,
                            encoder_input: torch.FloatTensor,
                            encoder_mask: torch.LongTensor,
                            batch_size: int,
                            inter_size: int,
                            world_flatten: List[SparcWorld],
                            linking_scores: torch.FloatTensor,
                            valid_action_flatten: List[List[CopyProductionRule]],
                            entity_type: torch.LongTensor,
                            # encoding_schema, batch_size * inter_size x schema_size
                            encoding_schema: torch.FloatTensor):

        encoder_mask = encoder_mask.float()
        iter_size = batch_size * inter_size

        # specific devices
        device = encoder_mask.device
        penalty = torch.zeros(1, device=device, requires_grad=True)

        padding_utterance_mask = encoder_mask.clone()
        padding_utterance_mask.data[:, 0].fill_(1)

        # encode and output encoder memory
        encoder_input = self.var_dropout(encoder_input)

        # an unified process to handle bert/non-bert embedding as input
        _, sequence_len, embedding_dim = encoder_input.size()

        # record state of each utterance
        encoder_vector_states = []

        if self.use_discourse_encoder:
            # iterative encoding on inputs
            encoder_input = encoder_input.view(batch_size, inter_size, sequence_len, embedding_dim)
            padding_utterance_mask = padding_utterance_mask.view(batch_size, inter_size, -1)

            discourse_inp_dim = self.discourse_level_encoder.get_input_dim()
            discourse_state_dim = self.discourse_level_encoder.get_output_dim()

            discourse_input = torch.zeros((batch_size, 1, discourse_inp_dim), device=encoder_input.device)
            discourse_state = torch.zeros((batch_size, 1, discourse_state_dim), device=encoder_input.device)

            turn_level_mask = (encoder_mask.sum(dim=1) != 0).view(batch_size, inter_size)
            # get encoder outputs
            encoder_outputs_states = []

            for i in range(inter_size):
                if i != 0:
                    # discourse_state is the last hidden state to cached
                    discourse_state = self.discourse_level_encoder.forward(discourse_input,
                                                                           turn_level_mask[:, i].unsqueeze(dim=1))
                text_input = encoder_input[:, i, :]
                text_input = torch.cat([text_input,
                                        discourse_state.repeat(1, sequence_len, 1)
                                        ], dim=2)
                encoder_outputs = self.text_encoder.forward(text_input,
                                                            padding_utterance_mask[:, i])
                discourse_input = util.get_final_encoder_states(encoder_outputs=encoder_outputs,
                                                                mask=padding_utterance_mask[:, i],
                                                                bidirectional=self.text_encoder.is_bidirectional()).unsqueeze(
                    dim=1)
                encoder_vector_states.append(discourse_input)
                # dimension 1 is the interaction dimension
                encoder_outputs_states.append(encoder_outputs.unsqueeze(dim=1))
            # recover outputs and padding mask
            utt_encoder_outputs = torch.cat(encoder_outputs_states, dim=1).view(batch_size * inter_size,
                                                                                sequence_len, -1)
            padding_utterance_mask = padding_utterance_mask.view(batch_size * inter_size, sequence_len)
        else:
            utt_encoder_outputs = self.text_encoder.forward(encoder_input, padding_utterance_mask)

        utt_encoder_outputs = self.var_dropout(utt_encoder_outputs)
        # This will be our initial hidden state and memory cell for the decoder LSTM.

        # if use discourse encoder, it means the context is independent, so we should concat encoder outputs to
        # compute `turn attention`
        if self.use_turn_position:
            # concat all outputs, meanwhile we should change the interaction mask
            # 1 x turn_size
            turn_ids = torch.arange(start=inter_size + 1, end=1, step=-1, device=device) \
                .unsqueeze(dim=0).repeat(batch_size, 1).view(batch_size * inter_size)
            loss_mask_ids = torch.zeros(turn_ids.size(), device=device).fill_(self.loss_mask).long()
            turn_ids = torch.where(turn_ids.float() > self.loss_mask, loss_mask_ids, turn_ids)
            turn_embedding = self.turn_embedder(turn_ids)
            # batch_size x inter_size x embedding_dim x sequence_len
            turn_encoder_embedding = turn_embedding.unsqueeze(dim=1).repeat(1, sequence_len, 1)
            utt_encoder_outputs = torch.cat([utt_encoder_outputs, turn_encoder_embedding], dim=2)
            # use a transform layer to normalize the shape
            # dropout on encoder_vector_states
            encoder_vector_states = torch.cat(encoder_vector_states, dim=1)

            # use dropout to avoid overfitting
            encoder_vector_states = self.dropout(encoder_vector_states)
            encoder_vector_states = torch.split(encoder_vector_states, split_size_or_sections=1, dim=1)
            encoder_vector_states = [torch.cat((encoder_vector_states[i], turn_embedding[i].
                                                unsqueeze(dim=0).unsqueeze(dim=0).repeat(batch_size, 1, 1)),
                                               dim=-1) for i in range(inter_size)]

        # TODO: in the original mask, it will cause into fetching nothing because there may be an empty sentence.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs=utt_encoder_outputs,
                                                             mask=padding_utterance_mask,
                                                             bidirectional=self.text_encoder.is_bidirectional())

        if self.use_attend_over_history:
            utt_encoder_outputs = utt_encoder_outputs.view(batch_size, inter_size, sequence_len, -1)
            encoder_mask = encoder_mask.view(batch_size, inter_size, sequence_len)

            _, col_size, _ = linking_scores.size()
            linking_scores = linking_scores.view(batch_size, inter_size, col_size, sequence_len)
            # notice here you should concat all encoder output and all linking scores
            his_encoder_outputs = []
            his_encoder_mask = []
            his_linking_score = []

            if self.use_context_gate:
                # info_gates[batch_ind, i, j] means keep how much information when encoding context_j in turn i
                info_gates = torch.eye(n=inter_size, m=inter_size, device=device, dtype=torch.float). \
                    unsqueeze(dim=0).repeat(batch_size, 1, 1)

                if self.use_discourse_encoder:
                    reattn_states = encoder_vector_states
                else:
                    utt_states = util.get_final_encoder_states(
                        encoder_outputs=utt_encoder_outputs.view(batch_size * inter_size, sequence_len, -1),
                        mask=padding_utterance_mask,
                        bidirectional=self.text_encoder.is_bidirectional())
                    reattn_states = utt_states.view(batch_size, inter_size, 1, -1).transpose(0, 1)
                    reattn_states = list(reattn_states)

                for i in range(1, inter_size):
                    # unk candidate is designed for softmax attention
                    if self.use_sigmoid_gate or not self.use_unk_candidate:

                        if self.attn_on_self:
                            cur_value_vector = torch.cat(reattn_states[:i + 1], dim=1)
                        else:
                            cur_value_vector = torch.cat(reattn_states[:i], dim=1)
                    else:
                        if self.attn_on_self:
                            cur_value_vector = torch.cat(reattn_states[:i + 1], dim=1)
                        else:
                            cur_value_vector = torch.cat(reattn_states[:i], dim=1)
                        # add a [UNK] candidate
                        candidate_vector = self.dropout(self.first_unk_context).unsqueeze(dim=0). \
                            unsqueeze(dim=0).repeat(batch_size, 1, 1)
                        cur_value_vector = torch.cat([candidate_vector,
                                                      cur_value_vector], dim=1)
                    # batch_size x sequence_len x hidden_size
                    cur_query_vector = reattn_states[i].repeat(1, cur_value_vector.size()[1], 1)
                    hidden_vector = torch.cat([self.context_u(cur_value_vector),
                                               self.context_w(cur_query_vector)], dim=-1)
                    if self.use_sigmoid_gate:
                        info_gate = torch.sigmoid(self.context_v(hidden_vector)).squeeze(dim=-1)
                        penalty = penalty + info_gate.mean().norm()
                    else:
                        # softmax gate
                        info_gate = torch.softmax(self.context_v(torch.tanh(hidden_vector)), dim=1).squeeze(dim=-1)
                        if self.use_unk_candidate:
                            # drop the unk candidate
                            info_gate = info_gate[:, 1:]
                    # record info_gate into logs
                    self.gate_metrics['info'](float(info_gate.mean()))

                    if self.attn_on_self:
                        info_gates[:, i, :i + 1] = info_gate
                    else:
                        info_gates[:, i, :i] = info_gate
                penalty = penalty / inter_size

            for i in range(batch_size):
                cur_all_encoder_output = []
                cur_all_encoder_mask = []
                cur_all_linking_score = []
                for turn_ind in range(inter_size):
                    cur_all_encoder_output.append(utt_encoder_outputs[i, turn_ind])
                    if self.use_context_gate:
                        gates = info_gates[i, turn_ind, : turn_ind + 1]
                        his_encoder_output = [cur_all_encoder_output[i] * gates[i]
                                              for i in range(turn_ind + 1)]
                        his_encoder_output = torch.cat(his_encoder_output, dim=0)
                    else:
                        his_encoder_output = torch.cat(cur_all_encoder_output, dim=0)
                    his_encoder_outputs.append(his_encoder_output)

                    cur_all_encoder_mask.append(encoder_mask[i, turn_ind])
                    his_encoder_mask.append(torch.cat(cur_all_encoder_mask, dim=0))

                    cur_all_linking_score.append(linking_scores[i, turn_ind])
                    his_linking_output = torch.cat(cur_all_linking_score, dim=1).transpose(0, 1)
                    # before padding, transpose col_size x utt_size -> utt_size x col_size
                    his_linking_score.append(his_linking_output)

            utt_encoder_outputs = pad_sequence(his_encoder_outputs, batch_first=True)
            encoder_mask = pad_sequence(his_encoder_mask, batch_first=True)
            linking_scores = pad_sequence(his_linking_score, batch_first=True).transpose(1, 2)

        memory_cell = utt_encoder_outputs.new_zeros(iter_size, self.encoder_output_dim)

        # prepared for sql attention
        if self.use_sql_attention:
            # add extra parameters
            sql_memory_cell = utt_encoder_outputs.new_zeros(iter_size, self.sql_output_size)
        else:
            sql_memory_cell = [None] * iter_size

        initial_score = torch.zeros(iter_size, device=device, dtype=torch.float32)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(iter_size)]
        encoder_output_list = [utt_encoder_outputs[i] for i in range(iter_size)]
        utterance_mask_list = [encoder_mask[i] for i in range(iter_size)]

        # TODO: reorganize the world and valid action list
        # FIXME: Hack for computing efficiency. Here we mask the world which cannot arrive the loss_mask

        db_context_flatten = [world.db_context if world is not None else None
                              for world in world_flatten]

        fetch_sql_inform = [{} for i in range(iter_size)]
        initial_grammar_state = [self._create_grammar_state(world_flatten[i],
                                                            valid_action_flatten[i],
                                                            linking_scores[i],
                                                            entity_type[i],
                                                            encoding_schema[i],
                                                            fetch_sql_inform[i])
                                 for i in range(iter_size)]

        if self.use_sql_attention:
            sql_output_list = [ins['sql_output'] for ins in fetch_sql_inform]
            sql_output_mask_list = [ins['sql_output_mask'] for ins in fetch_sql_inform]

            sql_output_list = list(pad_sequence(sql_output_list, batch_first=True))
            sql_output_mask_list = list(pad_sequence(sql_output_mask_list, batch_first=True))
        else:
            sql_output_list = [None] * iter_size
            sql_output_mask_list = [None] * iter_size

        initial_rnn_state = []
        for i in range(iter_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self.first_action_embedding,
                                                 self.first_attended_output,
                                                 encoder_output_list,
                                                 utterance_mask_list,
                                                 sql_memory_cell[i],
                                                 sql_output_list,
                                                 sql_output_mask_list))

        # initialize constrain state
        initial_condition_state = [ConditionStatelet(valid_action_flatten[i],
                                                     db_context_flatten[i],
                                                     # if self in training, the prune should not be pruned
                                                     enable_prune=not self.training)
                                   for i in range(iter_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(iter_size)),
                                          action_history=[[] for _ in range(iter_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          condition_state=initial_condition_state,
                                          possible_actions=valid_action_flatten)
        return initial_state, penalty

    def _create_grammar_state(self,
                              world: SparcWorld,
                              possible_actions: List[CopyProductionRule],
                              linking_scores: torch.Tensor,
                              entity_types: torch.LongTensor,
                              # col_size x col_embedding_size
                              encoded_schema: torch.FloatTensor,
                              # construct take away array
                              take_away: Dict = None) -> 'GrammarStatelet':
        """
        Construct initial grammar state let for decoding constraints
        :param world: ``SparcWorld``
        :param possible_actions: ``List[CopyProductionRule]``, tracking the all possible actions under current state
        this rule is different from the one in allennlp as it is support `is_copy` attribute
        :param linking_scores: ``torch.Tensor``, the linking score between every query token and each entity type
        :param entity_types: ``torch.Tensor``, the entity type of each schema in database
        :return:
        """
        # map action into ind
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        translated_valid_actions = {}
        device = linking_scores.device

        # fake an empty Statement because there must be valid actions
        if world is None:
            translated_valid_actions[Statement.__name__] = {}
            # assign to take away to keep consistent

            # append to take away
            if self.use_sql_attention and take_away is not None:
                # take a padding vector
                precedent_encoding = torch.zeros((1, self.sql_output_size), device=device) + 1e-6
                precedent_mask = torch.ones((1, 1), device=device)

                take_away['sql_output'] = precedent_encoding
                take_away['sql_output_mask'] = precedent_mask.squeeze(dim=0)

            return GrammarStatelet([Statement.__name__],
                                   translated_valid_actions,
                                   # callback function
                                   SparcParser.is_nonterminal)

        # map copy key to copy ids
        copy_action_dict = {}
        for copy_subtree in world.precedent_segment_seq:
            copy_action_dict[str(copy_subtree)] = copy_subtree

        # from non-terminal to action list, Dict[str, List[str]]
        valid_actions = world.valid_actions
        # 1 x precedent_action_len
        precedent_action = world.precedent_action_seq

        # we need the action entity mapping, indicating which action corresponds to which column/table
        action_to_entity = world.get_action_entity_mapping()

        ProductionTuple = namedtuple('ProductionTuple', ('rule', 'is_global', 'is_copy', 'tensor', 'nonterminal'))

        # we need construct an embedding book for sql query encoder
        if self.use_last_sql:
            copy_embedding_book = {}
            action_unit_indices = list(action_map.values())
            production_unit_arrays = [(ProductionTuple(*possible_actions[index]), index)
                                      for index in action_unit_indices]
            production_unit_arrays = [production_rule for production_rule in production_unit_arrays
                                      if not production_rule[0].is_copy]
            global_unit_actions = []
            linked_unit_actions = []
            for production_unit_array, action_index in production_unit_arrays:
                if production_unit_array.is_global:
                    global_unit_actions.append((production_unit_array.tensor, action_index))
                # avoid padding rules
                elif production_unit_array.rule in action_to_entity:
                    linked_unit_actions.append((production_unit_array.rule, action_index))

            # construct embedding book
            if global_unit_actions:
                action_tensors, action_ids = zip(*global_unit_actions)
                action_tensor = torch.cat(action_tensors, dim=0).long()

                # batch_size * inter_size x embedding_size
                action_unit_embedding = self.sql_global_embedder.forward(action_tensor)
                for ind, idx in enumerate(action_ids):
                    copy_embedding_book[idx] = action_unit_embedding[ind]

            if linked_unit_actions:
                action_rules, action_ids = zip(*linked_unit_actions)
                related_entity_ids = [action_to_entity[rule] for rule in action_rules]
                # FIXME: -1 means it is actually not an entity
                assert -1 not in related_entity_ids
                if self.copy_encode_anon:
                    entity_type_tensor = entity_types[related_entity_ids]
                    entity_type_embeddings = (self.sql_schema_embedder(entity_type_tensor)
                                              .to(entity_types.device)
                                              .float())
                    for ind, idx in enumerate(action_ids):
                        copy_embedding_book[idx] = entity_type_embeddings[ind]
                else:
                    # use specific representations of entity itself
                    for ind, idx in enumerate(action_ids):
                        entity_idx = related_entity_ids[ind]
                        copy_embedding_book[idx] = encoded_schema[entity_idx]
        else:
            copy_embedding_book = {}

        # prepare action encodings for token-level copy operation or segment-level copy
        if self.use_last_sql:
            if len(precedent_action):
                precedent_embedding = torch.stack([copy_embedding_book[action_idx] for action_idx in precedent_action])
                precedent_mask = torch.ones((1, len(precedent_action)), device=device)
                precedent_encoding = self.sql_context_encoder.forward(precedent_embedding.unsqueeze(dim=0),
                                                                      precedent_mask).squeeze(dim=0)
                precedent_forward = precedent_encoding[:, :self.sql_hidden_size]
                precedent_backward = precedent_encoding[:, self.sql_hidden_size:]
            else:
                # eps to avoid nan loss
                precedent_encoding = torch.zeros((1, self.sql_output_size), device=device) + 1e-6
                precedent_mask = torch.ones((1, 1), device=device)

            # append to take away
            if self.use_sql_attention and take_away is not None:
                take_away['sql_output'] = precedent_encoding
                take_away['sql_output_mask'] = precedent_mask.squeeze(dim=0)

        for key, action_strings in valid_actions.items():
            # allocate dictionary
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.
            action_indices = [action_map[action_string] for action_string in action_strings]

            # named tuple for better reading
            production_rule_arrays = [(ProductionTuple(*possible_actions[index]), index) for index in action_indices]

            # split rules into two category
            global_actions = []
            linked_actions = []
            copy_segment_actions = []
            copy_token_actions = []

            for production_rule_array, action_index in production_rule_arrays:
                # copy action
                if self.use_copy_segment and production_rule_array.is_copy:
                    # if encode segment without context, using its rule
                    if self.copy_encode_with_context:
                        # find the start/end pos in encode
                        related_copy_ids = copy_action_dict[production_rule_array.rule].copy_ins_idx
                        # remove padding index to find the position
                        related_copy_ids = [ind for ind in related_copy_ids if ind != self.action_padding_index]
                        # [start, end)
                        copy_action_start_end = find_start_end(precedent_action, related_copy_ids)
                        copy_segment_actions.append((copy_action_start_end, action_index))
                    else:
                        copy_segment_actions.append((production_rule_array.rule, action_index))
                elif not self.use_copy_segment and production_rule_array.is_copy:
                    continue
                else:
                    if self.use_copy_token and action_index in precedent_action:
                        # the index in precedent sequence
                        copy_token_actions.append((precedent_action.index(action_index), action_index))
                    # use copy token could work with global & linked
                    if production_rule_array.is_global:
                        global_actions.append((production_rule_array.tensor, action_index))
                    else:
                        linked_actions.append((production_rule_array.rule, action_index))

            if copy_segment_actions:
                assert len(precedent_action) > 0
                assert self.use_copy_segment
                if self.copy_encode_with_context:
                    # use the span repr
                    action_start_end, action_ids = zip(*copy_segment_actions)
                    copy_sql_encoding = []
                    for start_end_tup in action_start_end:
                        action_start, action_end = start_end_tup
                        copy_sql_encoding.append(get_span_representation(precedent_forward, precedent_backward,
                                                                         action_start, action_end))
                    copy_sql_encoding = torch.stack(copy_sql_encoding, dim=0)
                else:
                    action_rules, action_ids = zip(*copy_segment_actions)
                    # FIXME: we could consider encoding the segment within context
                    related_copy_ids = [copy_action_dict[rule].copy_ins_idx for rule in action_rules]
                    # construct tensor & encoder mask
                    related_copy_ids = torch.tensor(related_copy_ids, dtype=torch.long, device=device)
                    sql_encoder_mask = (related_copy_ids != self.action_padding_index).long()

                    # make related copy ids non-negative
                    related_copy_ids = torch.where(related_copy_ids > 0, related_copy_ids,
                                                   related_copy_ids.new_zeros(related_copy_ids.size()))

                    # construct embedding: action using sql global embedder, otherwise use sql schema embedder
                    copy_sql_embedding = torch.stack(
                        [torch.stack([copy_embedding_book[int(idx)] for idx in idx_list], dim=0)
                         for idx_list in related_copy_ids], dim=0)
                    copy_sql_encoding = self.sql_segment_encoder.forward(copy_sql_embedding, sql_encoder_mask)

                # segment input
                copy_sql_input = self.copy_sql_input.forward(copy_sql_encoding)
                copy_sql_output = self.copy_sql_output.forward(copy_sql_encoding)
                translated_valid_actions[key]['copy_seg'] = (copy_sql_input,
                                                             copy_sql_output,
                                                             list(action_ids))

            if copy_token_actions:
                assert len(precedent_action) > 0
                assert self.use_copy_token
                action_inds, action_ids = zip(*copy_token_actions)
                copy_sql_encoding = []
                for action_ind in action_inds:
                    if self.copy_encode_with_context:
                        copy_sql_encoding.append(precedent_encoding[action_ind])
                    else:
                        action_embedding = precedent_embedding[action_ind]
                        action_embedding = self.sql_embedder_transform.forward(action_embedding)
                        copy_sql_encoding.append(action_embedding)
                copy_sql_encoding = torch.stack(copy_sql_encoding, dim=0)

                # token input
                copy_sql_input = self.copy_sql_input.forward(copy_sql_encoding)
                copy_sql_output = self.copy_sql_output.forward(copy_sql_encoding)
                translated_valid_actions[key]['copy_token'] = (copy_sql_input,
                                                               copy_sql_output,
                                                               list(action_ids))

            if global_actions:
                action_tensors, action_ids = zip(*global_actions)
                action_tensor = torch.cat(action_tensors, dim=0).long()

                # batch_size * inter_size x embedding_size
                action_input_embedding = self.action_embedder.forward(action_tensor)
                action_output_embedding = self.output_action_embedder.forward(action_tensor)
                translated_valid_actions[key]['global'] = (action_input_embedding,
                                                           action_output_embedding,
                                                           list(action_ids))

            if linked_actions:
                # TODO: how to handle the embedding of *
                action_rules, action_ids = zip(*linked_actions)
                related_entity_ids = [action_to_entity[rule] for rule in action_rules]

                # assert related entity ids does not contain -1
                assert -1 not in related_entity_ids
                entity_linking_scores = linking_scores[related_entity_ids]

                entity_type_tensor = entity_types[related_entity_ids]
                entity_type_embeddings = (self.output_entity_type_embedder(entity_type_tensor)
                                          .to(entity_types.device)
                                          .float())

                translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                           entity_type_embeddings,
                                                           list(action_ids))

        return GrammarStatelet([Statement.__name__],
                               translated_valid_actions,
                               # callback function
                               SparcParser.is_nonterminal)

    @staticmethod
    def is_nonterminal(token: str):
        # nonterminal list
        nonterminals = [child.__name__ for child in Action.__subclasses__()]
        if token in nonterminals:
            return True
        else:
            return False

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        action_metrics = self.action_metric.get_metric(reset)
        sql_metrics = self.sql_metric.get_metric(reset)
        if self.use_copy_segment or self.use_copy_token or self.use_context_gate:
            gate_metrics = {key: self.gate_metrics[key].get_metric(reset) for key in self.gate_metrics.keys()}
            metrics = {**action_metrics, **sql_metrics, **gate_metrics}
        else:
            metrics = {**action_metrics, **sql_metrics}
        return metrics

    @staticmethod
    def _get_linking_probabilities(linking_scores: torch.Tensor,
                                   entity_type_mat: torch.LongTensor) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        linking_scores : ``torch.FloatTensor``
            Has shape (batch_size * inter_size, utt_token_size, col_size).
        entity_type_mat : ``torch.LongTensor``
            Has shape (batch_size * inter_size, col_size, entity_size)
        Returns
        -------
        batch_probabilities : ``torch.FloatTensor``
            Has shape ``(batch_size * inter_size, utt_token_size, entity_size)``.
            Contains all the probabilities of entity types given an utterance word
        """
        # normalize entity type mat into probability
        entity_type_base = entity_type_mat.sum(dim=2, keepdim=True).expand_as(entity_type_mat)
        # divide and get the probability, batch_size * inter_size x col_size x entity_size
        entity_type_prob = entity_type_mat / entity_type_base
        # bmm and get the result, batch_size * inter_size x utt_token_size x entity_size
        type_linking_score = torch.bmm(linking_scores, entity_type_prob)
        # normalize on entity dimension
        type_linking_prob = torch.softmax(type_linking_score, dim=2)

        return type_linking_prob
