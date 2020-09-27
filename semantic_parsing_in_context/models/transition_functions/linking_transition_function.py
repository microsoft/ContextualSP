# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
The code body is borrowed from allennlp package. We modify it to adapt our tree-level copy

@Author: Qian Liu
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation
from overrides import overrides
from models.states_machine.grammar_based_state import GrammarBasedState
from models.transition_functions.basic_transition_function import BasicTransitionFunction


class LinkingTransitionFunction(BasicTransitionFunction):
    """
    This transition function adds the ability to consider `linked` actions to the
    ``BasicTransitionFunction`` (which is just an LSTM decoder with attention).  These actions are
    potentially unseen at training time, so we need to handle them without requiring the action to
    have an embedding.  Instead, we rely on a `linking score` between each action and the words in
    the question/utterance, and use these scores, along with the attention, to do something similar
    to a copy mechanism when producing these actions.

    When both linked and global (embedded) actions are available, we need some way to compare the
    scores for these two sets of actions.  The original WikiTableQuestion semantic parser just
    concatenated the logits together before doing a joint softmax, but this is quite brittle,
    because the logits might have quite different scales.  So we have the option here of predicting
    a mixture probability between two independently normalized distributions.

    Parameters
    ----------
    encoder_output_dim : ``int``
    action_embedding_dim : ``int``
    input_attention : ``Attention``
    activation : ``Activation``, optional (default=relu)
        The activation that gets applied to the decoder LSTM input and to the action query.
    predict_start_type_separately : ``bool``, optional (default=True)
        If ``True``, we will predict the initial action (which is typically the base type of the
        logical form) using a different mechanism than our typical action decoder.  We basically
        just do a projection of the hidden state, and don't update the decoder RNN.
    num_start_types : ``int``, optional (default=None)
        If ``predict_start_type_separately`` is ``True``, this is the number of start types that
        are in the grammar.  We need this so we can construct parameters with the right shape.
        This is unused if ``predict_start_type_separately`` is ``False``.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, there has been a bias dimension added to the embedding of each action, which
        gets used when predicting the next action.  We add a dimension of ones to our predicted
        action vector in this case to account for that.
    dropout : ``float`` (optional, default=0.0)
    num_layers: ``int`` (optional, default=1)
        The number of layers in the decoder LSTM.
    """

    def __init__(self,
                 encoder_output_dim: int,
                 decoder_input_dim: int,
                 action_embedding_dim: int,
                 input_attention: Attention,
                 sql_attention: Attention = None,
                 sql_output_dim: int = 100,
                 activation: Activation = Activation.by_name('relu')(),
                 predict_start_type_separately: bool = True,
                 num_start_types: int = None,
                 add_action_bias: bool = True,
                 copy_gate: FeedForward = None,
                 dropout: float = 0.0,
                 num_layers: int = 1) -> None:
        super().__init__(encoder_output_dim=encoder_output_dim,
                         decoder_input_dim=decoder_input_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         sql_attention=sql_attention,
                         sql_output_dim=sql_output_dim,
                         num_start_types=num_start_types,
                         activation=activation,
                         predict_start_type_separately=predict_start_type_separately,
                         add_action_bias=add_action_bias,
                         dropout=dropout,
                         num_layers=num_layers)
        # control the copy gate
        self._copy_gate = copy_gate

    @overrides
    def _compute_action_probabilities(self,
                                      state: GrammarBasedState,
                                      hidden_state: torch.Tensor,
                                      attention_weights: torch.Tensor,
                                      predicted_action_embeddings: torch.Tensor
                                      ) -> Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]]:
        # In this section we take our predicted action embedding and compare it to the available
        # actions in our current state (which might be different for each group element).  For
        # computing action scores, we'll forget about doing batched / grouped computation, as it
        # adds too much complexity and doesn't speed things up, anyway, with the operations we're
        # doing here.  This means we don't need any action masks, as we'll only get the right
        # lengths for what we're computing.

        group_size = len(state.batch_indices)
        actions = state.get_valid_actions()

        batch_results: Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]] = defaultdict(list)
        for group_index in range(group_size):
            instance_actions = actions[group_index]
            predicted_action_embedding = predicted_action_embeddings[group_index]
            embedded_actions: List[int] = []

            output_action_embeddings = None
            embedded_action_logits = None

            if not instance_actions:
                action_ids = None
                current_log_probs = float('inf')
            else:
                if 'global' in instance_actions:
                    action_embeddings, output_action_embeddings, embedded_actions = instance_actions['global']
                    # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
                    # (embedding_dim, 1) matrix.
                    action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
                    action_ids = embedded_actions

                    # 'copy_seg' is designed to compatible with global, not with linked
                    if 'copy_seg' in instance_actions:
                        # we should concat copy logits into action_logits
                        copy_action_encodings, output_copy_action_encodings, copy_action_ids = instance_actions[
                            'copy_seg']
                        copy_action_logits = copy_action_encodings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(
                            -1)
                        # concat logits with action_logits
                        action_logits = torch.cat([action_logits, copy_action_logits], dim=0)
                        output_action_embeddings = torch.cat([output_action_embeddings, output_copy_action_encodings],
                                                             dim=0)
                        action_ids = action_ids + copy_action_ids

                elif 'linked' in instance_actions:
                    linking_scores, type_embeddings, linked_actions = instance_actions['linked']
                    action_ids = embedded_actions + linked_actions
                    # (num_question_tokens, 1)
                    linked_action_logits = linking_scores.mm(attention_weights[group_index].unsqueeze(-1)).squeeze(-1)

                    # The `output_action_embeddings` tensor gets used later as the input to the next
                    # decoder step.  For linked actions, we don't have any action embedding, so we use
                    # the entity type instead.
                    if output_action_embeddings is not None:
                        output_action_embeddings = torch.cat([output_action_embeddings, type_embeddings], dim=0)
                    else:
                        output_action_embeddings = type_embeddings

                    # 'copy_seg' is designed to compatible with global, not with linked
                    if embedded_action_logits is not None:
                        action_logits = torch.cat([embedded_action_logits, linked_action_logits], dim=-1)
                    else:
                        action_logits = linked_action_logits

                    # in hard token copy, the column could also be copied
                    if 'copy_seg' in instance_actions:
                        # we should concat copy logits into action_logits
                        copy_action_encodings, output_copy_action_encodings, copy_action_ids = instance_actions['copy_seg']
                        copy_action_logits = copy_action_encodings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
                        output_action_embeddings = torch.cat([output_action_embeddings, output_copy_action_encodings],
                                                             dim=0)
                        # concat logits with action_logits
                        action_logits = torch.cat([action_logits, copy_action_logits], dim=0)
                        output_action_embeddings = torch.cat([output_action_embeddings, output_copy_action_encodings],
                                                             dim=0)
                        action_ids = action_ids + copy_action_ids
                else:
                    raise Exception("Not support for such an instance action")

                # we will use copy gate to obtain the overall probability as:
                # p = gen * action_prob + copy * copy_action_prob
                if 'copy_token' in instance_actions:
                    copy_action_encodings, output_copy_action_encodings, copy_action_ids = instance_actions[
                        'copy_token']
                    copy_action_logits = copy_action_encodings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
                    copy_action_prob = torch.softmax(copy_action_logits, dim=0)
                    generate_action_prob = torch.softmax(action_logits, dim=0)
                    # align token id to generation ones
                    copy_to_gen_prob = torch.zeros(generate_action_prob.size(),
                                                   device=generate_action_prob.device).float()
                    for i in range(len(copy_action_ids)):
                        copy_action = copy_action_ids[i]
                        if copy_action in action_ids:
                            ind = action_ids.index(copy_action)
                            copy_to_gen_prob[ind] = copy_action_prob[i]

                    assert self._copy_gate is not None
                    # use copy_gate to calculate the copy gate
                    copy_gate = torch.sigmoid(self._copy_gate(hidden_state[group_index]))
                    action_prob = generate_action_prob * (1 - copy_gate) + copy_gate * copy_to_gen_prob
                    current_log_probs = torch.log(torch.clamp(action_prob, min=1e-10))
                else:
                    current_log_probs = torch.log_softmax(action_logits, dim=-1)

            # This is now the total score for each state after taking each action.  We're going to
            # sort by this later, so it's important that this is the total score, not just the
            # score for the current action.
            log_probs = state.score[group_index] + current_log_probs
            batch_results[state.batch_indices[group_index]].append((group_index,
                                                                    log_probs,
                                                                    current_log_probs,
                                                                    output_action_embeddings,
                                                                    action_ids))
        return batch_results
