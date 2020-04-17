# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List, Tuple
import torch
import statistics
from allennlp.nn import util
from allennlp.state_machines.constrained_beam_search import ConstrainedBeamSearch
from allennlp.state_machines.states import State
from allennlp.state_machines.trainers.decoder_trainer import DecoderTrainer
from allennlp.state_machines.transition_functions import TransitionFunction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MaximumMarginalLikelihood(DecoderTrainer[Tuple[torch.Tensor, torch.Tensor]]):
    """
    This class trains a decoder by maximizing the marginal likelihood of the targets.  That is,
    during training, we are given a `set` of acceptable or possible target sequences, and we
    optimize the `sum` of the probability the model assigns to each item in the set.  This allows
    the model to distribute its probability mass over the set however it chooses, without forcing
    `all` of the given target sequences to have high probability.  This is helpful, for example, if
    you have good reason to expect that the correct target sequence is in the set, but aren't sure
    `which` of the sequences is actually correct.

    This implementation of maximum marginal likelihood requires the model you use to be `locally
    normalized`; that is, at each decoding timestep, we assume that the model creates a normalized
    probability distribution over actions.  This assumption is necessary, because we do no explicit
    normalization in our loss function, we just sum the probabilities assigned to all correct
    target sequences, relying on the local normalization at each time step to push probability mass
    from bad actions to good ones.

    Parameters
    ----------
    beam_size : ``int``, optional (default=None)
        We can optionally run a constrained beam search over the provided targets during decoding.
        This narrows the set of transition sequences that are marginalized over in the loss
        function, keeping only the top ``beam_size`` sequences according to the model.  If this is
        ``None``, we will keep all of the provided sequences in the loss computation.
    """
    def __init__(self, beam_size: int = None, re_weight: bool = False, loss_mask: int = 6) -> None:
        self._beam_size = beam_size
        self._re_weight = re_weight
        # mask the loss to not back-propagate
        self._loss_mask = loss_mask

    def decode(self,
               initial_state: State,
               transition_function: TransitionFunction,
               supervision: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:

        targets, target_mask = supervision
        # batch_size x inter_size x action_size x index_size(no use)
        assert len(targets.size()) == 4
        # -> batch_size * inter_size x action_size
        batch_size, inter_size, _, _ = targets.size()

        # TODO: we must keep the shape because the loss_mask
        targets = targets.reshape(batch_size * inter_size, -1)

        target_mask = target_mask.reshape(batch_size * inter_size, -1)

        inter_mask = target_mask.sum(dim=1).ne(0)

        # un squeeze beam search dimension
        targets = targets.unsqueeze(dim=1)
        target_mask = target_mask.unsqueeze(dim=1)

        beam_search = ConstrainedBeamSearch(self._beam_size, targets, target_mask)
        finished_states: Dict[int, List[State]] = beam_search.search(initial_state, transition_function)

        inter_count = inter_mask.view(batch_size, inter_size).sum(dim=0).float()
        if 0 not in inter_count:
            inter_ratio = 1.0 / inter_count
        else:
            inter_ratio = torch.ones_like(inter_count)

        loss = 0

        for iter_ind, instance_states in finished_states.items():
            scores = [state.score[0].view(-1) for state in instance_states]
            lens = [len(state.action_history[0]) for state in instance_states]
            if not len(lens):
                continue
            # the i-round of an interaction, starting from 0
            cur_inter = iter_ind % inter_size
            if self._re_weight:
                loss_coefficient = inter_ratio[cur_inter]
            else:
                loss_coefficient = 1.0

            if self._loss_mask <= cur_inter:
                continue

            cur_loss = - util.logsumexp(torch.cat(scores)) / statistics.mean(lens)
            loss += loss_coefficient * cur_loss

        if self._re_weight:
            return {'loss': loss / len(inter_count)}
        elif self._loss_mask < inter_size:
            valid_counts = inter_count[:self._loss_mask].sum()
            return {'loss': loss / valid_counts}
        else:
            return {'loss': loss / len(finished_states)}
