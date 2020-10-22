import pdb
import random
import statistics
from itertools import chain

import math
import torch.nn.functional as F
from torch import nn
from masked_cross_entropy import *
from utils import Categorical
from modules.BinaryTreeBasedModule import BinaryTreeBasedModule
from utils import clamp_grad
import torch

USE_CUDA = torch.cuda.is_available()

PAD_token = 0
SOS_token = 1
EOS_token = 2
x1_token = 3
x2_token = 4
x3_token = 5
x4_token = 6

all_action_words = ["i_look", "i_jump", "i_walk", "i_run", "i_turn_right", "i_turn_left"]
available_src_vars = ['x1', 'x2', 'x3', 'x4']

MAX_LENGTH = 10


def flatten(l):
    for el in l:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el


class BottomUpTreeComposer(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim,
                 self_attention_in_tree=False, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.sr_linear = nn.Linear(in_features=hidden_dim, out_features=2)

        self.use_self_attention = self_attention_in_tree

        if self.use_self_attention:
            self.q = nn.Parameter(torch.empty(size=(hidden_dim * 2,), dtype=torch.float32))
        else:
            self.q = nn.Parameter(torch.empty(size=(hidden_dim,), dtype=torch.float32))
        # if use self attention, we should employ these parameters
        self.bilinear_w = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)
        nn.init.uniform_(self.bilinear_w.weight, -0.1, 0.1)

    def forward(self, pair, x, mask,
                relaxed=False, tau_weights=None, straight_through=False, noise=None,
                eval_actions=None, eval_sr_actions=None, eval_swr_actions=None, debug_info=None):
        input_span = pair[0].split()
        span_start_end = [[i, i] for i in range(len(input_span))]

        x = self.embd_parser(x)
        probs = []
        gumbel_noise = []
        actions = []
        entropy = []
        normalized_entropy = []
        log_prob = []

        sr_probs = []
        sr_gumbel_noise = []
        sr_actions = []
        sr_entropy = []
        sr_normalized_entropy = []
        sr_log_prob = []

        hidden, cell = self._transform_leafs(x, mask)
        swr_probs = []
        swr_gumbel_noise = []
        swr_actions = []
        swr_entropy = []
        swr_normalized_entropy = []
        swr_log_prob = []

        reduce_span = []

        all_span = []
        tree_sr_log_prob = []

        # make debug info List of List
        debug_reduce_probs = []
        debug_merge_probs = []

        for i in range(x.shape[1]):
            noise_i = None if noise is None else noise[i]
            eval_swr_actions_i = None if eval_swr_actions is None else eval_swr_actions[i]
            swr_cat_distr, swr_gumbel_noise_i, swr_actions_i = self._swr_make_step(hidden, i, relaxed, tau_weights,
                                                                                   straight_through, noise_i,
                                                                                   eval_swr_actions_i)
            hidden, cell = self.swr_abst_embed(hidden, cell, i, swr_actions_i)
            if swr_actions_i[0, 0] == 1:
                input_span[i] = "x1"
            debug_reduce_probs.append(float(swr_cat_distr.probs[0][0]))
            swr_probs.append(swr_cat_distr.probs)
            swr_gumbel_noise.append(swr_gumbel_noise_i)
            swr_actions.append(swr_actions_i)
            swr_entropy.append(swr_cat_distr.entropy)
            swr_normalized_entropy.append(swr_cat_distr.normalized_entropy)
            swr_log_prob.append(-swr_cat_distr.log_prob(swr_actions_i))

            span = [i, i]

            all_span.append(span)
            tree_sr_log_prob.append(-swr_cat_distr.log_prob(swr_actions_i))

            if swr_actions_i[0, 0] == 1 or x.shape[1] == 1:
                reduce_span.append(span)

        # swr_log_prob = None if relaxed else sum(swr_log_prob)
        swr_entropy = sum(swr_entropy)
        swr_normalized_entropy = sum(swr_normalized_entropy) / (torch.sum(mask[:, 0:], dim=-1) + 1e-17)

        swr_rl_info = [swr_entropy, swr_normalized_entropy, swr_actions, swr_log_prob]

        if x.shape[1] == 1:
            actions = []
            sr_actions = []
            entropy, normalized_entropy, log_prob = 0, 0, 0
            sr_entropy, sr_normalized_entropy, sr_log_prob = 0, 0, 0
        else:
            for i in range(1, x.shape[1]):

                noise_i = None if noise is None else noise[i - 1]
                ev_actions_i = None if eval_actions is None else eval_actions[i - 1]
                eval_sr_actions_i = None if eval_sr_actions is None else eval_sr_actions[i - 1]
                cat_distr, gumbel_noise_i, actions_i, hidden, cell = self._make_step(hidden, cell, mask[:, i:], relaxed,
                                                                                     tau_weights,
                                                                                     straight_through, noise_i,
                                                                                     ev_actions_i)
                # add merge prob distribution
                debug_merge_probs.extend([float(ele) for ele in cat_distr.probs[0]])

                probs.append(cat_distr.probs)
                gumbel_noise.append(gumbel_noise_i)
                actions.append(actions_i)
                entropy.append(cat_distr.entropy)
                normalized_entropy.append(cat_distr.normalized_entropy)
                log_prob.append(-cat_distr.log_prob(actions_i))

                sr_cat_distr, sr_gumbel_noise_i, sr_actions_i = self._sr_make_step(hidden, actions_i, relaxed,
                                                                                   tau_weights,
                                                                                   straight_through, noise_i,
                                                                                   eval_sr_actions_i)
                action_idx = actions_i[0].argmax().item()
                merged_span = " ".join(input_span[action_idx:action_idx + 2])

                if len(merged_span.split()) >= 3:
                    sr_actions_i = torch.tensor([[1., 0.]]).cuda()
                if merged_span.count(available_src_vars[0]) >= 2:
                    sr_actions_i = torch.tensor([[1., 0.]]).cuda()

                if sr_actions_i[0, 0] == 1:
                    merged_span = available_src_vars[0]

                input_span = input_span[:action_idx] + [merged_span] + input_span[action_idx + 2:]

                hidden, cell = self.abst_embed(hidden, cell, actions_i, sr_actions_i)

                debug_reduce_probs.append(float(sr_cat_distr.probs[0][0]))
                sr_probs.append(sr_cat_distr.probs)
                sr_gumbel_noise.append(sr_gumbel_noise_i)
                sr_actions.append(sr_actions_i)

                sr_entropy.append(sr_cat_distr.entropy)
                sr_normalized_entropy.append(sr_cat_distr.normalized_entropy)
                sr_log_prob.append(-sr_cat_distr.log_prob(sr_actions_i))

                log_prob_step = - cat_distr.log_prob(actions_i) - sr_cat_distr.log_prob(sr_actions_i)
                span_left = span_start_end[action_idx]
                span_right = span_start_end[action_idx + 1]
                span = [span_left[0], span_right[1]]

                all_span.append(span)
                tree_sr_log_prob.append(log_prob_step)

                if (sr_actions_i[0, 0] == 1) or (i == x.shape[1] - 1):
                    reduce_span.append(span)

                span_start_end = span_start_end[:action_idx] + [span] + span_start_end[action_idx + 2:]

            # log_prob = None if relaxed else sum(log_prob)
            entropy = sum(entropy)
            # normalize by the number of layers - 1.
            # -1 because the last layer contains only one possible action and the entropy is zero anyway.
            normalized_entropy = sum(normalized_entropy) / (torch.sum(mask[:, 2:], dim=-1) + 1e-17)

            # sr_log_prob = None if relaxed else sum(sr_log_prob)
            sr_entropy = sum(sr_entropy)
            sr_normalized_entropy = sum(sr_normalized_entropy) / (torch.sum(mask[:, 1:], dim=-1) + 1e-17)

            assert relaxed is False

        tree_rl_infos = [entropy, normalized_entropy, actions, log_prob]
        sr_rl_infos = [sr_entropy, sr_normalized_entropy, sr_actions, sr_log_prob]

        reduce_span = [reduce_span[-1]]

        spans_info = []
        for span in all_span:
            span_start = span[0]
            span_end = span[1]
            for fspan in reduce_span:
                fspan_start = fspan[0]
                fspan_end = fspan[1]
                if fspan_start <= span_start and span_end <= fspan_end:
                    distance = (span_start - fspan_start) + (fspan_end - span_end)
                    span_info = [span, fspan, distance]
                    spans_info.append(span_info)
                    break

        if debug_info is not None:
            debug_info["merge_prob"] = debug_merge_probs
            debug_info["reduce_prob"] = debug_reduce_probs

        return tree_rl_infos, sr_rl_infos, swr_rl_info, tree_sr_log_prob, spans_info

    def swr_abst_embed(self, hidden, cell, i, swr_actions_i):

        if swr_actions_i[0, 0] == 1:
            word_index = i
            x_mask = torch.tensor([[1.]]).cuda()
            src_var_id = x1_token
            h_x, c_x = self._transform_leafs(self.embd_parser(torch.tensor([[src_var_id]]).cuda()), mask=x_mask)
            h_p_new = torch.cat([hidden[:, :word_index], h_x, hidden[:, word_index + 1:]], dim=1)
            c_p_new = torch.cat([cell[:, :word_index], c_x, cell[:, word_index + 1:]], dim=1)
        else:
            h_p_new, c_p_new = hidden, cell
        return h_p_new, c_p_new

    def abst_embed(self, hidden, cell, actions_i, sr_actions_i):

        if sr_actions_i[0, 0] == 1:
            actions_index = actions_i[0].argmax().item()
            x_mask = torch.tensor([[1.]]).cuda()
            src_var_id = x1_token
            h_x, c_x = self._transform_leafs(self.embd_parser(torch.tensor([[src_var_id]]).cuda()), mask=x_mask)
            h_p_new = torch.cat([hidden[:, :actions_index], h_x, hidden[:, actions_index + 1:]], dim=1)
            c_p_new = torch.cat([cell[:, :actions_index], c_x, cell[:, actions_index + 1:]], dim=1)
        else:
            h_p_new, c_p_new = hidden, cell
        return h_p_new, c_p_new

    def _swr_make_step(self, hidden, i, relaxed, tau_weights, straight_through, gumbel_noise, ev_swr_actions):
        # ==== calculate the prob distribution over the merge actions and sample one ====
        word_index = i
        h_word = hidden[:, word_index]
        sr_score = self.sr_linear(h_word)
        sr_mask = torch.ones_like(sr_score)

        sr_cat_distr = Categorical(sr_score, sr_mask)
        if ev_swr_actions is None:
            sr_actions, gumbel_noise = self._sample_action(sr_cat_distr, sr_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            sr_actions = ev_swr_actions

        return sr_cat_distr, gumbel_noise, sr_actions

    def _sr_make_step(self, hidden, actions_i, relaxed, tau_weights, straight_through, gumbel_noise, ev_sr_actions):
        # ==== calculate the prob distribution over the merge actions and sample one ====
        actions_index = actions_i.argmax(dim=-1)[0]
        h_act = hidden[:, actions_index]
        sr_score = self.sr_linear(h_act)
        sr_mask = torch.ones_like(sr_score)

        sr_cat_distr = Categorical(sr_score, sr_mask)
        if ev_sr_actions is None:
            sr_actions, gumbel_noise = self._sample_action(sr_cat_distr, sr_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            sr_actions = ev_sr_actions

        return sr_cat_distr, gumbel_noise, sr_actions

    def _make_step(self, hidden, cell, mask, relaxed, tau_weights, straight_through, gumbel_noise, ev_actions):
        # ==== calculate the prob distribution over the merge actions and sample one ====

        h_l, c_l = hidden[:, :-1], cell[:, :-1]
        h_r, c_r = hidden[:, 1:], cell[:, 1:]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)

        if self.use_self_attention:
            cand_size = h_p.shape[1]
            query_vector = h_p.unsqueeze(dim=2).repeat(1, 1, cand_size, 1). \
                view(-1, cand_size * cand_size, self.hidden_dim)
            value_vector = h_p.unsqueeze(dim=1).repeat(1, cand_size, 1, 1). \
                view(-1, cand_size * cand_size, self.hidden_dim)
            attn_score = torch.tanh(self.bilinear_w(query_vector, value_vector))
            attn_weights = F.softmax(attn_score.view(-1, cand_size, cand_size), dim=2).view(-1, cand_size * cand_size,
                                                                                            1)
            value_vector_flatten = value_vector * attn_weights
            attn_vector = value_vector_flatten.view(-1, cand_size, cand_size, self.hidden_dim).sum(dim=2)
            q_mul_vector = torch.cat([h_p, attn_vector], dim=-1)
        else:
            q_mul_vector = h_p

        score = torch.matmul(q_mul_vector, self.q)  # (N x L x d, d) -> (N x L)
        cat_distr = Categorical(score, mask)
        if ev_actions is None:
            actions, gumbel_noise = self._sample_action(cat_distr, mask, relaxed, tau_weights, straight_through,
                                                        gumbel_noise)
        else:
            actions = ev_actions
        # ==== incorporate sampled action into the agent's representation of the environment state ====
        h_p, c_p = BinaryTreeBasedModule._merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask)

        return cat_distr, gumbel_noise, actions, h_p, c_p

    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise


class BinaryTreeEncoder(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim, vocab_size, input_lang,
                 leaf_transformation=BinaryTreeBasedModule.no_transformation, trans_hidden_dim=None, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.input_lang = input_lang
        self.embd_tree = nn.Embedding(vocab_size, input_dim)

    def forward(self, input_token, parse_tree, mask, reduce_info, actions_scale):
        input_embed = self.embd_tree(input_token)
        hidden_reduce, cell_reduce = self._transform_leafs(input_embed, mask)
        stop_idx = reduce_info['stop_idx']
        reduce_idx = reduce_info['reduce_idx']
        reduce_idx2reduce_x = reduce_info['reduce_idx2reduce_x']
        mask_idx = 1
        hidden_subtrees_dict = {}
        for idx in range(hidden_reduce.shape[1]):
            hidden_subtree = hidden_reduce[:, idx:idx + 1, :]
            hidden_subtrees_dict[str([idx, idx])] = hidden_subtree

        for i in range(stop_idx + 1):
            if isinstance(parse_tree[i], int):
                hidden, cell = hidden_reduce, cell_reduce
                merge_pos = parse_tree[i]
                if i in reduce_idx:

                    reduce_x = reduce_idx2reduce_x[i]
                    x_token = self.input_lang.word2index[reduce_x]
                    x_embed = self.embd_tree(torch.tensor([[x_token]]).cuda())
                    x_mask = torch.tensor([[1.]]).cuda()
                    hidden_x, cell_x = self._transform_leafs(x_embed, x_mask)
                    hidden_reduce = torch.cat([hidden[:, :merge_pos, :], hidden_x, hidden[:, merge_pos + 1:, :]], dim=1)
                    cell_reduce = torch.cat([cell[:, :merge_pos, :], cell_x, cell[:, merge_pos + 1:, :]], dim=1)
                else:
                    hidden_reduce, cell_reduce = hidden, cell

            elif parse_tree[i] is None:
                hidden, cell = hidden_reduce, cell_reduce
                merge_pos = parse_tree[i - 1]

            else:

                h_l, c_l = hidden_reduce[:, :-1], cell_reduce[:, :-1]
                h_r, c_r = hidden_reduce[:, 1:], cell_reduce[:, 1:]
                h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)
                hidden, cell = self._merge(parse_tree[i], h_l, c_l, h_r, c_r, h_p, c_p, mask[:, mask_idx:])
                mask_idx += 1
                merge_pos = (parse_tree[i][0] == 1).nonzero()[0, 0]
                if i in reduce_idx:

                    reduce_x = reduce_idx2reduce_x[i]
                    x_token = self.input_lang.word2index[reduce_x]
                    x_embed = self.embd_tree(torch.tensor([[x_token]]).cuda())
                    x_mask = torch.tensor([[1.]]).cuda()
                    hidden_x, cell_x = self._transform_leafs(x_embed, x_mask)
                    hidden_reduce = torch.cat([hidden[:, :merge_pos, :], hidden_x, hidden[:, merge_pos + 1:, :]], dim=1)
                    cell_reduce = torch.cat([cell[:, :merge_pos, :], cell_x, cell[:, merge_pos + 1:, :]], dim=1)
                else:
                    hidden_reduce, cell_reduce = hidden, cell

            if i < stop_idx:
                scale = actions_scale[i][0]
                hidden_subtrees_dict[str(scale)] = hidden_reduce[:, merge_pos:merge_pos + 1, :]

        if reduce_idx:
            for reduce_i in reduce_idx:
                start, end = actions_scale[reduce_i][0]
                for scale in hidden_subtrees_dict.copy():

                    scale_start, scale_end = scale.lstrip('[').rstrip(']').split(',')
                    scale_start, scale_end = int(scale_start), int(scale_end)
                    if (scale_start > start and scale_end <= end) or (scale_start >= start and scale_end < end):
                        hidden_subtrees_dict.pop(scale)

        scale_stop = actions_scale[stop_idx][0]
        start, end = scale_stop
        hidden_subtree_list = []
        for scale in hidden_subtrees_dict:
            scale_start, scale_end = scale.lstrip('[').rstrip(']').split(',')
            scale_start, scale_end = int(scale_start), int(scale_end)
            if scale_start >= start and scale_end <= end:
                hidden_subtree_list.append(hidden_subtrees_dict[scale])

        hidden_subtree = torch.cat(hidden_subtree_list, dim=1)

        final_merge_hidden = hidden[:, merge_pos:merge_pos + 1, :]
        final_merge_cell = cell[:, merge_pos:merge_pos + 1, :]

        return final_merge_hidden.transpose(0, 1), final_merge_cell.transpose(0, 1), hidden_subtree.transpose(0, 1),


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.0, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # outputs, hidden = self.gru(packed, hidden)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return hidden, cell, outputs


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        # self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        rnn_input = word_embedded

        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # output = F.log_softmax(self.out(output), dim=-1)
        output = self.out(output)
        # Return final output, hidden state
        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = (torch.ByteTensor(mask).unsqueeze(1)).cuda()  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        # self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, n_layers)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        # self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, last_cell, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        rnn_input = torch.cat((word_embedded, context), 2)
        output, (hidden, cell) = self.lstm(rnn_input, (last_hidden, last_cell))
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        output = self.out(output)
        return output, hidden, cell


class EncoderDecoderSolver(nn.Module):
    def __init__(self, word_dim, hidden_dim, vocab_size, label_dim, input_lang, output_lang,
                 encode_mode=None, x_ratio_rate=None):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.encode_mode = encode_mode
        self.encoder = EncoderRNN(vocab_size, word_dim, hidden_dim)
        self.decoder = BahdanauAttnDecoderRNN(hidden_dim, word_dim, label_dim)
        self.x_ratio_rate = x_ratio_rate

    def get_action_scale_infer(self, actions, all_reduce_idx):
        actions_scale = []
        all_scale = []
        for idx in range(len(actions)):
            action = actions[idx]
            if isinstance(action, int):
                action_pos = [action, action]
                all_scale.append([action_pos, []])
                action_scale = [all_scale[action][0], all_scale[action][1]]
                actions_scale.append(action_scale)
                if idx in all_reduce_idx:
                    all_scale[action] = [action_pos, [action_pos]]

                continue

            elif action is None:
                assert len(actions) == 2
                action_scale = [all_scale[0][0], all_scale[0][1]]
                actions_scale.append(action_scale)
                continue

            action_index = action.argmax(dim=-1)[0]
            scale_left_pos = all_scale[action_index][0][0]
            scale_right_pos = all_scale[action_index + 1][0][1]
            action_scale_pos = [scale_left_pos, scale_right_pos]

            scale_left_x = all_scale[action_index][1]
            scale_right_x = all_scale[action_index + 1][1]
            action_scale_x = scale_left_x + scale_right_x

            action_scale = [action_scale_pos, action_scale_x]

            if idx not in all_reduce_idx:
                actions_scale.append(action_scale)
            elif idx in all_reduce_idx:
                actions_scale.append(action_scale)
                action_scale = [action_scale_pos, [action_scale_pos]]

            all_scale = all_scale[:action_index] + [action_scale] + all_scale[action_index + 2:]

        return actions_scale

    def forward(self, pair, actions, sr_actions, swr_actions, input_batches, input_mask, epoch, debug_info=None):

        rewards = []
        reward_scale = []
        reward_scale2idx = {}

        all_reduce_idx = []
        global_memory_output = {}

        all_actions = [idx for idx in range(len(swr_actions))] + actions
        all_sr_actions = swr_actions + sr_actions

        actions_scale = self.get_action_scale_infer(all_actions, all_reduce_idx)

        if len(pair[0].split()) == 1:
            scale2idx = {'[0, 0]': 0}
        else:
            scale2idx = {}
            for idx, scale in enumerate(actions_scale):
                scale2idx[str(scale[0])] = idx

        compose_infos = []
        for idx in range(len(actions_scale)):
            compose_info = {'step': idx,
                            'stop_idx': idx,
                            'reduce_idx': [],
                            'reduce_idx2reduce_x': {},
                            'reduce_x2reduce_idx': {}
                            }
            compose_infos.append(compose_info)

        normalized_entropy = []
        log_probs = []
        decoded_words = []
        count_x_ratio = []
        debug_decoder_results = []
        debug_encoder_inputs = []

        for idx in range(len(all_actions)):
            compose_info = compose_infos[idx]
            if self.encode_mode == 'seq':

                input_sen = pair[0].split()
                span_start_pos, span_end_pos = actions_scale[idx][0][0], actions_scale[idx][0][1]
                x_spans = actions_scale[idx][1]
                if len(x_spans) == 0:
                    input_var_constant_seq = input_sen[span_start_pos:span_end_pos + 1]
                elif len(x_spans) == 1:
                    src_slot = x_spans[0]
                    src_slot_idx = scale2idx[str(src_slot)]
                    src_var = compose_info['reduce_idx2reduce_x'][src_slot_idx]
                    src_slot_start_pos, src_slot_end_pos = src_slot[0], src_slot[1]
                    input_var_constant_seq = input_sen[span_start_pos:src_slot_start_pos] + \
                                             [src_var] + input_sen[src_slot_end_pos + 1:span_end_pos + 1]
                elif len(x_spans) == 2:
                    src_slot_first = x_spans[0]
                    src_slot_second = x_spans[1]
                    assert src_slot_second[0] > src_slot_first[1]
                    src_slot_first_idx = scale2idx[str(src_slot_first)]
                    src_slot_second_idx = scale2idx[str(src_slot_second)]
                    src_var_first = compose_info['reduce_idx2reduce_x'][src_slot_first_idx]
                    src_var_second = compose_info['reduce_idx2reduce_x'][src_slot_second_idx]
                    src_slot_first_start_pos, src_slot_first_end_pos = src_slot_first[0], src_slot_first[1]
                    src_slot_second_start_pos, src_slot_second_end_pos = src_slot_second[0], src_slot_second[1]
                    input_var_constant_seq = input_sen[span_start_pos:src_slot_first_start_pos] + \
                                             [src_var_first] + input_sen[
                                                               src_slot_first_end_pos + 1:src_slot_second_start_pos] + \
                                             [src_var_second] + input_sen[src_slot_second_end_pos + 1:span_end_pos + 1]

                input_var_constant_idx = [self.input_lang.word2index[word]
                                          for word in input_var_constant_seq] + [EOS_token]
                input_batches = torch.tensor(input_var_constant_idx).unsqueeze(1).cuda()
                input_length = [len(input_batches)]

                encoder_hidden, encoder_cell, hidden_subtree = self.encoder(input_batches, input_length)

            else:
                encoder_hidden, encoder_cell, hidden_subtree = self.encoder(input_batches, all_actions, input_mask,
                                                                            compose_info,
                                                                            actions_scale)

            if all_sr_actions[idx][0, 0] == 1:
                if self.training:
                    get_sup, possible_tokens = self.jud_sup(idx, all_actions, compose_info, global_memory_output, pair)
                    if random.random() < 0.8:
                        use_sup = False
                    else:
                        use_sup = True
                else:
                    get_sup, possible_tokens = False, {}
                    use_sup = False

                # get_sup = False

                if get_sup is False or use_sup is False:
                    decoded_words, normalized_entropy_step, log_prob_step, decoded_words_ori = self.get_sub_output(
                        encoder_hidden, encoder_cell, hidden_subtree,
                        compose_info,
                        all_actions[idx], global_memory_output)
                else:
                    decoded_words, normalized_entropy_step, log_prob_step, decoded_words_ori = self.get_sub_output_sup(
                        encoder_hidden, encoder_cell, hidden_subtree,
                        compose_info,
                        all_actions[idx], global_memory_output,
                        possible_tokens, pair)

                normalized_entropy.append(normalized_entropy_step)
                log_probs.append(log_prob_step)
                all_reduce_idx.append(idx)

                # memorize all intermediate outputs
                global_memory_output[idx] = decoded_words

                # add debug outputs
                debug_decoder_results.append(" ".join([self.output_lang.index2word[idx]
                                                       for idx in decoded_words_ori]))

                actions_scale = self.get_action_scale_infer(all_actions, all_reduce_idx)

                reward = self.cal_reward(decoded_words, pair[1])

                if idx >= len(swr_actions):
                    temp_count_x = 0
                    for src_var in available_src_vars:
                        if src_var in compose_info['reduce_x2reduce_idx']:
                            tgt_var_idx = available_src_vars.index(src_var) + x1_token
                            if tgt_var_idx in decoded_words_ori:
                                temp_count_x += decoded_words_ori.count(tgt_var_idx)
                    # sum of count
                    count_x_ratio.append(temp_count_x / len(decoded_words_ori))

                rewards.append(reward)
                reward_scale.append(actions_scale[idx])
                reward_scale2idx[str(actions_scale[idx][0])] = len(rewards) - 1

            if all_sr_actions[idx][0, 0] == 0 and idx == len(all_actions) - 1:
                rewards.append(0.)
                reward_scale.append(actions_scale[idx])
                reward_scale2idx[str(actions_scale[idx][0])] = len(rewards) - 1
                log_probs.append(torch.tensor([0.]).cuda())

            if all_sr_actions[idx][0, 0] == 1:
                # add debug inputs
                debug_reduce_info = compose_infos[idx]
                # fetch out the anonymize span
                debug_action_scale = actions_scale[idx]
                tokens = pair[0].split(" ")
                if len(debug_reduce_info['reduce_idx']) == 0:
                    temp_span = debug_action_scale[0]
                    debug_encoder_inputs.append(" ".join(tokens[temp_span[0]: temp_span[1] + 1]))
                else:
                    reduce_steps = debug_reduce_info['reduce_idx']
                    anno_tokens = list(tokens)
                    for step in reduce_steps:
                        temp_span = actions_scale[step][0]
                        anno_tokens[temp_span[0]] = debug_reduce_info['reduce_idx2reduce_x'][step]
                        for i in range(temp_span[0] + 1, temp_span[1] + 1):
                            anno_tokens[i] = ""
                    anno_tokens = anno_tokens[actions_scale[idx][0][0]: actions_scale[idx][0][1] + 1]
                    debug_encoder_inputs.append(" ".join([token for token in anno_tokens
                                                          if token != '']))
            else:
                debug_decoder_results.append("NONE")
                debug_encoder_inputs.append("NONE")

            compose_infos = []
            for idy in range(len(actions_scale)):
                compose_info = {'step': idy,
                                'stop_idx': idy,
                                'reduce_idx': [],
                                'reduce_idx2reduce_x': {},
                                'reduce_x2reduce_idx': {}
                                }
                action_scale = actions_scale[idy]
                if action_scale[1] != []:
                    # x_index = 1
                    x_index = 0
                    src_var_list = available_src_vars[:2]
                    if self.training:
                        random.shuffle(src_var_list)
                    for scale in action_scale[1]:
                        scale_idx = scale2idx[str(scale)]
                        compose_info['reduce_idx'].append(scale_idx)
                        # x_str = 'x' + str(x_index)
                        if x_index > 1:
                            # print("x num > 1")
                            x_index = 0
                        x_str = src_var_list[x_index]
                        compose_info['reduce_idx2reduce_x'][scale_idx] = x_str
                        compose_info['reduce_x2reduce_idx'][x_str] = scale_idx
                        x_index += 1
                compose_infos.append(compose_info)

        if decoded_words and all_sr_actions[-1][0, 0] == 1:
            final_output = [word for word in flatten(decoded_words)]
            final_output_words = [self.output_lang.index2word[token] for token in final_output]
            pred_labels = " ".join(final_output_words)
        else:
            pred_labels = ""

        if normalized_entropy:
            normalized_entropy = sum(normalized_entropy) / len(normalized_entropy)
        else:
            normalized_entropy = 0.

        avg_full_var_ratio = statistics.mean(
            [1.0 if count_x_ratio[i] >= 0.99 else 0.0 for i in range(len(count_x_ratio))]) \
            if len(count_x_ratio) > 0 else 0.0

        if pred_labels == pair[1]:
            rewards[-1] = rewards[-1] + avg_full_var_ratio * self.x_ratio_rate
        rewards = self.iter_rewards(rewards, reward_scale, reward_scale2idx)

        span2reward = {}
        for reward, scale in zip(rewards, reward_scale):
            span2reward[str(scale[0])] = reward

        if debug_info is not None:
            debug_info["decoder_outputs"] = debug_decoder_results
            debug_info["decoder_inputs"] = debug_encoder_inputs

        return pred_labels, normalized_entropy, log_probs, rewards, span2reward

    def jud_sub_right(self, idx, all_actions, reduce_info, all_sub_output, pair):
        sub_right = False
        if idx == len(all_actions) - 1:
            if reduce_info['reduce_idx2reduce_x']:
                x_output_pair = []
                for reduce_idx in reduce_info['reduce_idx2reduce_x']:
                    sub_output = all_sub_output[reduce_idx]
                    if sub_output:
                        sub_output_flat = [word for word in flatten(sub_output)]
                        final_output_words = [self.output_lang.index2word[token] for token in sub_output_flat]
                        sub_pred_labels = " ".join(final_output_words)
                    else:
                        sub_pred_labels = ""
                    x_output_pair.append([reduce_info['reduce_idx2reduce_x'][reduce_idx], sub_pred_labels])
                if len(x_output_pair) == 1:
                    if x_output_pair[0][1] != '':
                        if x_output_pair[0][1] in pair[1]:
                            sub_right = True
                else:
                    assert len(x_output_pair) == 2
                    if x_output_pair[0][1] != '' and x_output_pair[1][1] != '':
                        if x_output_pair[0][1] in pair[1] and x_output_pair[1][1] in pair[1]:
                            sub_right = True
        return sub_right

    def jud_sup(self, idx, all_actions, reduce_info, all_sub_output, pair):
        get_sup = False

        possible_tokens = {len(pair[1].split()): [EOS_token]}
        output_tokens = [self.output_lang.word2index[word] for word in pair[1].split()]
        for idx, token in enumerate(output_tokens):
            possible_tokens[idx] = [token]

        if idx == len(all_actions) - 1:
            if reduce_info['reduce_idx2reduce_x']:
                x_output_pair = []
                for reduce_idx in reduce_info['reduce_idx2reduce_x']:
                    sub_output = all_sub_output[reduce_idx]
                    sub_output_flat = [word for word in flatten(sub_output)]
                    final_output_words = [self.output_lang.index2word[token] for token in sub_output_flat]
                    sub_pred_labels = " ".join(final_output_words)
                    x_output_pair.append([reduce_info['reduce_idx2reduce_x'][reduce_idx], sub_pred_labels])

                if len(x_output_pair) == 1:
                    if x_output_pair[0][1] in pair[1]:
                        get_sup = True

                        output_with_x = pair[1]
                        sub_labels = x_output_pair[0][1]
                        x_used = x_output_pair[0][0]
                        output_with_x = output_with_x.replace(sub_labels, x_used)
                        plus_idy = 0
                        plus_length = len(sub_labels.split()) - 1
                        possible_tokens[x_used] = plus_length
                        for idy, word in enumerate(output_with_x.split()):
                            if word == x_used:
                                possible_tokens[plus_idy + idy].append(self.output_lang.word2index[x_used])
                                plus_idy = plus_idy + plus_length

                else:
                    assert len(x_output_pair) == 2
                    if x_output_pair[0][1] in pair[1] and x_output_pair[1][1] in pair[1]:
                        get_sup = True

                        output_with_x = pair[1]
                        sub_labels = x_output_pair[0][1]
                        x_used = x_output_pair[0][0]
                        output_with_x = output_with_x.replace(sub_labels, x_used)
                        plus_idy = 0
                        plus_length = len(sub_labels.split()) - 1
                        possible_tokens[x_used] = plus_length
                        for idy, word in enumerate(output_with_x.split()):
                            if word == x_used:
                                possible_tokens[plus_idy + idy].append(self.output_lang.word2index[x_used])
                                plus_idy = plus_idy + plus_length

                        output_with_x = pair[1]
                        sub_labels = x_output_pair[1][1]
                        x_used = x_output_pair[1][0]
                        output_with_x = output_with_x.replace(sub_labels, x_used)
                        plus_idy = 0
                        plus_length = len(sub_labels.split()) - 1
                        possible_tokens[x_used] = plus_length
                        for idy, word in enumerate(output_with_x.split()):
                            if word == x_used:
                                possible_tokens[plus_idy + idy].append(self.output_lang.word2index[x_used])
                                plus_idy = plus_idy + plus_length

        return get_sup, possible_tokens

    def get_local_refer(self, sr_scale_sent, final_sent):
        if sr_scale_sent in self.composes_instrs:
            return self.composes_outputs[sr_scale_sent]
        else:
            return final_sent

    def same_rewards(self, rewards):
        rewards_all_same = []
        for idx, _ in enumerate(rewards):
            rewards_all_same.append(rewards[-1])
        return rewards

    def iter_rewards(self, rewards, reward_scale, reward_scale2idx):
        rewards_num = len(rewards)
        for idx in reversed(range(rewards_num)):
            reward_global = rewards[idx]
            scale = reward_scale[idx]
            scales_affect = scale[1]
            if not scales_affect:
                continue
            else:
                for scale_affect in scales_affect:
                    scale_affect_idx = reward_scale2idx[str(scale_affect)]
                    rewards[scale_affect_idx] = reward_global
        return rewards

    def cal_reward(self, decoded_words, target_sent):
        if decoded_words:
            final_output = [word for word in flatten(decoded_words)]
            final_output_words = [self.output_lang.index2word[token] for token in final_output]
            pred_labels = " ".join(final_output_words)
        else:
            pred_labels = ""

        common_labels = self.get_longest_common_substring(pred_labels, target_sent)
        common_labels_length = len(common_labels)
        pred_labels_length = len(pred_labels.split())
        target_labels_length = len(target_sent.split())
        iou_similar = common_labels_length / (pred_labels_length + target_labels_length - common_labels_length)

        reward = iou_similar
        return reward

    def get_sub_output_sup(self, encoder_hidden, encoder_cell, hidden_subtree, reduce_info, action, all_sub_output,
                           possible_tokens,
                           pair):
        x_tokens = reduce_info['reduce_x2reduce_idx']

        ################################################################################
        if isinstance(action, int):
            possible_output_tokens_first = [self.output_lang.word2index[x] for x in all_action_words]
            possible_output_tokens = possible_output_tokens_first + [EOS_token]

        else:
            # possible_output_tokens = [EOS_token]
            possible_output_tokens_first = [self.output_lang.word2index[x] for x in all_action_words]
            for x in available_src_vars:
                if x in x_tokens:
                    possible_output_tokens_first.append(self.output_lang.word2index[x])
                else:
                    continue
            possible_output_tokens = possible_output_tokens_first + [EOS_token]

        mask_first = [1. if token in possible_output_tokens_first else 0. for token in range(self.output_lang.n_words)]
        decode_mask_first = torch.tensor([mask_first]).cuda()

        mask = [1. if token in possible_output_tokens else 0. for token in range(self.output_lang.n_words)]
        decode_mask = torch.tensor([mask]).cuda()

        decoder_input = torch.LongTensor([SOS_token])
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        decoded_words = []
        decoded_words_ori = []

        normalized_entropy = []
        log_prob = []

        di = 0
        while True:
            assert di <= len(pair[1].split())

            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell, hidden_subtree
            )
            decode_score = decoder_output
            # if reduce_info['step'] == 4:
            #     pdb.set_trace()
            if di == 0:
                cat_distr = Categorical(decode_score, decode_mask_first)
            else:
                cat_distr = Categorical(decode_score, decode_mask)

            mask_di = [1. if token in possible_tokens[di] else 0. for token in range(self.output_lang.n_words)]
            decode_mask_di = torch.tensor([mask_di]).cuda()
            cat_distr_di = Categorical(decode_score, decode_mask_di)
            decode_actions, gumbel_noise = self._sample_action(cat_distr_di, False, None)

            normalized_entropy.append(cat_distr.normalized_entropy)
            log_prob.append(-cat_distr.log_prob(decode_actions))

            topv, topi = decode_actions.data.topk(1)

            ni = topi[0][0]

            if ni == EOS_token:
                if di == 0:
                    pdb.set_trace()
                # decoded_words.append('<EOS>')
                break
            else:

                ni_word = self.output_lang.index2word[ni.item()]
                if ni_word in x_tokens:
                    x_reduce_idx = reduce_info['reduce_x2reduce_idx'][ni_word]
                    x_sub_output = all_sub_output[x_reduce_idx]
                    decoded_words.append(x_sub_output)
                    di = di + possible_tokens[ni_word]
                else:
                    decoded_words.append(ni.item())
                decoded_words_ori.append(ni.item())
            decoder_input = torch.LongTensor([ni]).cuda()
            di += 1

        normalized_entropy = sum(normalized_entropy) / len(normalized_entropy)
        log_prob = sum(log_prob)

        return decoded_words, normalized_entropy, log_prob, decoded_words_ori

    def get_sub_output_length_sup(self, encoder_hidden, hidden_subtree, reduce_info, action, all_sub_output, pair):

        x_tokens = reduce_info['reduce_x2reduce_idx']

        min_length = len(pair[1].split())

        # final_output_tokens = [self.output_lang.word2index[word] for word in final_output_sent.split()]

        ################################################################################
        if isinstance(action, int):
            possible_output_tokens = [self.output_lang.word2index[x] for x in all_action_words] + [EOS_token]
            max_length = MAX_LENGTH

        else:
            possible_output_tokens = [EOS_token]
            # possible_output_tokens = [self.output_lang.word2index[x] for x in all_output_prims] + [EOS_token]
            for x in available_src_vars:
                if x in x_tokens:
                    possible_output_tokens.append(self.output_lang.word2index[x])
                else:
                    continue
            max_length = MAX_LENGTH

        mask = [1. if token in possible_output_tokens else 0. for token in range(self.output_lang.n_words)]
        decode_mask = torch.tensor([mask]).cuda()

        while True:
            if x_tokens:
                expected_tokens = [token for token in x_tokens]
            else:
                expected_tokens = []

            decoder_input = Variable(torch.LongTensor([SOS_token]))
            decoder_hidden = encoder_hidden

            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoded_words = []
            decoded_words_ori = []

            normalized_entropy = []
            log_prob = []

            total_length = 0

            for di in range(max_length):

                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, hidden_subtree
                )
                decode_score = decoder_output
                # if reduce_info['step'] == 4:
                #     pdb.set_trace()
                cat_distr = Categorical(decode_score, decode_mask)
                decode_actions, gumbel_noise = self._sample_action(cat_distr, False, None)

                if total_length < min_length:
                    pdb.set_trace()

                normalized_entropy.append(cat_distr.normalized_entropy)
                log_prob.append(-cat_distr.log_prob(decode_actions))

                topv, topi = decode_actions.data.topk(1)

                ni = topi[0][0]

                if ni == EOS_token:
                    # decoded_words.append('<EOS>')
                    break
                else:

                    ni_word = self.output_lang.index2word[ni.item()]
                    if ni_word in x_tokens:
                        x_reduce_idx = reduce_info['reduce_x2reduce_idx'][ni_word]
                        x_sub_output = all_sub_output[x_reduce_idx]
                        decoded_words.append(x_sub_output)
                        length_step = len([word for word in flatten(x_sub_output)])
                        if ni_word in expected_tokens:
                            expected_tokens.remove(ni_word)
                    else:
                        decoded_words.append(ni.item())
                        length_step = 1
                    decoded_words_ori.append(ni.item())
                    total_length = total_length + length_step
                decoder_input = torch.LongTensor([ni]).cuda()
            break

        normalized_entropy = sum(normalized_entropy) / len(normalized_entropy)
        # normalized_entropy = sum(normalized_entropy)
        log_prob = sum(log_prob)

        return decoded_words, normalized_entropy, log_prob, decoded_words_ori

    def get_sub_output(self, encoder_hidden, encoder_cell, hidden_subtree, reduce_info, action, all_sub_output):

        used_src_vars = reduce_info['reduce_x2reduce_idx']

        if isinstance(action, int):
            possible_output_tokens_first = [self.output_lang.word2index[x] for x in all_action_words]
            max_length = MAX_LENGTH
            possible_output_tokens = possible_output_tokens_first + [EOS_token]

        else:
            # possible_output_tokens = [EOS_token]
            possible_output_tokens_first = [self.output_lang.word2index[x] for x in all_action_words]
            for src_var in available_src_vars:
                if src_var in used_src_vars:
                    tgt_var = self.output_lang.word2index[src_var]
                    possible_output_tokens_first.append(tgt_var)
                else:
                    continue
            max_length = MAX_LENGTH
            possible_output_tokens = possible_output_tokens_first + [EOS_token]

        mask_first = [1. if token in possible_output_tokens_first else 0. for token in range(self.output_lang.n_words)]
        decode_mask_first = torch.tensor([mask_first]).cuda()

        mask = [1. if token in possible_output_tokens else 0. for token in range(self.output_lang.n_words)]
        decode_mask = torch.tensor([mask]).cuda()

        while True:
            if used_src_vars:
                expected_tokens = [token for token in used_src_vars]
            else:
                expected_tokens = []

            decoder_input = Variable(torch.LongTensor([SOS_token]))
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoded_words = []
            decoded_words_ori = []

            normalized_entropy = []
            log_prob = []

            for di in range(max_length):

                decoder_output, decoder_hidden, decoder_cell = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, hidden_subtree
                )
                decode_score = decoder_output
                if di == 0:
                    cat_distr = Categorical(decode_score, decode_mask_first)
                else:
                    cat_distr = Categorical(decode_score, decode_mask)
                decode_actions, gumbel_noise = self._sample_action(cat_distr, False, None)

                normalized_entropy.append(cat_distr.normalized_entropy)
                log_prob.append(-cat_distr.log_prob(decode_actions))

                topv, topi = decode_actions.data.topk(1)

                ni = topi[0][0]

                if ni == EOS_token:
                    break
                else:
                    ni_word = self.output_lang.index2word[ni.item()]
                    if ni_word in used_src_vars:
                        x_reduce_idx = reduce_info['reduce_x2reduce_idx'][ni_word]
                        x_sub_output = all_sub_output[x_reduce_idx]
                        decoded_words.append(x_sub_output)
                        if ni_word in expected_tokens:
                            expected_tokens.remove(ni_word)
                    else:
                        decoded_words.append(ni.item())
                    decoded_words_ori.append(ni.item())
                decoder_input = Variable(torch.LongTensor([ni])).cuda()

            break

        normalized_entropy = sum(normalized_entropy) / len(normalized_entropy)
        log_prob = sum(log_prob)

        return decoded_words, normalized_entropy, log_prob, decoded_words_ori

    def _sample_action(self, cat_distr, relaxed, gumbel_noise):
        if self.training:
            assert relaxed is False
            actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise

    def get_longest_common_substring(self, pred_sent, target_sent):
        output_ori2syb = {"i_look": "a",
                          "i_jump": "b",
                          "i_walk": "c",
                          "i_run": "d",
                          "i_turn_right": "e",
                          "i_turn_left": "f",
                          " ": ""}
        for ori in output_ori2syb:
            pred_sent = pred_sent.replace(ori, output_ori2syb[ori])
            target_sent = target_sent.replace(ori, output_ori2syb[ori])

        lstr1 = len(pred_sent)
        lstr2 = len(target_sent)
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        maxNum = 0
        p = 0

        for i in range(lstr1):
            for j in range(lstr2):
                if pred_sent[i] == target_sent[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]
                        p = i + 1

        return pred_sent[p - maxNum:p]


class HRLModel(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, label_dim,
                 decay_r, encode_mode, x_ratio_rate,
                 composer_leaf=BinaryTreeBasedModule.no_transformation, composer_trans_hidden=None,
                 input_lang=None, output_lang=None):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.label_dim = label_dim
        self.composer = BottomUpTreeComposer(word_dim, hidden_dim, vocab_size, composer_leaf,
                                             composer_trans_hidden)
        self.solver = EncoderDecoderSolver(word_dim, hidden_dim, vocab_size, label_dim, input_lang,
                                           output_lang,
                                           encode_mode=encode_mode,
                                           x_ratio_rate=x_ratio_rate)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.reset_parameters()
        self.is_test = False

        self.decay_r = decay_r

    def get_policy_parameters(self):
        return list(chain(self.composer.parameters()))

    def get_environment_parameters(self):
        return list(chain(self.solver.parameters()))

    def forward(self, pair, x, mask, is_test=False, epoch=None, debug_info=None):
        self.is_test = is_test

        normalized_entropy, tree_actions, sr_actions, swr_actions, pred_labels, tree_sr_log_prob, tree_sr_rewards, decoder_log_probs, decode_rewards = self._forward(
            pair, x, mask, epoch, debug_info=debug_info)
        return pred_labels, tree_sr_log_prob, tree_sr_rewards, decoder_log_probs, decode_rewards, \
               tree_actions, sr_actions, swr_actions, normalized_entropy

    def _forward(self, pair, x, mask, epoch, debug_info):

        tree_rl_infos, sr_rl_infos, swr_rl_info, tree_sr_log_prob, spans_info = self.composer(pair, x, mask,
                                                                                              debug_info=debug_info)
        tree_entropy, tree_normalized_entropy, tree_actions, tree_log_prob = tree_rl_infos
        sr_entropy, sr_normalized_entropy, sr_actions, sr_log_prob = sr_rl_infos
        swr_entropy, swr_normalized_entropy, swr_actions, swr_log_prob = swr_rl_info

        # actions = self.get_left_actions(x)
        pred_labels, decoder_normalized_entropy, decoder_log_probs, decode_rewards, span2reward = self.solver(
            pair, tree_actions,
            sr_actions, swr_actions, x, mask,
            epoch, debug_info=debug_info)

        assert len(decoder_log_probs) == len(decode_rewards) == len(span2reward)

        tree_sr_rewards = []
        decode_from_root_rewards = []
        swr_sr_actions = swr_actions + sr_actions
        decay_r = self.decay_r

        for idx, span_info in enumerate(spans_info):
            fspan = span_info[1]
            freward = span2reward[str(fspan)]
            fr = span_info[2]
            tree_sr_reward = freward * (decay_r ** fr)
            tree_sr_rewards.append(tree_sr_reward)
            if swr_sr_actions[idx][0, 0] == 1 or idx == len(swr_sr_actions) - 1:
                decode_from_root_rewards.append(tree_sr_reward)

        assert len(decode_rewards) == len(decode_from_root_rewards)
        assert len(tree_sr_rewards) == len(tree_sr_log_prob)

        normalized_entropy = tree_normalized_entropy + sr_normalized_entropy + swr_normalized_entropy + decoder_normalized_entropy

        return normalized_entropy, tree_actions, sr_actions, swr_actions, pred_labels, tree_sr_log_prob, tree_sr_rewards, decoder_log_probs, decode_from_root_rewards
