# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.algo_utils import BipartiteGraphSolver


class HingeLoss(nn.Module):
    def __init__(self, margin=0.6, aggregation='max', l1_norm_weight=0, entropy_norm_weight=0):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.aggregation = aggregation

        self.l1_norm_weight = l1_norm_weight
        self.entropy_norm_weight = entropy_norm_weight

        self.bipartite_graph_solver = BipartiteGraphSolver()

    def forward(self, pos_align, neg_align, lengths):
        # src_lengths, pos_tgt_lengths, neg_tgt_lengths = lengths
        positive_lengths, negative_lengths = lengths
        positive_lengths = positive_lengths.permute(1, 0)
        negative_lengths = negative_lengths.permute(1, 0)
        src_lengths = positive_lengths[0]
        pos_tgt_lengths = positive_lengths[1]
        neg_tgt_lengths = negative_lengths[1]

        '''
        temp = torch.sqrt(torch.FloatTensor([self.args.hidden_size * 2]))
        if self.args.cuda:
            temp = temp.cuda()
        pos_align = torch.div(pos_align, temp)
        neg_align = torch.div(neg_align, temp)
        '''
        # print('pos_align', pos_align)
        # print('neg_align', neg_align)
        positive_n = sum(positive_lengths[0] * positive_lengths[1])
        negative_n = sum(negative_lengths[1] * negative_lengths[1])
        pos_l1_norm, neg_l1_norm = torch.norm(pos_align, p=1) / positive_n, torch.norm(neg_align, p=1) / negative_n
        # print('pos_norm', type(pos_l1_norm), pos_l1_norm)
        # print('neg_norm', type(neg_l1_norm), neg_l1_norm)
        # print('pos_norm', pos_align)
        # print('neg_norm', neg_align)
        # Entropy loss
        pos_row_entropy = F.softmax(pos_align, dim=-1) * F.log_softmax(pos_align, dim=-1)
        neg_row_entropy = F.softmax(neg_align, dim=-1) * F.log_softmax(neg_align, dim=-1)
        pos_row_entropy = -1 * pos_row_entropy.sum()
        neg_row_entropy = -1 * neg_row_entropy.sum()
        pos_col_entropy = F.softmax(pos_align, dim=0) * F.log_softmax(pos_align, dim=0)
        neg_col_entropy = F.softmax(neg_align, dim=0) * F.log_softmax(neg_align, dim=0)
        pos_col_entropy = -1 * pos_col_entropy.sum()
        neg_col_entropy = -1 * neg_col_entropy.sum()

        entropy_norm = pos_row_entropy - neg_row_entropy + pos_col_entropy - neg_col_entropy

        # print('entropy', type(entropy_norm), entropy_norm)

        if self.aggregation == 'max':
            pos_align_score, neg_align_score = torch.max(pos_align, -1)[0], torch.max(neg_align, -1)[0]
        elif self.aggregation == 'sum':
            pos_align_score, neg_align_score = torch.sum(pos_align, -1), torch.sum(neg_align, -1)
            pos_align_score = torch.div(pos_align_score, src_lengths.float().reshape((-1, 1)))
            neg_align_score = torch.div(neg_align_score, src_lengths.float().reshape((-1, 1)))
        elif self.aggregation == 'match':
            pos_align_score = 0
            pos_matrix = [x.detach().cpu().numpy() for x in pos_align]
            pos_assignment_positions = [self.bipartite_graph_solver.find_max(x)[1] for x in pos_matrix]
            for idx, pos_assignment_position in enumerate(pos_assignment_positions):
                for x, y in zip(*pos_assignment_position):
                    pos_align_score += pos_align[idx, x, y]
            pos_align_score /= sum(positive_lengths[0])
            # pos_assignment = [list(zip([i] * len(pos_assignment_positions[0][0]),
            #                            pos_assignment_positions[i][0],
            #                            pos_assignment_positions[i][1]))
            #                   for i in range(len(pos_assignment_positions))]
            # pos_assignment = [_ for x in pos_assignment for _ in x]
            neg_align_score = 0
            neg_matrix = [x.detach().cpu().numpy() for x in neg_align]
            neg_assignment_positions = [self.bipartite_graph_solver.find_max(x)[1] for x in neg_matrix]
            for idx, neg_assignment_position in enumerate(neg_assignment_positions):
                for x, y in zip(*neg_assignment_position):
                    neg_align_score += neg_align[idx, x, y]
            neg_align_score /= sum(negative_lengths[0])
            pass
            # neg_assignment = [list(zip([i] * len(neg_assignment_positions[0][0]),
            #                            neg_assignment_positions[i][0],
            #                            neg_assignment_positions[i][1]))
            #                   for i in range(len(neg_assignment_positions))]
            # neg_assignment = [_ for x in neg_assignment for _ in x]
            # pos_align_score = sum([pos_align[point] for point in pos_assignment])
            # neg_align_score = sum([neg_align[point] for point in neg_assignment])
        else:
            raise ValueError("Hinge loss only supports max/sum aggregation.")

        pos_align_score = torch.sum(pos_align_score, -1)
        neg_align_score = torch.sum(neg_align_score, -1)

        pos_align_score = torch.div(pos_align_score, pos_tgt_lengths.float())
        neg_align_score = torch.div(neg_align_score, neg_tgt_lengths.float())

        hinge_loss = torch.mean(torch.clamp(self.margin - (pos_align_score - neg_align_score), min=0.0)) + \
                     self.l1_norm_weight * (pos_l1_norm + neg_l1_norm) + self.entropy_norm_weight * entropy_norm

        return hinge_loss
