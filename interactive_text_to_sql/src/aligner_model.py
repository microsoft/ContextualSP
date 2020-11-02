# coding: utf-8

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel

from src.data import SpiderAlignDataset, BertUtil, SpiderSemQLConverter
from src.loss import HingeLoss
from src.utils.schema_linker import SchemaLinker
from src.utils.semql_tree_util import Node as SemQLTree


class BatchCosineSimilarity(nn.Module):
    def __init__(self):
        super(BatchCosineSimilarity, self).__init__()

    def forward(self, input1, input2):
        """
        input1 & input2 are both embedding sequences from rnn encoder
        """
        batch_dot_product = torch.bmm(input1, input2.transpose(1, 2))
        # print('bdp', batch_dot_product)
        norm_1, norm_2 = torch.norm(input1, p=2, dim=-1), torch.norm(input2, p=2, dim=-1)
        # print('norm1', norm_1)
        # print('norm2', norm_2)
        norm_matrix = torch.bmm(torch.unsqueeze(norm_1, -1), torch.unsqueeze(norm_2, 1)) + 1e-8
        # print('norm_matrix', norm_matrix)
        assert norm_matrix.size() == batch_dot_product.size()

        cosine_similarity = torch.div(batch_dot_product, norm_matrix)
        return cosine_similarity


class BertAlignerModel(torch.nn.Module):
    def __init__(self, hidden_dim=100, use_autoencoder=False):
        self.use_autoencoder = use_autoencoder
        super().__init__()
        bert_pretrained_weights_shortcut = 'bert-base-uncased'
        bert_output_dim = 768
        self.hidden_dim = hidden_dim
        # self.hidden_dim = bert_output_dim
        self.bert_chosen_layer = -2
        self.bert_model = BertModel.from_pretrained(bert_pretrained_weights_shortcut, output_hidden_states=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(bert_output_dim, hidden_dim, bias=True),
            nn.Sigmoid()
        )
        # fc_weight = np.eye(bert_output_dim)[:hidden_dim]
        # self.fc[0].weight = nn.Parameter(torch.from_numpy(fc_weight).float())
        if self.use_autoencoder:
            self.defc = nn.Linear(hidden_dim, bert_output_dim, bias=True)  # using autoencoder for better supervision
        self.similarity_layer = BatchCosineSimilarity()
        self.criterion = HingeLoss(l1_norm_weight=0.1)
        self.optimizer = optim.Adam([
            {'params': self.bert_model.parameters(), 'lr': 0},
            {'params': self.fc.parameters(), 'lr': 1e-3}
        ])

    def forward(self, positive_tensor, positive_lengths, positive_weight_matrix=None,
                negative_tensor=None, negative_lengths=None, negative_weight_matrix=None,
                ques_max_len=None, pos_max_len=None, neg_max_len=None, mode='train'):
        assert mode in ('train', 'eval')
        positive_lengths = positive_lengths.permute(1, 0)
        positive_bert_output_all = self.bert_model(positive_tensor)
        if self.bert_chosen_layer == -1:
            positive_bert_output = positive_bert_output_all[0].detach()[:, 1:-1]
        else:
            positive_bert_hidden_all = positive_bert_output_all[2]
            positive_bert_output = positive_bert_hidden_all[self.bert_chosen_layer][:, 1:-1]
        positive_output = self.fc(self.dropout(positive_bert_output))
        # positive_output = positive_bert_output
        if self.use_autoencoder:
            rebuild_positive_bert_output = self.defc(positive_output)
            positive_diff = (rebuild_positive_bert_output - positive_bert_output).sum()
        batch_size = positive_output.size(0)
        question_max_len = ques_max_len[0] if ques_max_len is not None else positive_lengths[0].max()
        positive_max_len = pos_max_len[0] if pos_max_len is not None else positive_lengths[1].max()
        _DeviceTensor = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
        positive_question_matrix = _DeviceTensor(batch_size, question_max_len, self.hidden_dim).zero_()
        positive_bert_question_matrix = _DeviceTensor(batch_size, question_max_len, 768).zero_()
        positive_matrix = _DeviceTensor(batch_size, positive_max_len, self.hidden_dim).zero_()
        positive_bert_matrix = _DeviceTensor(batch_size, positive_max_len, 768).zero_()
        if negative_tensor is not None:
            negative_lengths = negative_lengths.permute(1, 0)
            negative_bert_output_all = self.bert_model(negative_tensor)
            if self.bert_chosen_layer == -1:
                negative_bert_output = negative_bert_output_all[0][:, 1:-1]
            else:
                negative_bert_hidden_all = negative_bert_output_all[2]
                negative_bert_output = negative_bert_hidden_all[self.bert_chosen_layer].detach()[:, 1:-1]
            # negative_output = negative_bert_output
            negative_output = self.fc(self.dropout(negative_bert_output))
            if self.use_autoencoder:
                rebuild_negative_bert_output = self.defc(negative_output)
                negative_diff = (rebuild_negative_bert_output - negative_bert_output).sum()
            negative_max_len = neg_max_len[0] if neg_max_len is not None else negative_lengths[1].max()
            negative_question_matrix = _DeviceTensor(batch_size, question_max_len, self.hidden_dim).zero_()
            negative_bert_question_matrix = _DeviceTensor(batch_size, question_max_len, 768).zero_()
            negative_matrix = _DeviceTensor(batch_size, negative_max_len, self.hidden_dim).zero_()
            negative_bert_matrix = _DeviceTensor(batch_size, negative_max_len, 768).zero_()
        for batch_idx in range(batch_size):
            positive_question_matrix[batch_idx, :positive_lengths[0][batch_idx]] = positive_output[batch_idx, :positive_lengths[0][batch_idx]]
            positive_matrix[batch_idx, :positive_lengths[1][batch_idx]] = \
                positive_output[batch_idx, positive_lengths[0][batch_idx] + 1: positive_lengths[0][batch_idx] + positive_lengths[1][batch_idx] + 1]
            positive_bert_question_matrix[batch_idx, :positive_lengths[0][batch_idx]] = positive_bert_output[batch_idx, :positive_lengths[0][batch_idx]]
            positive_bert_matrix[batch_idx, :positive_lengths[1][batch_idx]] = \
                positive_bert_output[batch_idx, positive_lengths[0][batch_idx] + 1: positive_lengths[0][batch_idx] + positive_lengths[1][batch_idx] + 1]
            if negative_tensor is not None:
                negative_question_matrix[batch_idx, :negative_lengths[0][batch_idx]] = negative_output[batch_idx, :negative_lengths[0][batch_idx]]
                negative_matrix[batch_idx, :negative_lengths[1][batch_idx]] = \
                    negative_output[batch_idx, negative_lengths[0][batch_idx] + 1: negative_lengths[0][batch_idx] + negative_lengths[1][batch_idx] + 1]
                negative_bert_question_matrix[batch_idx, :negative_lengths[0][batch_idx]] = negative_bert_output[batch_idx, :negative_lengths[0][batch_idx]]
                negative_bert_matrix[batch_idx, :negative_lengths[1][batch_idx]] = \
                    negative_bert_output[batch_idx,  negative_lengths[0][batch_idx] + 1: negative_lengths[0][batch_idx] + negative_lengths[1][batch_idx] + 1]
        if mode == 'train':
            positive_similarity_matrix = self.similarity_layer(positive_question_matrix, positive_matrix)
        else:
            positive_similarity_matrix = (self.similarity_layer(positive_question_matrix, positive_matrix) + self.similarity_layer(positive_bert_question_matrix, positive_bert_matrix)) / 2
        if mode == 'eval' and positive_weight_matrix is not None:
            positive_similarity_matrix *= positive_weight_matrix
        if negative_tensor is not None:
            if mode == 'train':
                negative_similarity_matrix = self.similarity_layer(positive_question_matrix, negative_matrix)
            else:
                negative_similarity_matrix = (self.similarity_layer(positive_question_matrix, negative_matrix) + self.similarity_layer(negative_bert_question_matrix, negative_bert_matrix)) / 2
            if mode == 'eval' and negative_weight_matrix is not None:
                negative_similarity_matrix *= negative_weight_matrix
        if self.use_autoencoder:
            if negative_tensor is not None:
                return positive_similarity_matrix, negative_similarity_matrix, positive_diff + negative_diff
            else:
                return positive_similarity_matrix, positive_diff
        else:
            if negative_tensor is not None:
                return positive_similarity_matrix, negative_similarity_matrix
            else:
                return positive_similarity_matrix


class BertAligner:
    def __init__(self, aligner_model: BertAlignerModel = None):
        if aligner_model is not None:
            self.aligner_model = aligner_model
        else:
            self.aligner_model = BertAlignerModel()
            if os.path.exists('saved/spider/model.pt'):
                self.aligner_model.load_state_dict(torch.load('saved/spider/model.pt'))
            else:
                logging.warning("No pretrined aligned model loaded!!!")
        if torch.cuda.is_available():
            self.aligner_model = self.aligner_model.cuda()
        self.bert_util = BertUtil()
        self.semql_converter = SpiderSemQLConverter()
        self.schema_linker = SchemaLinker()

    def calculate_alignment(self, nl, restatement):
        assert nl and restatement
        tokens, lengths = self.bert_util.join_sentences([nl,  restatement])
        ids = torch.LongTensor(self.bert_util.tokens_to_ids(tokens)).unsqueeze(0)
        lengths_tensor = torch.LongTensor(lengths).unsqueeze(0)
        if torch.cuda.is_available():
            ids = ids.cuda()
            lengths_tensor = lengths_tensor.cuda()
        if self.aligner_model.use_autoencoder:
            alignment_matrix, autoencoder_diff = self.aligner_model(ids, lengths_tensor, mode='eval')
        else:
            alignment_matrix = self.aligner_model(ids, lengths_tensor, mode='eval')
        return alignment_matrix, ids, tokens, lengths

    def split_tokens(self, tokens, lengths):
        assert len(tokens) == sum(lengths) + 3
        tokens1 = tokens[1:1 + lengths[0]]
        tokens2 = tokens[2 + lengths[0]:-1]
        return tokens1, tokens2

    def load_example(self, example):
        nl = example['question']
        semql = self.semql_converter.convert_example(example)
        semql_tree = SemQLTree.from_statements([str(_) for _ in semql])
        restatement = semql_tree.restatement()
        return nl, restatement

    def link_example_schema(self, example):
        ret = self.schema_linker.link_example(example)
        return ret


if __name__ == '__main__':
    bert_aligner = BertAligner()
    examples = json.load(open('data/spider/train_spider.json', 'r', encoding='utf-8'))
    nl, restatement = bert_aligner.load_example(examples[0])
    alignment_matrix = bert_aligner.calculate_alignment(nl, restatement)
