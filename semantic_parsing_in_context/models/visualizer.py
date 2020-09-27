# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List
import matplotlib
import torch
from allennlp.data.vocabulary import Vocabulary
from tensorboardX import SummaryWriter

matplotlib.use('agg', warn=False, force=True)

EMOJI_CORRECT = "&#128523;"
EMOJI_ERROR = "&#128545;"


class Visualizer(object):

    def __init__(self, summary_dir, validation_size, vocab: Vocabulary):
        """

        :param summary_dir: folder to store the tensorboard X log files
        :param validation_size:
        """
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self.log_writer = SummaryWriter(summary_dir)
        self.validation_size = validation_size
        self.global_step = 0
        self.ind_to_token = vocab.get_token_from_index

        # define template
        self.sql_template = '**Utterance** : {0} \n\n**GroundTruth**: {3}\n\n{1}  **SQL**: {2}'

    def log_sql(self, inter_utterance: Dict[str, torch.LongTensor],
                judge_result: List[int],
                ground_truth: List[str],
                encoder_mask: torch.LongTensor,
                inter_sql: List[str]):
        """
        This method is designed to log latent rotated text into tensorboard
        """
        logging_strs = []

        if 'tokens' in inter_utterance:
            inter_tokens = inter_utterance['tokens']
            name_space = 'tokens'
        else:
            inter_tokens = inter_utterance['bert']
            name_space = 'bert'

        for inter_ind, token_seq in enumerate(inter_tokens):
            # fetch the actual sequence length and convert them into token str
            token_len = encoder_mask[inter_ind].sum().long().data.cpu().item()
            origin_tokens = [self.ind_to_token(ind, name_space) for ind in token_seq[:token_len].data.cpu().numpy()]
            # original string
            utterance_str = ' '.join(origin_tokens)
            # segment ids logging
            sql_str = ' , '.join(inter_sql[inter_ind])
            emoji_str = EMOJI_CORRECT if judge_result[inter_ind] == 1 else EMOJI_ERROR
            # record the actual translating length for avoiding extra logging
            logging_str = self.sql_template.format(utterance_str, emoji_str, sql_str, ground_truth[inter_ind])
            logging_strs.append(logging_str)

        # if not anyone, log into the EMPTY
        if len(logging_strs) == 0:
            logging_strs.append('*EMPTY*')

        # merge multiple segment
        logging_str = ('\n\n' + '=' * 120 + '\n\n').join(logging_strs)

        dev_case = self.global_step % self.validation_size
        dev_step = self.global_step // self.validation_size
        self.log_writer.add_text(f'{dev_case}-th Latent Interaction Text', logging_str, global_step=dev_step)

    def update_global_step(self):
        """
        Update global step for logging
        :return:
        """
        self.global_step += 1
