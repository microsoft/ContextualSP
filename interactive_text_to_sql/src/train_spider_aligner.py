# coding: utf-8

import logging
import sys

import random
import os

from tqdm import tqdm
import numpy as np
import torch

from src.data import SpiderAlignDataset
from src.aligner_model import BertAlignerModel
from src.utils.utils import AverageMeter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

# What are the names of colleges that have two or more players, listed in descending alphabetical order?

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# random.seed(1229)
# torch.manual_seed(1229)
# torch.cuda.manual_seed(1229)

batch_size = 16 * len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
n_negative = 50

total_training_iter = 0

coterms = [x.strip() for x in open('data/spider/coterms.txt', 'r').readlines()]


def train(model, dataloader, criterion, optimizer):
    global total_training_iter
    model.train()
    with tqdm(dataloader) as tqdm_dataloader:
        average_meter = AverageMeter()
        for batch_data in tqdm_dataloader:
            tensors, weights, lengths, texts = batch_data
            positive_tensors, negative_tensors = tensors
            positive_weights, negative_weights = weights
            positive_lengths, negative_lengths = lengths
            positive_texts, negative_texts = texts
            positive_tensors, negative_tensors = positive_tensors.to(device), negative_tensors.to(device)
            positive_weights, negative_weights = positive_weights.to(device), negative_weights.to(device)
            positive_lengths, negative_lengths = positive_lengths.to(device), negative_lengths.to(device)
            batch_size = positive_tensors.size(0)
            ques_max_len = torch.LongTensor([positive_lengths[:, 0].max()]).expand(batch_size, 1)
            pos_max_len = torch.LongTensor([positive_lengths[:, 1].max()]).expand(batch_size, 1)
            neg_max_len = torch.LongTensor([negative_lengths[:, 1].max()]).expand(batch_size, 1)
            # positive_similar_matrix, negative_similar_matrix = model(positive_tensors, positive_lengths, negative_tensors, negative_lengths)
            if (not isinstance(model, torch.nn.DataParallel) and model.use_autoencoder) or \
                    (isinstance(model, torch.nn.DataParallel) and model.module.use_autoencoder):
                positive_similar_matrix, negative_similar_matrix, autoencoder_diff = \
                    model(positive_tensors, positive_lengths, positive_weights,
                          negative_tensors, negative_lengths, negative_weights,
                          ques_max_len, pos_max_len, neg_max_len, mode='train')
            else:
                positive_similar_matrix, negative_similar_matrix = \
                    model(positive_tensors, positive_lengths, positive_weights,
                          negative_tensors, negative_lengths, negative_weights,
                          ques_max_len, pos_max_len, neg_max_len, mode='train')
                autoencoder_diff = None
            if torch.cuda.is_available():
                positive_lengths = positive_lengths.cuda()
                negative_lengths = negative_lengths.cuda()
            matrix_loss = criterion(positive_similar_matrix, negative_similar_matrix, (positive_lengths, negative_lengths))
            loss = matrix_loss
            if autoencoder_diff:
                loss = matrix_loss + autoencoder_diff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_meter.update(loss.item(), 1)
            tqdm_dataloader.set_postfix_str('loss = {:.4f}'.format(average_meter.avg))
            total_training_iter += 1
            # if total_training_iter % 500 == 0:
            #     return


def validate(model, dataloader, criterion):
    model.eval()
    with tqdm(dataloader) as tqdm_dataloader:
        average_meter = AverageMeter()

        all_ques_lens, all_pos_lens, all_neg_lens = [], [], []
        all_pos_alignments, all_neg_alignments = [], []

        for batch_data in tqdm_dataloader:
            tensors, weights, lengths, texts = batch_data
            positive_tensors, negative_tensors = tensors
            positive_weights, negative_weights = weights
            positive_lengths, negative_lengths = lengths
            positive_texts, negative_texts = texts
            positive_tensors, negative_tensors = positive_tensors.to(device), negative_tensors.to(device)
            positive_weights, negative_weights = positive_weights.to(device), negative_weights.to(device)
            positive_lengths, negative_lengths = positive_lengths.to(device), negative_lengths.to(device)
            batch_size = positive_tensors.size(0)
            ques_max_len = torch.LongTensor([positive_lengths[:, 0].max()]).expand(batch_size, 1)
            pos_max_len = torch.LongTensor([positive_lengths[:, 1].max()]).expand(batch_size, 1)
            neg_max_len = torch.LongTensor([negative_lengths[:, 1].max()]).expand(batch_size, 1)
            # positive_similar_matrix, negative_similar_matrix = \
            #     model(positive_tensors, positive_lengths, negative_tensors, negative_lengths)
            if (not isinstance(model, torch.nn.DataParallel) and model.use_autoencoder) or \
                    (isinstance(model, torch.nn.DataParallel) and model.module.use_autoencoder):
                positive_similar_matrix, negative_similar_matrix, autoencoder_diff = \
                    model(positive_tensors, positive_lengths, positive_weights,
                          negative_tensors, negative_lengths, negative_weights,
                          ques_max_len, pos_max_len, neg_max_len, mode='train')
            else:
                positive_similar_matrix, negative_similar_matrix = \
                    model(positive_tensors, positive_lengths, positive_weights,
                          negative_tensors, negative_lengths, negative_weights,
                          ques_max_len, pos_max_len, neg_max_len, mode='train')
                autoencoder_diff = None
            if torch.cuda.is_available():
                positive_lengths = positive_lengths.cuda()
                negative_lengths = negative_lengths.cuda()
            loss = criterion(positive_similar_matrix, negative_similar_matrix, (positive_lengths, negative_lengths))
            average_meter.update(loss.item(), 1)
            tqdm_dataloader.set_postfix_str('loss = {:.4f}'.format(average_meter.avg))

            all_ques_lens.extend(positive_lengths[:, 0].squeeze().cpu().numpy())
            all_pos_lens.extend(positive_lengths[:, 1].squeeze().cpu().numpy())
            all_neg_lens.extend(negative_lengths[:, 1].squeeze().cpu().numpy())

            all_pos_alignments.extend(positive_similar_matrix.detach().cpu().numpy())
            all_neg_alignments.extend(negative_similar_matrix.detach().cpu().numpy())

    alignments = [all_pos_alignments, all_neg_alignments]
    lengths = [all_ques_lens, all_pos_lens, all_neg_lens]

    val_examples, val_corrects, val_acc = validate_acc(alignments, lengths, n_negative)
    print(f'Validate acc = {val_acc}')
    return val_acc


def validate_acc(alignments, lengths, neg_sample_num=100):
    """ Validate accuracy: whether model can choose the positive
    sentence over other negative samples """
    pos_scores, neg_scores = [], []
    pos_alignments, neg_alignments = alignments
    src_lengths, pos_tgt_lengths, neg_tgt_lengths = lengths

    assert len(pos_alignments) == len(neg_alignments) == len(src_lengths) == len(pos_tgt_lengths) == len(neg_tgt_lengths)

    for pos_alignment, neg_alignment, src_len, pos_tgt_len, neg_tgt_len \
            in zip(pos_alignments, neg_alignments, src_lengths, pos_tgt_lengths, neg_tgt_lengths):
        # print(np.shape(pos_alignment))
        # print(src_len)
        # print(pos_tgt_len, neg_tgt_len)
        pos_score = np.sum(pos_alignment.max(0)) / src_len / pos_tgt_len
        neg_score = np.sum(neg_alignment.max(0)) / src_len / neg_tgt_len

        pos_scores.append(pos_score)
        neg_scores.append(neg_score)

    num_examples, num_corrects = 0, 0
    for i in range(0, len(pos_scores), neg_sample_num):
        one_pos_scores = pos_scores[i: i + neg_sample_num]
        one_neg_scores = neg_scores[i: i + neg_sample_num]
        num_examples += 1

        if one_pos_scores[0] > max(one_neg_scores):
            num_corrects += 1

    return num_examples, num_corrects, 1. * num_corrects / num_examples


def main():
    logger.info('********************  Spider Alignment  ********************')
    use_autoencoder = False
    table_file = 'data/spider/tables.json'
    train_data_file = 'data/spider/train_spider.json'
    dev_data_file = 'data/spider/dev.json'
    train_align_dataset = SpiderAlignDataset(table_file=table_file, data_file=train_data_file, n_negative=n_negative,
                                             negative_sampling_mode='mix')
    train_dataloader = train_align_dataset.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=8)
    dev_align_dataset = SpiderAlignDataset(table_file=table_file, data_file=dev_data_file, n_negative=n_negative,
                                           negative_sampling_mode='modify')
    dev_dataloader = dev_align_dataset.get_dataloader(batch_size=batch_size, shuffle=False, num_workers=8)
    aligner_model = BertAlignerModel(use_autoencoder=use_autoencoder)
    if os.path.exists('saved/spider/model.pt'):
        aligner_model.load_state_dict(torch.load('saved/spider/model.pt'))
    if torch.cuda.is_available():
        aligner_model = aligner_model.cuda()
    else:
        logger.warning("Model is running on CPU. The progress will be very slow.")
    criterion = aligner_model.criterion
    optimizer = aligner_model.optimizer
    # aligner_model = torch.nn.DataParallel(aligner_model)
    validate(aligner_model, dev_dataloader, criterion)
    for epoch in range(100):
        train(aligner_model, train_dataloader, criterion, optimizer)
        validate(aligner_model, dev_dataloader, criterion)
        if not os.path.exists('./saved/spider'):
            os.makedirs('./saved/spider')
        if isinstance(aligner_model, torch.nn.DataParallel):
            torch.save(aligner_model.module.state_dict(), f'saved/spider/model-{epoch}.pt')
            torch.save(aligner_model.module.state_dict(), 'saved/spider/model.pt')
        else:
            torch.save(aligner_model.state_dict(), f'saved/spider/model-{epoch}.pt')
            torch.save(aligner_model.state_dict(), 'saved/spider/model.pt')


if __name__ == '__main__':
    main()
