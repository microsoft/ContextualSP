# coding: utf-8

import json
import logging
import os
import random
from enum import Enum

import dill
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.semql_tree_util import Node as SemQLTree
from src.components.human_simulator import HumanSimulator
from src.utils.semql_converter import SpiderSemQLConverter, WikiSQLConverter
from src.utils.utils import STOP_WORD_LIST, TEMPLATE_KEYWORDS


logger = logging.getLogger(__name__)


generate_negative_tree_static = HumanSimulator.generate_negative_tree_static


class BertUtil:
    def __init__(self, shortcut='bert-base-uncased'):
        bert_pretrained_weights_shortcut = shortcut
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_weights_shortcut)

    def tokenize_sentence(self, sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        return tokenized_sentence

    def join_sentences(self, sentences):
        sentences = [sentence if isinstance(sentence, list) else self.tokenize_sentence(sentence)
                     for sentence in sentences]
        ret = ['[CLS]']
        for sentence in sentences:
            ret += sentence
            ret.append('[SEP]')
        return ret, [len(_) for _ in sentences]

    def tokens_to_ids(self, sentence_tokens):
        return self.tokenizer.convert_tokens_to_ids(sentence_tokens)


class AlignDataset(Dataset):
    def __init__(self, table_file, data_file, n_negative, dataset_name='unknown_dataset', restatement_with_tag=True,
                 negative_sampling_mode='sample'):
        self.n_negative = n_negative
        self.dataset_name = dataset_name
        self.restatement_with_table = restatement_with_tag
        if self.dataset_name == 'spider':
            self.db_id_key = 'db_id'
        elif self.dataset_name == 'wikisql':
            self.db_id_key = 'table_id'
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}")
        self.semql_converter = self.get_semql_converter()
        self.bert_util = BertUtil()
        if 'train' in data_file:
            self.split = 'train'
        elif 'dev' in data_file:
            self.split = 'dev'
        elif 'test' in data_file:
            self.split = 'test'
        else:
            self.split = 'unknown_split'
        self.db_infos, self.examples = self.load_data_file(table_file, data_file)
        self.training_pairs = self.build_training_data(negative_sampling_mode)
        logger.info(f'{self.__len__()} examples build')

    @staticmethod
    def load_data_file(db_file, data_file):
        raise NotImplementedError

    def get_semql_converter(self):
        raise NotImplementedError

    def build_training_data(self, mode):
        if os.path.exists(f'cache/{self.dataset_name}/semqls_{self.split}.bin'):
            semqls = dill.load(open(f'cache/{self.dataset_name}/semqls_{self.split}.bin', 'rb'))
        else:
            if not os.path.exists(f'cache/{self.dataset_name}'):
                os.makedirs(f'cache/{self.dataset_name}')
            semqls = [self.semql_converter.convert_example(example) for example in tqdm(self.examples)]
            logger.info(json.dumps(self.semql_converter.log))
            dill.dump(semqls, open(f'cache/{self.dataset_name}/semqls_{self.split}.bin', 'wb'))

        if os.path.exists(f'cache/{self.dataset_name}/aligner_{self.split}_data.bin'):
            restate_training_triples = dill.load(open(f'cache/{self.dataset_name}/aligner_{self.split}_data.bin', 'rb'))
        else:
            train_triples = self.build_negative_from_positive(self.db_infos, self.examples, semqls,
                                                              n_negative=self.n_negative, mode=mode)
            random.shuffle(train_triples)
            restate_training_triples = []
            for question, pos_semql_tree, neg_semql_tree in tqdm(train_triples):
                restate_training_triples.append((question,
                                                 pos_semql_tree.restatement(with_table=self.restatement_with_table),
                                                 neg_semql_tree.restatement(with_table=self.restatement_with_table)))
            dill.dump(restate_training_triples, open(f'cache/{self.dataset_name}/aligner_{self.split}_data.bin', 'wb'))
        return restate_training_triples

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        text = self.training_pairs[idx]
        question, positive_query, negative_query = text
        question = question.lower()
        positive_query = positive_query.lower()
        negative_query = negative_query.lower()
        positive_tokens, positive_lengths = self.bert_util.join_sentences([question, positive_query])
        negative_tokens, negative_lengths = self.bert_util.join_sentences([question, negative_query])
        assert len(positive_lengths) == len(negative_lengths) == 2
        assert positive_lengths[0] == negative_lengths[0]
        positive_ids = torch.LongTensor(self.bert_util.tokens_to_ids(positive_tokens))
        negative_ids = torch.LongTensor(self.bert_util.tokens_to_ids(negative_tokens))
        positive_weight_mat = self.get_weight_mask_matrix(positive_tokens, positive_lengths)
        negative_weight_mat = self.get_weight_mask_matrix(negative_tokens, negative_lengths)
        return (positive_ids, negative_ids), (positive_weight_mat, negative_weight_mat), \
               (positive_tokens, negative_tokens), (positive_lengths, negative_lengths)

    @staticmethod
    def get_weight_mask_matrix(tokens, lengths, col_stopwords=STOP_WORD_LIST, row_stopwords=TEMPLATE_KEYWORDS):
        col_tokens = tokens[1: 1 + lengths[0]]
        row_tokens = tokens[2 + lengths[0]: -1]
        weight_matrix = np.ones((lengths[0], lengths[1]))
        for idx, col_token in enumerate(col_tokens):
            if col_token in col_stopwords:
                weight_matrix[idx, :] = 0.5
        for idx, row_token in enumerate(row_tokens):
            if row_token in row_stopwords:
                weight_matrix[:, idx] = 0.5
        return weight_matrix

    def build_negative_from_positive(self, db_infos, examples, semqls, n_negative=10, mode='sample'):
        assert mode in ('sample', 'modify', 'mix')
        if mode == 'mix':
            mode = 'sample modify'
            n_negative //= 2
        ret = []
        if 'sample' in mode:
            logger.info('Sampling negative examples')
            for i in tqdm(range(len(examples))):
                question = examples[i]['question']
                positive_semql = semqls[i]
                positive_semql_tree = SemQLTree.from_statements(map(str, positive_semql))
                negative_semqls = semqls[:i] + semqls[i+1:]
                negative_samples = random.sample(negative_semqls, n_negative)
                for negative_semql in negative_samples:
                    negative_semql_tree = SemQLTree.from_statements(map(str, negative_semql))
                    ret.append((question, positive_semql_tree, negative_semql_tree))
        if 'modify' in mode:
            logger.info('Modifying negative examples')
            for i in tqdm(range(len(examples))):
                negative_examples = []
                question = examples[i]['question']
                for _ in range(n_negative):
                    positive_semql = semqls[i]
                    positive_semql_tree = SemQLTree.from_statements(map(str, positive_semql))
                    db_info = db_infos[examples[i][self.db_id_key]]
                    column_table_pairs = self.get_column_table_pairs(db_info)
                    negative_semql_tree, modified_A_nodes = \
                        generate_negative_tree_static(positive_semql_tree, column_table_pairs, mistake_prob=0.2)
                    assert positive_semql_tree.restatement() != negative_semql_tree.restatement()
                    negative_examples.append((question, positive_semql_tree, negative_semql_tree))
                ret += negative_examples
        return ret

    def get_column_table_pairs(self, db_info):
        raise NotImplementedError

    @staticmethod
    def collate_fn(data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        tensors, weights, texts, lengths = zip(*data)
        positive_tensors, negative_tensors = zip(*tensors)
        positive_weights, negative_weights = zip(*weights)
        positive_lengths, negative_lengths = zip(*lengths)
        positive_texts, negative_texts = zip(*texts)
        batch_size = len(positive_texts)

        # tensors
        positive_tensors = merge(positive_tensors)[0]
        negative_tensors = merge(negative_tensors)[0]

        # weights
        positive_matrix_shape = batch_size, max([_[0] for _ in positive_lengths]), max([_[1] for _ in positive_lengths])
        negative_matrix_shape = batch_size, max([_[0] for _ in negative_lengths]), max([_[1] for _ in negative_lengths])
        positive_weight_matrix = np.zeros(positive_matrix_shape, dtype=np.float32)
        negative_weight_matrix = np.zeros(negative_matrix_shape, dtype=np.float32)
        assert len(positive_weights) == len(negative_weights) == batch_size
        for idx, (positive_weight, negative_weight) in enumerate(zip(positive_weights, negative_weights)):
            positive_weight_matrix[idx, :positive_weight.shape[0], :positive_weight.shape[1]] = positive_weight
            negative_weight_matrix[idx, :negative_weight.shape[0], :negative_weight.shape[1]] = negative_weight
        positive_weight_matrix = torch.Tensor(positive_weight_matrix)
        negative_weight_matrix = torch.Tensor(negative_weight_matrix)

        # lengths
        positive_lengths = torch.LongTensor([[_[0] for _ in positive_lengths], [_[1] for _ in positive_lengths]]).permute(1, 0)
        negative_lengths = torch.LongTensor([[_[0] for _ in negative_lengths], [_[1] for _ in negative_lengths]]).permute(1, 0)
        # if torch.cuda.is_available():
        #     positive_lengths = positive_lengths.cuda()
        #     negative_lengths = negative_lengths.cuda()
        #     positive_tensors = positive_tensors.cuda()
        #     negative_tensors = negative_tensors.cuda()
        return (positive_tensors, negative_tensors), (positive_weight_matrix, negative_weight_matrix), \
               (positive_lengths, negative_lengths), (positive_texts, negative_texts)

    def get_dataloader(self, batch_size, num_workers=8, shuffle=True):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=self.collate_fn)


class SpiderAlignDataset(AlignDataset):
    def __init__(self, table_file, data_file, n_negative, negative_sampling_mode):
        super().__init__(table_file, data_file, n_negative, dataset_name='spider', restatement_with_tag=True,
                         negative_sampling_mode=negative_sampling_mode)

    @staticmethod
    def load_data_file(db_file, data_file):
        db_infos = {_['db_id']: _ for _ in json.load(open(db_file, 'r', encoding='utf-8'))}
        examples = json.load(open(data_file, 'r', encoding='utf-8'))
        return db_infos, examples

    def get_semql_converter(self):
        return SpiderSemQLConverter()

    def get_column_table_pairs(self, db_info):
        table_names = db_info['table_names_original']
        column_table_pairs = [('_'.join(column_name.split()).lower(), table_names[table_idx].lower())
                              for table_idx, column_name in db_info['column_names_original'][1:]]
        return column_table_pairs


class WikiSQLAlignDataset(AlignDataset):
    def __init__(self, table_file, data_file, n_negative, negative_sampling_mode):
        self.table_file = table_file
        super().__init__(table_file, data_file, n_negative, dataset_name='wikisql', restatement_with_tag=False,
                         negative_sampling_mode=negative_sampling_mode)

    @staticmethod
    def load_data_file(db_file, data_file):
        db_infos = [json.loads(line) for line in open(db_file, 'r', encoding='utf-8')]
        db_infos = {table_info['id']: table_info for table_info in db_infos}
        examples = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8')]
        return db_infos, examples

    def get_semql_converter(self):
        return WikiSQLConverter(self.table_file)

    def get_column_table_pairs(self, db_info):
        column_table_pairs = [('_'.join(column_name.split()).lower(), 'table') for column_name in db_info['header']]
        return column_table_pairs


if __name__ == '__main__':
    # # spider
    # table_file = 'data/spider/tables.json'
    # data_file = 'data/spider/train_spider.json'
    # example = json.load(open(data_file, 'r', encoding='utf-8'))[0]
    # dataset = SpiderAlignDataset(table_file, data_file, n_negative=30)
    # semql_converter = SpiderSemQLConverter()

    # wikisql
    table_file = 'data/wikisql/data/train.tables.jsonl'
    data_file = 'data/wikisql/data/train.jsonl'
    example = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8')][0]
    dataset = WikiSQLAlignDataset(table_file, data_file, n_negative=20)
    semql_converter = WikiSQLConverter(table_file)

    statements = semql_converter.convert_example(example)
    item = dataset.__getitem__(3)
    dataloader = dataset.get_dataloader(batch_size=4)
    for batch_data in dataloader:
        (positive_tensors, negative_tensors), (positive_lengths, negative_lengths), (positive_texts, negative_texts) = batch_data
