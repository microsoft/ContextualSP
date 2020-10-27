# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Author: Qian Liu (SivilTaram)
# Original Repo: https://github.com/microsoft/ContextualSP

import os
import pickle
import re
import sys
import traceback
from typing import List, Iterable

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.data import Token
from allennlp.data.fields import TextField, MetadataField, ArrayField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides
from tqdm import tqdm

from data_utils import SpecialSymbol
from data_utils import export_word_edit_matrix, get_class_mapping


@DatasetReader.register('rewrite')
class RewriteDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_ratio: float = 1.0,
                 enable_unparse: bool = True,
                 use_bert: bool = False,
                 # insert before/after/both
                 super_mode: str = 'before',
                 word_level: bool = False,
                 # if joint encoding, means that
                 joint_encoding: bool = False,
                 language: str = 'zh',
                 extra_stop_words: List = None):
        super().__init__(lazy=lazy)

        # different languages share the same word splitter
        self._tokenizer = WordTokenizer(JustSpacesWordSplitter())

        if use_bert:
            pretrain_model_name = 'bert-base-chinese' if language == 'zh' else 'bert-base-uncased'
            self._indexer = {'bert': PretrainedBertIndexer(pretrained_model=pretrain_model_name,
                                                           use_starting_offsets=False,
                                                           do_lowercase=True,
                                                           never_lowercase=['[UNK]', '[PAD]', '[CLS]', '[SEP]'],
                                                           truncate_long_sequences=False)}
        else:
            self._indexer = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        # loading ratio is designed for hyper-parameter fine-tuning
        self._loading_ratio = loading_ratio
        self._load_cache = load_cache
        self._save_cache = save_cache

        self._use_bert = use_bert
        self._language = language
        self._word_level = word_level
        self._super_mode = super_mode
        self._enable_unparse = enable_unparse
        self._joint_encoding = joint_encoding
        self._extra_stop_words = extra_stop_words

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        if not file_path.endswith('.txt'):
            raise ConfigurationError(
                f'The file path is not designed for Rewrite dataset {file_path}'
            )
        self._cache_dir = os.path.join('../cache', '_'.join(os.path.normpath(file_path).split(os.sep)[-2:]))
        # add a suffix
        if self._joint_encoding:
            self._cache_dir += '.joint'

        if self._use_bert:
            self._cache_dir += '.bert'

        extension = 'pkl'
        cache_all_file = os.path.join(self._cache_dir, f'cache_all.{extension}')
        if self._load_cache:
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
            elif os.path.exists(cache_all_file):
                instances = [ins for ins in pickle.load(open(cache_all_file,
                                                             'rb')) if ins]
                return instances
        with open(file_path, 'r', encoding='utf8') as data_file:
            lines = data_file.readlines()
            instances: List[Instance] = []
            loading_limit = len(lines) * self._loading_ratio if 'train' in file_path else len(lines)
            counter = 0
            for total_cnt, line in tqdm(enumerate(lines)):
                utterances = line.strip().split('\t\t')
                context_utt = utterances[:-2]
                cur_utt = utterances[-2]
                restate_utt = utterances[-1]

                try:
                    ins = self.text_to_instance(context_utt,
                                                cur_utt,
                                                restate_utt,
                                                # TODO: ad-hoc solution
                                                'train' in file_path)
                except Exception or OSError:
                    print(f'Error in line: {total_cnt}')
                    exec_info = sys.exc_info()
                    traceback.print_exception(*exec_info)

                if ins:
                    instances.append(ins)
                    counter += 1
                if counter >= loading_limit:
                    break
            if self._save_cache:
                with open(cache_all_file, 'wb') as cache:
                    pickle.dump(instances, cache, protocol=pickle.
                                HIGHEST_PROTOCOL)
            return instances

    @overrides
    def text_to_instance(self, context_utt: List[str],
                         cur_utt: str,
                         restate_utt: str,
                         training: bool) -> Instance:
        # if extra words, append it into context utterance
        if self._extra_stop_words is not None and len(self._extra_stop_words) > 0:
            context_utt = [' '.join(self._extra_stop_words)] + context_utt

        context_utt = re.sub('\\s+', ' ', ' '.join([sen.lower() + ' ' + SpecialSymbol.context_internal
                                                    for sen in context_utt]))
        cur_utt = re.sub('\\s+', ' ', cur_utt.lower())
        restate_utt = re.sub('\\s+', ' ', restate_utt.lower())
        fields = {}

        tokenized_context = self._tokenizer.tokenize(context_utt)
        tokenized_context = [Token(text=t.text, lemma_=t.lemma_) for t in
                             tokenized_context]

        tokenized_cur = self._tokenizer.tokenize(cur_utt.lower())
        # always the placeholder to build the border.
        tokenized_cur = [Token(text=t.text, lemma_=t.lemma_) for t in tokenized_cur] + \
                        [Token(text=SpecialSymbol.end_placeholder, lemma_=SpecialSymbol.end_placeholder)]

        tokenized_restate = self._tokenizer.tokenize(restate_utt)
        tokenized_restate = [Token(text=t.text, lemma_=t.lemma_) for t in tokenized_restate]

        if self._joint_encoding:
            tokenized_joint = tokenized_context + tokenized_cur
            fields['joint_tokens'] = TextField(tokenized_joint, self._indexer)
            fields['joint_border'] = IndexField(index=len(tokenized_context),
                                                sequence_field=fields['joint_tokens'])
        else:
            fields['context_tokens'] = TextField(tokenized_context, self._indexer)
            fields['cur_tokens'] = TextField(tokenized_cur, self._indexer)

        if self._extra_stop_words:
            # maybe not reasonable
            attn_operations = export_word_edit_matrix(tokenized_context,
                                                  tokenized_cur[:-1],
                                                      tokenized_restate,
                                                      self._super_mode,
                                                      only_one_insert=False)
        else:
            attn_operations = export_word_edit_matrix(tokenized_context,
                                                  tokenized_cur[:-1],
                                                      tokenized_restate,
                                                      self._super_mode,
                                                      only_one_insert=True)

        matrix_map = np.zeros((len(tokenized_context), len(tokenized_cur)),
                              dtype=np.long)

        class_mapping = get_class_mapping(super_mode=self._super_mode)

        # build distant supervision
        if training:
            keys = [op_tuple[0] for op_tuple in attn_operations]
            if not self._enable_unparse and 'remove' in keys:
                # the training supervision may not be accurate
                print("Invalid Case")
            else:
                for op_tuple in attn_operations:
                    op_name = op_tuple[0]

                    if op_name == 'remove':
                        continue

                    assert op_name in class_mapping.keys()
                    label_value = class_mapping[op_name]

                    cur_start, cur_end = op_tuple[1]
                    con_start, con_end = op_tuple[2]
                    if op_name == 'replace':
                        matrix_map[con_start:con_end, cur_start:cur_end] = label_value
                    else:
                        assert cur_start == cur_end
                        matrix_map[con_start:con_end, cur_start] = label_value

        fields['matrix_map'] = ArrayField(matrix_map, padding_value=-1)
        fields['context_str'] = MetadataField(context_utt)
        fields['cur_str'] = MetadataField(cur_utt)
        fields['restate_str'] = MetadataField(restate_utt)

        return Instance(fields)
