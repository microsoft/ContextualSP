# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random
import sys
import traceback
from typing import List, Dict, Iterable, Optional

import dill
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data import Token
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField, ArrayField
from context.copy_production_rule_field import CopyProductionRuleField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer, Tokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, JustSpacesWordSplitter
from overrides import overrides
from spacy.symbols import ORTH, LEMMA
import multiprocessing as mp
import string
from context.db_context import SparcDBContext
from context.grammar import Grammar
from context.world import SparcWorld
from .util import SparcKnowledgeGraphField, index_entity_type
from constant import *
from .reader_queue import QIterable
import pickle
from tqdm import tqdm
from dataset_reader.util import diff_tree


@DatasetReader.register('sparc')
class SparcDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 utterance_tokenizer: Tokenizer = None,
                 utterance_token_indexers: Dict[str, TokenIndexer] = None,
                 # none, dis, concat
                 context_mode: str = ContextMode.context_independent,
                 copy_mode: str = CopyMode.no_copy,
                 # none, token, segment
                 # copy_mode: str = CopyMode.no_copy,
                 bert_mode: str = "v0",
                 num_workers: int = 1,
                 tables_file: str = None,
                 database_path: str = 'dataset\\database',
                 cache_method: str = 'pickle',
                 cache_mode: str = 'all',
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_limit: int = -1,
                 # utilize how many context
                 maximum_history_len: int = 5,
                 memory_friend: bool = False):
        super().__init__(lazy=lazy)

        # we use spacy tokenizer as the default tokenizer
        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_splitter = SpacyWordSplitter(pos_tags=True)
        spacy_splitter.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])

        self._tokenizer = utterance_tokenizer or WordTokenizer(spacy_splitter)
        self._indexer = utterance_token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}

        # space tokenizer is used for nonterminal tokenize
        self._non_terminal_tokenizer = WordTokenizer(JustSpacesWordSplitter())
        self._non_terminal_indexer = {'nonterminals': SingleIdTokenIndexer(namespace='nonterminals')}

        self._table_file = tables_file
        self._database_path = database_path
        self._loading_limit = loading_limit

        self._load_cache = load_cache
        self._save_cache = save_cache

        # determine the context mode
        self._context_mode = context_mode

        self._copy_mode = copy_mode

        # v0: no bert
        # v3: encode utterance with table jointly, add [CLS] and [SEP]
        self._bert_mode = bert_mode

        # we do not care the problem of maximum length because allen nlp has helped us to do this
        self._maximum_seq_len = np.inf

        self._number_workers = num_workers
        # dill, json
        self._cache_method = cache_method
        # overall, single
        self._cache_mode = cache_mode
        # maximum_context
        self._maximum_history_len = maximum_history_len

        # if enable memory friend, use sentence rather than interaction as a basic unit of batch
        self._memory_friend = memory_friend

        if memory_friend:
            assert self._context_mode == "concat", "We only support to use less memory in concat mode since others" \
                                                   "depend on dynamic context such as context representations" \
                                                   "and generated SQLs, which assumes batching on interactions."
            assert self._copy_mode == "none", "We only support to use less memory in concat mode since others" \
                                              "depend on dynamic context such as context representations" \
                                              "and generated SQLs, which assumes batching on interactions."

    def build_instance(self, parameter) -> Iterable[Instance]:
        # loading some examples
        # if self._loading_limit == total_cnt:
        #     break
        total_cnt, inter_ex = parameter
        extension = 'bin' if self._cache_method == CacheMethod.dil else 'pkl'
        cache_file = os.path.join(self._cache_dir, f'ins-{total_cnt}.{extension}')

        if self._load_cache and os.path.exists(cache_file) and self._cache_mode == CacheMode.single:
            if self._cache_method == CacheMethod.dil:
                ins = dill.load(open(cache_file, 'rb'))
            elif self._cache_method == CacheMethod.pick:
                ins = pickle.load(open(cache_file, 'rb'))
            else:
                raise ConfigurationError("Not such cache method!")
            # None passing.
            if ins is not None:
                # print(max([len(action.field_list) for action in ins.fields['action_sequence']]))
                return ins

        db_id = inter_ex['database_id']
        inter_utter_list = [ex['utterance'] for ex in inter_ex['interaction']]
        # use structural sql instead of plain tokens
        sql_list = [ex['sql'] for ex in inter_ex['interaction']]
        sql_query_list = [ex['query'] for ex in inter_ex['interaction']]

        # TODO: now one interaction composes a instance, we should do a more careful design
        try:
            ins = self.text_to_instance(
                utter_list=inter_utter_list,
                db_id=db_id,
                sql_list=sql_list,
                sql_query_list=sql_query_list
            )

            if self._save_cache and self._cache_mode == CacheMode.single:
                # save cache into file
                if self._cache_method == CacheMethod.dil:
                    dill.dump(ins, open(cache_file, 'wb'))
                elif self._cache_method == CacheMethod.pick:
                    pickle.dump(ins, open(cache_file, 'wb'))

            if ins is not None:
                return ins
        except Exception as e:
            print(f'Error in db_id: {db_id}, utterance: {inter_utter_list}')
            exec_info = sys.exc_info()
            traceback.print_exception(*exec_info)

    @staticmethod
    def _dill_load(file_path):
        return dill.load(open(file_path, "rb"))

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        if not file_path.endswith(".json"):
            raise ConfigurationError(f"The file path is not designed for SParC dataset {file_path}")

        self._cache_dir = os.path.join('cache',
                                       "_".join(file_path.split("\\")[-2:]) +
                                       "_con[" + self._context_mode +
                                       "]_bert[" + str(self._bert_mode) +
                                       "]_cache[" + str(self._cache_mode) + "]")

        if self._copy_mode != CopyMode.no_copy:
            self._cache_dir += "_copy[" + str(self._copy_mode) + "]"

        if self._memory_friend:
            self._cache_dir += "_memory[true]"

        extension = 'bin' if self._cache_method == CacheMethod.dil else 'pkl'
        cache_all_file = os.path.join(self._cache_dir, f"cache_all.{extension}")

        if self._load_cache:
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
            elif self._cache_mode == CacheMode.all and os.path.exists(cache_all_file):
                # read without multiple instance
                instances = [ins for ins in pickle.load(open(cache_all_file, "rb")) if ins]
                return instances
            elif self._cache_mode == CacheMode.single and self._number_workers > 1:
                instances = []
                for ins in QIterable(output_queue_size=400,
                                     epochs_per_read=1,
                                     num_workers=self._number_workers,
                                     call_back=SparcDatasetReader._dill_load,
                                     file_path="{}\\ins-*.bin".format(self._cache_dir)):
                    if ins: instances.append(ins)
                return instances

        with open(file_path, "r", encoding="utf8") as data_file:
            json_obj = json.load(data_file)
            # list of interactions
            assert isinstance(json_obj, list)
            if self._cache_mode == CacheMode.all:
                # write cache here
                instances = []
                # FIXME: we do not use multiprocessing in caching all
                for json_ins in tqdm(enumerate(json_obj)):
                    ins = self.build_instance(json_ins)
                    if isinstance(ins, List):
                        instances.extend(ins)
                    else:
                        instances.append(ins)

                if self._save_cache:
                    with open(cache_all_file, 'wb') as cache:  # Use highest protocol for speed.
                        if self._cache_method == CacheMethod.pick:
                            pickle.dump(instances, cache, protocol=pickle.HIGHEST_PROTOCOL)
                        elif self._cache_method == CacheMethod.dil:
                            dill.dump(instances, cache)
                return instances
            else:
                instances = []
                # write cache inside build_instance
                if self._number_workers > 1:
                    pool = mp.Pool(processes=self._number_workers)
                    for ins in tqdm(pool.imap(self.build_instance, enumerate(json_obj))):
                        if ins: instances.append(ins)
                else:
                    for json_ins in tqdm(enumerate(json_obj)):
                        ins = self.build_instance(json_ins)
                        if ins: instances.append(ins)
                return instances

    @overrides
    def text_to_instance(self,
                         utter_list: List[str],
                         db_id: str,
                         sql_list: Optional[List[Dict]] = None,
                         sql_query_list: Optional[List[Dict]] = None) -> Optional[Instance]:

        # return invalid instances
        if len(utter_list) == 0:
            return None

        entity_mask_fields = []

        for ind, utter in enumerate(utter_list):
            if utter[-1] not in string.punctuation:
                utter_list[ind] += ' ' + random.choice(['.', '?'])

        cached_global_utterance = utter_list[:]

        cached_db_contexts = []
        # expand all possible entities
        for i in range(len(cached_global_utterance)):
            tokenized_utterance = self._tokenizer.tokenize(" ".join(
                cached_global_utterance[i - self._maximum_history_len: i + 1]).lower())
            tokenized_utterance = [Token(text=t.text, lemma_=t.lemma_) if t.lemma_ != '-PRON-'
                                   else Token(text=t.text, lemma_=t.text) for t in tokenized_utterance]

            # unify the entity number for different utterances
            temp_db_context = SparcDBContext(db_id=db_id,
                                             utterance=tokenized_utterance,
                                             tokenizer=self._tokenizer,
                                             tables_file=self._table_file,
                                             database_path=self._database_path,
                                             bert_mode=self._bert_mode)

            cur_entity_num = len(temp_db_context.knowledge_graph.entities)
            cached_db_contexts.append(temp_db_context)
            entity_mask_fields.append(ArrayField(np.array([1] * cur_entity_num)))
            # from the first to the last

        fields: Dict[str, Field] = {}
        # world list
        world_fields: List[Field] = []
        # knowledge list
        knowledge_fields: List[Field] = []

        # record all utterances
        utter_fields: List[Field] = []

        # record all segments
        segment_fields: List[Field] = []
        # record all nonterminals
        nonterminal_fields: List[Field] = []
        # record all valid actions
        valid_rules_fields: List[ListField] = []
        # record the action sequence index
        index_fields: List[ListField] = []
        # record the action sequence (mixed with copy operation) index
        index_with_copy_fields: List[ListField] = []
        # record the entity type index
        entity_type_fields: List[Field] = []
        # sql_clauses metadata

        schema_position_fields: List[Field] = []

        past_utters = []
        past_history_len = []
        # TODO: record all segment ids, which is supposed no larger than 5
        past_segments = []

        # if sql_list is None, use default sql list to fill in it to avoid fault
        if sql_list is None:
            new_sql_list = [Grammar.default_sql_clause() for _ in range(len(utter_list))]
        else:
            new_sql_list = sql_list

        if sql_query_list is None:
            new_sql_query_list = ['' for _ in range(len(utter_list))]
        else:
            new_sql_query_list = sql_query_list

        utter_ind = 1

        use_hard_token_as_segment = self._context_mode in [ContextMode.copy_hard_token,
                                                           ContextMode.concat_hard_token]

        # record precedent action sequence, used in copy (either segment-level or token-level)
        precedent_action_seq = None

        index = 0

        # allocate memory instances
        memory_instances = []

        for fol_utter, sql_clause, sql_query in zip(utter_list, new_sql_list, new_sql_query_list):

            # tokenize history and so on
            tokenized_utter = self._tokenizer.tokenize(fol_utter.lower())

            # the extra END means the end of prediction on latent
            tokenized_utter = [Token(text=t.text, lemma_=t.lemma_) if t.lemma_ != '-PRON-'
                               else Token(text=t.text, lemma_=t.text) for t in tokenized_utter]

            # TODO: cur_segment is only used for scenarios where not joint training.
            cur_segment = [utter_ind] * len(tokenized_utter)

            # TODO: default we use tokens with context as data

            if self._context_mode in [ContextMode.context_independent,
                                      ContextMode.turn_model,
                                      ContextMode.copy_hard_token]:
                tokenized_utterance = tokenized_utter
                segment_ids = np.array(cur_segment)
            elif self._context_mode in [ContextMode.concat_previous,
                                        ContextMode.concat_history,
                                        ContextMode.concat_hard_token]:
                tokenized_utterance = past_utters + tokenized_utter
                segment_ids = np.array(past_segments + cur_segment)
            else:
                raise Exception("Not support for mode :{}".format(self._context_mode))

            if self._context_mode == ContextMode.concat_previous:
                # update past utterance, ignore the last element(which is the @END@ symbol)
                past_utters = tokenized_utter
                # add segments, start from 0 (no padding)
                past_segments = cur_segment
                past_history_len = [len(cur_segment)]
            elif self._context_mode in [ContextMode.concat_history, ContextMode.concat_hard_token]:
                # update past utterance, ignore the last element(which is the @END@ symbol)
                past_utters.extend(tokenized_utter)
                # add segments, start from 0 (no padding)
                past_segments.extend(cur_segment)
                past_history_len.append(len(cur_segment))
            else:
                past_utters = []
                past_segments = []

            db_context = cached_db_contexts[index]
            assert len(past_segments) == len(past_utters)

            table_field = SparcKnowledgeGraphField(db_context.knowledge_graph,
                                                   tokenized_utterance,
                                                   self._indexer,
                                                   bert_mode=db_context.bert_mode,
                                                   entity_tokens=db_context.entity_tokens,
                                                   include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                                   max_table_tokens=None)  # self._max_table_tokens)

            if self._bert_mode == "v3":
                # we prepare [SEP] before each entity text, and use the indexer to identify them automatically
                # here we concat the utterance with database schemas together, and feed it into the BERT model.
                schema_position = []
                schema_tokens = []
                utterance_len = len(tokenized_utterance)
                for ind, entity_id in enumerate(db_context.knowledge_graph.entities):
                    entity_tokens = db_context.entity_tokens[ind]
                    # Utterance [SEP] Col1 [SEP] Col2 [SEP]
                    schema_tokens.extend([Token(text="[SEP]")])
                    schema_start = utterance_len + len(schema_tokens)
                    # currently we only add Table name and Col name
                    if entity_id.startswith("column"):
                        # add column
                        schema_tokens.extend(entity_tokens)
                        schema_end = utterance_len + len(schema_tokens)
                    elif entity_id.startswith("table"):
                        # add table
                        schema_tokens.extend(entity_tokens)
                        schema_end = utterance_len + len(schema_tokens)
                    else:
                        raise Exception("Currently we do not support encoding for other entities!")
                    schema_position.append([schema_start, schema_end])

                tokenized_utterance = tokenized_utterance + schema_tokens
                schema_position_fields.append(ArrayField(np.array(schema_position, dtype=np.int)))

            # build world
            world = SparcWorld(db_context=db_context,
                               sql_clause=sql_clause,
                               sql_query=sql_query)

            entity_type_fields.append(ArrayField(index_entity_type(world=world)))
            action_non_terminal, action_seq, all_valid_actions = world.get_action_sequence_and_all_actions()

            # the update precedent sql must be executed after the get_action_sequence_and_all!
            if precedent_action_seq is not None:
                # assign precedent subtrees
                world.update_precedent_state(precedent_action_seq,
                                             extract_tree=not use_hard_token_as_segment)
                # make precedent action under the nonterminal rules
                world.update_copy_valid_action()

            assert action_seq is not None
            for action_rule in action_seq:
                assert action_rule in all_valid_actions

            # append utterance into utter field
            utter_fields.append(TextField(tokenized_utterance, self._indexer))
            segment_fields.append(ArrayField(segment_ids))

            # tokenize non terminal
            nonterminal_utter = ' '.join(action_non_terminal)
            tokenized_nonterminal = self._non_terminal_tokenizer.tokenize(nonterminal_utter)
            nonterminal_fields.append(TextField(tokenized_nonterminal, self._non_terminal_indexer))

            # allocate new product file field
            temp_rule_fields: List[CopyProductionRuleField] = []
            temp_index_fields: List[IndexField] = []

            for prod_rule in all_valid_actions:
                # get rule's nonterminal name
                nonterminal = prod_rule.nonterminal
                field = CopyProductionRuleField(rule=str(prod_rule),
                                                is_global_rule=prod_rule.is_global(),
                                                is_copy_rule=False,
                                                nonterminal=nonterminal)
                temp_rule_fields.append(field)

            single_rule_field = ListField(temp_rule_fields)

            # add action sequence into list
            action_map = {action.rule: i  # type: ignore
                          for i, action in enumerate(single_rule_field.field_list)}

            for prod_rule in action_seq:
                field = IndexField(index=action_map[str(prod_rule)],
                                   sequence_field=single_rule_field)
                temp_index_fields.append(field)

            single_index_field = ListField(temp_index_fields)

            # update copy actions
            if precedent_action_seq is not None:
                # index copy rule
                copy_rule_fields: List[CopyProductionRuleField] = []
                copy_rule_dict = {}

                for local_ind, prod_rule in enumerate(world.precedent_segment_seq):
                    # get nonterminal name
                    nonterminal = prod_rule.nonterminal
                    rule_repr = str(prod_rule)
                    field = CopyProductionRuleField(rule=rule_repr,
                                                    # the copy rule is appended dynamically
                                                    is_global_rule=False,
                                                    is_copy_rule=True,
                                                    nonterminal=nonterminal)
                    copy_rule_fields.append(field)
                    copy_rule_dict[local_ind] = prod_rule
                    # add rule into action_map
                    copy_rule_idx = len(action_map)
                    action_map[rule_repr] = copy_rule_idx

                # use diff to find
                # TODO: we do not use simplediff to avoid the following scenarios:
                # Precedent: ... T -> department, Order -> des A limit, ...
                # Follow: ... T -> department, Order -> des A limit, ...
                # the same sub-sequence will be matched, but it is NOT a subtree!
                action_seq_with_copy = diff_tree(precedent_action_seq,
                                                 action_seq,
                                                 copy_rule_dict,
                                                 ret_tree=not use_hard_token_as_segment)
                # merge copy rule fields with temp rule fields
                temp_rule_fields.extend(copy_rule_fields)

                # update single rule_field
                single_rule_field = ListField(temp_rule_fields)

                temp_index_with_copy_fields: List[IndexField] = []

                for prod_rule in action_seq_with_copy:
                    field = IndexField(index=action_map[str(prod_rule)],
                                       sequence_field=single_rule_field)
                    temp_index_with_copy_fields.append(field)

                single_index_with_copy_field = ListField(temp_index_with_copy_fields)
                index_with_copy_fields.append(single_index_with_copy_field)
            else:
                index_with_copy_fields.append(single_index_field)

            # record into the instance-level fields
            valid_rules_fields.append(single_rule_field)
            index_fields.append(single_index_field)
            world_fields.append(MetadataField(world))
            knowledge_fields.append(table_field)

            # assign state
            utter_ind += 1
            precedent_action_seq = action_seq

            if len(past_history_len) >= self._maximum_history_len:
                pop_seq_len = past_history_len.pop(0)
                past_segments = past_segments[pop_seq_len:]
                past_utters = past_utters[pop_seq_len:]
                past_segments = [segment_id - 1 for segment_id in past_segments]

            # yield multiple instances by sentences
            if self._memory_friend:
                if self._bert_mode == "v3":
                    fields['schema_position'] = ListField(schema_position_fields)
                fields['inter_utterance'] = ListField(utter_fields)
                fields['inter_schema'] = ListField(knowledge_fields)
                fields['inter_nonterminal'] = ListField(nonterminal_fields)
                fields['inter_segment'] = ListField(segment_fields)
                fields['valid_actions_list'] = ListField(valid_rules_fields)
                fields['action_sequence'] = ListField(index_fields)
                fields['action_sequence_with_copy'] = ListField(index_with_copy_fields)
                fields['worlds'] = ListField(world_fields)
                fields['entity_type'] = ListField(entity_type_fields)
                # clear fields
                schema_position_fields = []
                utter_fields = []
                knowledge_fields = []
                nonterminal_fields = []
                segment_fields = []
                valid_rules_fields = []
                index_fields = []
                index_with_copy_fields = []
                world_fields = []
                entity_type_fields = []
                # entity mask is prepared already
                fields['entity_mask'] = ListField([entity_mask_fields[index]])
                memory_instances.append(Instance(fields))

            # update index
            index += 1

        if self._memory_friend:
            return memory_instances

        if self._bert_mode == "v3":
            fields['schema_position'] = ListField(schema_position_fields)

        # all utterances in one interaction
        fields['inter_utterance'] = ListField(utter_fields)
        fields['inter_schema'] = ListField(knowledge_fields)
        fields['inter_nonterminal'] = ListField(nonterminal_fields)
        fields['inter_segment'] = ListField(segment_fields)
        fields['valid_actions_list'] = ListField(valid_rules_fields)
        fields['action_sequence'] = ListField(index_fields)
        fields['action_sequence_with_copy'] = ListField(index_with_copy_fields)
        fields['worlds'] = ListField(world_fields)
        fields['entity_type'] = ListField(entity_type_fields)
        fields['entity_mask'] = ListField(entity_mask_fields)

        # return the instance
        return Instance(fields)
