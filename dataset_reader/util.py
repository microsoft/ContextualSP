# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict
from typing import List
from typing import Tuple

import editdistance
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.fields.knowledge_graph_field import KnowledgeGraphField
from allennlp.data.tokenizers.token import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from overrides import overrides

from context.grammar import Grammar, Action
from context.world import SparcWorld

"""
Code mainly borrowed from https://github.com/benbogin/spider-schema-gnn
"""


class SparcKnowledgeGraphField(KnowledgeGraphField):
    """
    This implementation calculates all non-graph-related features (i.e. no related_column),
    then takes each one of the features to calculate related column features, by taking the max score of all neighbours
    """

    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 utterance_tokens: List[Token],
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = None,
                 bert_mode: str = "v0",
                 feature_extractors: List[str] = None,
                 entity_tokens: List[List[Token]] = None,
                 linking_features: List[List[List[float]]] = None,
                 include_in_vocab: bool = True,
                 max_table_tokens: int = None) -> None:

        if bert_mode == "v0":
            feature_extractors = feature_extractors if feature_extractors is not None else [
                # 'number_token_match',
                'exact_token_match',
                'contains_exact_token_match',
                'lemma_match',
                'contains_lemma_match',
                'edit_distance',
                'span_overlap_fraction',
                'span_lemma_overlap_fraction']
        else:
            feature_extractors = feature_extractors if feature_extractors is not None else [
                # 'number_token_match',
                'exact_token_match',
                'contains_exact_token_match',
                'lemma_match',
                'contains_lemma_match',
                'edit_distance',
                'span_overlap_fraction',
                'span_lemma_overlap_fraction']

        super().__init__(knowledge_graph, utterance_tokens, token_indexers,
                         tokenizer=tokenizer, feature_extractors=feature_extractors, entity_tokens=entity_tokens,
                         linking_features=linking_features, include_in_vocab=include_in_vocab,
                         max_table_tokens=max_table_tokens)

        self.linking_features = self._compute_related_linking_features(self.linking_features)

        # hack needed to fix calculation of feature extractors in the inherited as_tensor method
        self._feature_extractors = feature_extractors * 2

    def _compute_related_linking_features(self,
                                          non_related_features: List[List[List[float]]]) -> List[List[List[float]]]:
        linking_features = non_related_features
        entity_to_index_map = {}
        for entity_id, entity in enumerate(self.knowledge_graph.entities):
            entity_to_index_map[entity] = entity_id
        for entity_id, (entity, entity_text) in enumerate(zip(self.knowledge_graph.entities, self.entity_texts)):
            # FIXME: if [CLS] and [SEP] in entity_text, remove them for cleaning features
            for token_index, token in enumerate(self.utterance_tokens):
                entity_token_features = linking_features[entity_id][token_index]
                for feature_index, feature_extractor in enumerate(self._feature_extractors):
                    neighbour_features = []
                    for neighbor in self.knowledge_graph.neighbors[entity]:
                        # we only care about table/columns relations here, not foreign-primary
                        if entity.startswith('column') and neighbor.startswith('column'):
                            continue
                        neighbor_index = entity_to_index_map[neighbor]
                        neighbour_features.append(non_related_features[neighbor_index][token_index][feature_index])

                    entity_token_features.append(max(neighbour_features))
        return linking_features

    @overrides
    def _edit_distance(self,
                       entity: str,
                       entity_text: List[Token],
                       token: Token,
                       token_index: int,
                       tokens: List[Token]) -> float:
        entity_text = ' '.join(e.text for e in entity_text)
        edit_distance = float(editdistance.eval(entity_text, token.text))
        # normalize length
        maximum_len = max(len(entity_text), len(token.text))
        return 1.0 - edit_distance / maximum_len

    @overrides
    def empty_field(self) -> 'SparcKnowledgeGraphField':
        # TODO: HACK the error. We use utterance mask to judge whether the position is masked, not the KG field.
        return self


def index_entity_type(world: SparcWorld):
    column_type_ids = ['@@PAD@@',
                       'boolean', 'foreign', 'number', 'others', 'primary', 'text', 'time', 'string', 'table']

    # now we have 9 types
    assert len(column_type_ids) == 10

    # record the entity index
    entity_type_indices = []

    for entity_index, entity in enumerate(world.db_context.knowledge_graph.entities):
        parts = entity.split(':')
        entity_main_type = parts[0]
        if entity_main_type == 'column' or entity_main_type == 'string' or entity_main_type == 'table':
            if entity_main_type in column_type_ids:
                entity_type = column_type_ids.index(entity_main_type)
            else:
                column_type = parts[1]
                entity_type = column_type_ids.index(column_type)
        else:
            raise ConfigurationError("Get the unknown entity: {}".format(entity))
        # TODO: 0 for padding
        entity_type_indices.append(entity_type)

    return np.array(entity_type_indices)


def find_start_end(cus_list, pattern, min_start=0) -> Tuple:
    """
    Find the start & end of pattern in cus_list. If none, return 0,0.
    :param cus_list:
    :param pattern:
    :param min_start: at least from which position to match
    :return:
    """
    for i in range(len(cus_list)):
        if i < min_start:
            continue
        if cus_list[i] == pattern[0] and cus_list[i:i + len(pattern)] == pattern:
            return i, i + len(pattern)
    return 0, 0


def diff_tree(precedent_action_seq: List[Action],
              action_seq: List[Action],
              copy_rule_dict: Dict[int, Action],
              ret_tree: bool = True):
    """

    :param precedent_action_seq:
    :param action_seq:
    :param copy_rule_dict:
    :param ret_tree: if return True, return the segment-level supervision; else return the token-level supervision.
    :return:
    """
    copy_subtree_list = []
    action_seq_with_copy = []

    precedent_tree = Grammar.extract_all_subtree(precedent_action_seq)
    cur_tree = Grammar.extract_all_subtree(action_seq)

    precedent_tree_match = [False] * len(precedent_tree)
    cur_tree_match = [False] * len(cur_tree)

    action_seq_match = [False] * len(action_seq)
    precedent_action_seq_match = [False] * len(precedent_action_seq)

    for pre_ind in range(len(precedent_tree)):
        # we will change precedent_tree in the process
        pre_sub_tree = precedent_tree[pre_ind]
        # has matched, continue
        if precedent_tree_match[pre_ind]:
            continue
        for cur_ind in range(len(cur_tree)):
            cur_sub_tree = cur_tree[cur_ind]
            # has matched, continue
            if cur_tree_match[cur_ind]:
                continue
            if str(pre_sub_tree) == str(cur_sub_tree):
                # find cur_sub_tree start/end in action_seq, and its corresponding str
                cur_start = 0
                pre_start = 0
                while True:
                    cur_start, cur_end = find_start_end(action_seq, cur_sub_tree, min_start=cur_start)
                    pre_start, pre_end = find_start_end(precedent_action_seq, cur_sub_tree, min_start=pre_start)

                    pre_used = True in precedent_action_seq_match[pre_start: pre_end]
                    cur_used = True in action_seq_match[cur_start: cur_end]
                    if not pre_used and not cur_used:
                        break
                    elif pre_used and cur_used:
                        pre_start += 1
                        cur_start += 1
                    elif pre_used:
                        pre_start += 1
                    elif cur_used:
                        cur_start += 1

                if cur_end != 0 and pre_end != 0:
                    # record the precedent copy index
                    copy_subtree_list.append((cur_start, cur_end, pre_ind, pre_start, pre_end))
                    # make all the subtrees marked as True
                    for ind in range(cur_start, cur_end):
                        action_seq_match[ind] = True
                    for ind in range(pre_start, pre_end):
                        precedent_action_seq_match[ind] = True

                    # mark the pre_ind and fol_ind as True
                    precedent_tree_match[pre_ind] = True
                    cur_tree_match[cur_ind] = True

    # sort copy_subtree_list via start idx
    copy_subtree_list = sorted(copy_subtree_list, key=lambda x: x[0])

    ind = 0
    copy_pointer = 0
    while ind < len(action_seq):
        if action_seq_match[ind] is False:
            # add original action
            action_seq_with_copy.append(action_seq[ind])
            ind += 1
        else:
            cur_start, cur_end, pre_ind, pre_start, pre_end = copy_subtree_list[copy_pointer]
            assert cur_start == ind
            if ret_tree:
                action_seq_with_copy.append(copy_rule_dict[pre_ind])
            else:
                for i in range(pre_start, pre_end):
                    action_seq_with_copy.append(copy_rule_dict[i])
            ind = cur_end
            copy_pointer += 1

    return action_seq_with_copy
