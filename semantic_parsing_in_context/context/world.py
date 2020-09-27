# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Tuple, Dict

from allennlp.common.util import pad_sequence_to_length

from context.converter import SQLConverter
from context.db_context import SparcDBContext
from context.grammar import Grammar, Action, C, T, Segment


class SparcWorld:
    """
    World representation for spider dataset.
    """

    def __init__(self, db_context: SparcDBContext, sql_clause, sql_query):
        """
        :param sql_clause: structural SQL clause(parsed)
        :param sql_query: plain SQL query for evaluation
        """
        self.db_id = db_context.db_id
        self.db_context = db_context
        self.sql_clause = sql_clause
        self.sql_query = sql_query
        self.sql_converter = SQLConverter(db_context=self.db_context)
        # keep a list of entities names as they are given in sql queries
        self.entities_indexer = {}
        for i, entity in enumerate(self.db_context.knowledge_graph.entities):
            parts = entity.split(':')
            if parts[0] in ['table', 'string']:
                self.entities_indexer[parts[1]] = i
            else:
                # TODO: here we assume the same column name always map into the same text
                _, _, column_name = parts
                self.entities_indexer[f'{column_name}'] = i

        self.valid_actions: Dict[str, List[str]] = {}
        self.valid_actions_flat: List[Action] = []

        # to support precedent SQL query copy in token-level or segment-level
        # this attribute will be assigned in runtime.
        self.precedent_action_seq: List[int] = []
        # the action is exactly Segment Action
        self.precedent_segment_seq: List[Segment] = []

    def update_copy_valid_action(self):
        """
        For grammar-based decoding method, the copy action will also be constrained under the nonterminal.
        Therefore, we should update the valid_actions for considering the copyed action
        :return:
        """
        for action in self.precedent_segment_seq:
            action_key = action.nonterminal
            if action_key not in self.valid_actions:
                self.valid_actions[action_key] = []
            # record action
            self.valid_actions[action_key].append(str(action))
            self.valid_actions_flat.append(action)

    def clear_precedent_state(self, copy_action_ids):
        # clear all precedent state
        for action in self.precedent_segment_seq:
            action_key = action.nonterminal
            if action_key in self.valid_actions and str(action) in self.valid_actions[action_key]:
                self.valid_actions[action_key].remove(str(action))

        copy_action_ids = sorted(copy_action_ids, reverse=True)
        for action_idx in copy_action_ids:
            del self.valid_actions_flat[action_idx]

        self.precedent_action_seq = []
        self.precedent_segment_seq = []

    def update_precedent_state(self, precedent_sql_query, extract_tree=True):
        """
        Receiving string input (in training), or idx input (in testing), convert them into action sequence.
        Furthermore, build it as a parsing tree.
        **Note this function must be called after `get_action_sequence_and_all_actions` ! **
        :param precedent_sql_query: `Dict` or `List[int]`, `Dict` is used in pre-processing,
        `List[int]` is used in real-time testing.
        :return:
        """
        def sub_finder(cus_list, pattern):
            indices = []
            for i in range(len(cus_list)):
                if cus_list[i] == pattern[0] and cus_list[i:i + len(pattern)] == pattern:
                    indices.append((i, i + len(pattern)))
            return indices

        # translate string sql query into action sequence
        if isinstance(precedent_sql_query, Dict):
            precedent_action_seq = self.sql_converter.translate_to_intermediate(precedent_sql_query)
        elif isinstance(precedent_sql_query, List):
            if isinstance(precedent_sql_query[0], Action):
                # convert idx into action string
                precedent_action_seq = precedent_sql_query
            elif isinstance(precedent_sql_query[0], int):
                precedent_action_seq = [self.valid_actions_flat[ind]
                                        for ind in precedent_sql_query]
            else:
                raise Exception("No support for input format for precedent_sql_query")
        else:
            precedent_action_seq = []

        # Type: List[int]
        self.precedent_action_seq = [self.valid_actions_flat.index(action)
                                     for action in precedent_action_seq]
        # build AST tree
        if extract_tree:
            precedent_tree_action = Grammar.extract_all_subtree(precedent_action_seq)
        else:
            precedent_tree_action = [[action] for action in precedent_action_seq]
        # we should convert the action into ids as `text_to_instance` do
        precedent_tree_idx = [[self.valid_actions_flat.index(action) for action in action_seq]
                              for action_seq in precedent_tree_action]
        # default action ind is -1
        max_len = max([len(sub_tree) for sub_tree in precedent_tree_idx])
        precedent_tree_idx = [pad_sequence_to_length(sub_tree, default_value=lambda: -1, desired_length=max_len)
                              for sub_tree in precedent_tree_idx]

        # add to self's action
        self.precedent_segment_seq = []
        for tree_str, tree_idx in zip(precedent_tree_action, precedent_tree_idx):
            self.precedent_segment_seq.append(Segment(tree_str, tree_idx))

    def get_action_sequence_and_all_actions(self) -> Tuple[List[str], List[Action], List[Action]]:
        """
        Translate the sql clause when initialization into action sequence corresponding to their SemQL.
        And return the instantiated local grammars and global grammars.
        :return: action sequence corresponding to the sql clause, all valid actions(which has been sorted)
        """
        # build global grammar and local grammar
        grammar = Grammar(db_context=self.db_context)

        global_grammar = grammar.global_grammar
        local_grammar = grammar.local_grammar
        all_actions = global_grammar + local_grammar

        # the sorted actions must follow the same order and
        # global grammar will be converted into tensor automatically in allennlp
        self.valid_actions_flat = [action for action in all_actions]

        # add every action into nonterminal key
        for action in self.valid_actions_flat:
            action_key = action.nonterminal
            if action_key not in self.valid_actions:
                self.valid_actions[action_key] = []
            # record action
            self.valid_actions[action_key].append(str(action))

        if self.sql_clause is not None:
            action_sequence = self.sql_converter.translate_to_intermediate(self.sql_clause)
            # validate action sequence
        else:
            action_sequence = None

        # fetch action_non_terminal
        action_non_terminal = None
        if action_sequence is not None:
            action_non_terminal = [action.__class__.__name__ for action in action_sequence]

        return action_non_terminal, action_sequence, all_actions

    def get_action_entity_mapping(self) -> Dict[str, int]:
        """
        Get the entity index of every local grammar(also named after linked action)
        :return:
        """
        mapping = {}

        for action in self.valid_actions_flat:
            # default is padding
            mapping[str(action)] = -1

            # lowercase for all entities
            ins_id = action.ins_id

            if isinstance(ins_id, str):
                ins_id = ins_id.lower()

            # only instance class should apply entity map
            if type(action) not in [C, T] or ins_id not in self.entities_indexer:
                continue
            # record the entity id
            mapping[str(action)] = self.entities_indexer[ins_id]

        return mapping
