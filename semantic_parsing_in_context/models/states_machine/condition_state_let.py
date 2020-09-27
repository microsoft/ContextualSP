# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Union, Optional
from context.db_context import SparcDBContext
from context.copy_production_rule_field import CopyProductionRule
from typing import Dict
import copy
from context.grammar import A, C, T, Keywords, Statement
from constant import SpecialSymbol

logger = logging.getLogger(__name__)


class ConditionStatelet:
    """
    This class is designed to bring more SQL related common sense into current decoding phase. The main principle is:
    1. The accompanying Column and Table should be consistent.
    2. The same column cannot be repeated under the Select -> A A and so on. (FIXME: we do not support this now)
    """
    def __init__(self,
                 possible_actions: List[CopyProductionRule],
                 db_context: SparcDBContext,
                 enable_prune: bool = True):
        self.possible_actions = [action[0] for action in possible_actions]
        self.action_history = []
        self.valid_tables = self._get_valid_tables(db_context)
        self.current_stack: List[Union[str, List[str]]] = []

        self.used_terminals = []

        self.parent_path = []
        self.enable_prune = enable_prune

    @staticmethod
    def _get_valid_tables(db_context: Optional[SparcDBContext]) -> Dict[str, List[str]]:
        col_valid_tables = {}
        if db_context is not None:
            for entity_name in db_context.knowledge_graph.neighbors_with_table:
                # record column to table
                entity_parts = entity_name.split(':')
                if entity_parts[0] == 'column':
                    col_name = entity_parts[-1]
                    tab_name = entity_parts[-2]
                    # if not
                    if col_name not in col_valid_tables:
                        col_valid_tables[col_name] = []
                    col_valid_tables[col_name].append(tab_name)
            return col_valid_tables
        else:
            return {}

    def take_action(self, production_rule: str) -> 'ConditionStatelet':
        if not self.enable_prune:
            return self

        # the action to copy is actually correct
        special_str = SpecialSymbol.copy_delimiter
        # larger than 1 action is segment copy
        if special_str in production_rule and production_rule.count(special_str) >= 2:
            return self
        elif special_str in production_rule:
            production_rule = production_rule.replace(special_str, '')

        # clean stack
        new_sql_state = copy.deepcopy(self)
        lhs, rhs = production_rule.split(' -> ')

        # append current production rule
        new_sql_state.action_history.append(production_rule)
        new_sql_state.current_stack.append([lhs, []])

        if lhs not in [C.__name__, T.__name__]:
            rhs_tokens = rhs.split(' ')
        else:
            # default terminal not append into current stack
            # record when lhs equal to C.__name__
            parent_path = [new_sql_state.current_stack[i][0] for i in range(len(new_sql_state.current_stack))]
            # record parent path
            new_sql_state.parent_path = copy.deepcopy(parent_path)
            parent_path.append(rhs)
            new_sql_state.used_terminals.append(':'.join(parent_path))
            rhs_tokens = []

        for token in rhs_tokens:
            is_terminal = token in Keywords
            if not is_terminal:
                new_sql_state.current_stack[-1][1].append(token)

        while len(new_sql_state.current_stack) > 0 and \
                len(new_sql_state.current_stack[-1][1]) == 0:
            finished_item = new_sql_state.current_stack[-1][0]
            del new_sql_state.current_stack[-1]
            if finished_item == Statement.__name__:
                break
            # pop the non-terminals
            if new_sql_state.current_stack[-1][1][0] == finished_item:
                new_sql_state.current_stack[-1][1] = new_sql_state.current_stack[-1][1][1:]

        # append current stack
        return new_sql_state

    def get_valid_actions(self, valid_actions: dict):
        if not self.enable_prune:
            return valid_actions

        current_clause = self._get_current_clause()

        # used terminals to avoid repeated role, specially for Select -> A A A ...
        valid_actions_ids = []
        for key, items in valid_actions.items():
            valid_actions_ids += [(key, rule_id) for rule_id in valid_actions[key][2]]
        valid_actions_rules = [self.possible_actions[rule_id] for rule_type, rule_id in valid_actions_ids]

        # k is the group index
        actions_to_remove = {k: set() for k in valid_actions.keys()}

        # if not None
        if current_clause:
            # repeat constraints
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                # rhs is the key for querying
                lhs, rhs = rule.split(' -> ')

                # C, T should be in the same table
                if lhs == T.__name__:
                    # take the rhs
                    column_name = self.action_history[-1].split(' -> ')[1]
                    # column name is *, no limited tables
                    if column_name == '*':
                        continue
                    assert column_name in self.valid_tables
                    valid_table_name = self.valid_tables[column_name]
                    if rhs not in valid_table_name:
                        actions_to_remove[rule_type].add(rule_id)

                unique_key = ':'.join(self.parent_path) + lhs + ':' + rhs
                # repeated column/table
                if unique_key in self.used_terminals:
                    actions_to_remove[rule_type].add(rule_id)

        # now we only prevent linked rules
        new_valid_actions = {}
        new_global_actions = self._remove_actions(valid_actions, 'global',
                                                  actions_to_remove['global']) if 'global' in valid_actions else None
        new_linked_actions = self._remove_actions(valid_actions, 'linked',
                                                  actions_to_remove['linked']) if 'linked' in valid_actions else None

        if new_linked_actions is not None:
            new_valid_actions['linked'] = new_linked_actions
        if new_global_actions is not None:
            new_valid_actions['global'] = new_global_actions

        for key in valid_actions.keys():
            if key == 'copy_seg' or key == 'copy_token':
                new_valid_actions[key] = valid_actions[key]

        return new_valid_actions

    def _get_current_clause(self):
        relevant_clauses = [
            A.__name__
        ]
        for rule in self.current_stack[::-1]:
            # the first nonterminal which should be parsed
            if rule[0] in relevant_clauses:
                return rule[0]

        return None

    @staticmethod
    def _remove_actions(valid_actions, key, ids_to_remove):
        if len(ids_to_remove) == 0:
            return valid_actions[key]

        if len(ids_to_remove) == len(valid_actions[key][2]):
            return None

        current_ids = valid_actions[key][2]
        keep_ids = []
        keep_ids_loc = []

        for loc, rule_id in enumerate(current_ids):
            if rule_id not in ids_to_remove:
                keep_ids.append(rule_id)
                keep_ids_loc.append(loc)

        items = list(valid_actions[key])
        items[0] = items[0][keep_ids_loc]
        items[1] = items[1][keep_ids_loc]
        items[2] = keep_ids

        if len(items) >= 4:
            items[3] = items[3][keep_ids_loc]
        return tuple(items)
