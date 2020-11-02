from typing import List, Tuple, Dict, Set, Optional
from copy import deepcopy

from .db_context import SparcDBContext
from .grammar import Grammar, Action, C, T
from .converter import SQLConverter


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
                _, _, _, column_name = parts
                # TODO: here we assume the same column name always map into the same text
                self.entities_indexer[f'{column_name}'] = i

        self.valid_actions: Dict[str, List[str]] = {}
        self.valid_actions_flat: List[Action] = []

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
            action_key = action.__class__.__name__
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

    def get_oracle_relevance_score(self, oracle_entities: set):
        """
        return 0/1 for each schema item if it should be in the graph,
        given the used entities in the gold answer
        """
        scores = [0 for _ in range(len(self.db_context.knowledge_graph.entities))]

        for i, entity in enumerate(self.db_context.knowledge_graph.entities):
            parts = entity.split(':')
            if parts[0] == 'column':
                name = parts[2] + '@' + parts[3]
            else:
                name = parts[-1]
            if name in oracle_entities:
                scores[i] = 1

        return scores

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
