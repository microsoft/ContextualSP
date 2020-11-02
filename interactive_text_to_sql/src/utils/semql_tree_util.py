# coding: utf-8

import random
import copy
from typing import List


class Node:
    STATEMENT_TYPE = 0
    ROOT_TYPE = 1
    SELECT_TYPE = 2
    FILTER_TYPE = 3
    ORDER_TYPE = 4
    A_TYPE = 5

    COLUMN_TYPE = 11
    TABLE_TYPE = 12
    KEYWORD_TYPE = 13

    VALUE_TYPE = '3'

    TYPE_DICT = {'SQL': None, 'Statement': STATEMENT_TYPE, 'Root': ROOT_TYPE, 'Select': SELECT_TYPE, 'Filter': FILTER_TYPE,
                 'Order': ORDER_TYPE, 'A': A_TYPE, 'C': COLUMN_TYPE, 'T': TABLE_TYPE, 'Value': VALUE_TYPE}
    KEYWORD_LIST = [
        'intersect', 'union', 'except',  # Statement
        'asc', 'des', 'limit',  # Order
        'and', 'or', '>', '<', '>=', '<=', '=', '!=', 'between', 'like', 'not_like', 'in', 'not_in',  # Filter
        'max', 'min', 'count', 'sum', 'avg', 'none'  # A
    ]
    RULE_TEMPLATES = {
        # SQL
        'Statement': 'Find out {0}',
        # Statement
        'intersect Root Root': 'the common part of the set of {0} and the set of {1}',
        'union Root Root': 'everyone in the set of {0} and the set of {1}',
        'except Root Root': 'everyone in the set of {0} but not in the set of {1}',
        'Root': '{0}',
        # Root
        'Select Filter Order': '{0}, {1}, {2}',
        'Select Filter': '{0}, {1}',
        'Select Order': '{0}, {1}',
        'Select': '{0}',
        # Select
        'A': '{0}',
        'A A': '{0} and {1}',
        'A A A': '{0} , {1} and {2}',
        'A A A A': '{0} , {1} , {2} and {3}',
        'A A A A A': '{0} , {1} , {2} , {3} and {4}',
        'A A A A A A': '{0} , {1} , {2} , {3} , {4} and {5}',
        # A
        'none C T': 'the {0} of {1}',
        'max C T': 'the maximum {0} of {1}',
        'min C T': 'the minimum {0} of {1}',
        'count C T': 'the number of {0} of {1}',
        'sum C T': 'the sum of {0} of {1}',
        'avg C T': 'the average {0} of {1}',
        # Filter
        'Filter and Filter': '{0} and {1}',
        'Filter or Filter': '{0} or {1}',
        '= A': 'where {0} is {1}'.format('{0}', VALUE_TYPE),#['where {0} equals to {1}'.format('{0}', VALUE_TYPE), 'where {0} is {1}'.format('{0}', VALUE_TYPE)],
        '> A': 'where {0} greater than {1}'.format('{0}', VALUE_TYPE),
        '< A': 'where {0} less than {1}'.format('{0}', VALUE_TYPE),
        '>= A': 'where {0} greater than or equals to {1}'.format('{0}', VALUE_TYPE),
        '<= A': 'where {0} less than or equals to {1}'.format('{0}', VALUE_TYPE),
        '!= A': 'where {0} not equals to {1}'.format('{0}', VALUE_TYPE),
        'between A': 'where {0} between {1} and {2}'.format('{0}', VALUE_TYPE, VALUE_TYPE),
        'like A': 'where {0} like {1}'.format('{0}', VALUE_TYPE),
        'not_like A': 'where {0} not like {1}'.format('{0}', VALUE_TYPE),
        '= A Root': ['where {0} equals to {1}', 'where {0} is {1}'],
        '> A Root': 'where {0} greater than {1}',
        '< A Root': 'where {0} less than {1}',
        '>= A Root': 'where {0} greater than or equals to {1}',
        '<= A Root': 'where {0} less than or equals to {1}',
        '!= A Root': 'where {0} not equals to {1}',
        'between A Root': 'where {0} is between {1}',  # todo: useless
        'in A Root': 'where {0} is in the set of {1}',
        'not_in A Root': 'where {0} is not in the set of {1}',
        # Order
        'asc A': 'in ascending order of {0}',
        'des A': 'in descending order of {0}',
        'asc A limit': 'in ascending order of {0}' + 'with maximum {0} item(s)'.format(VALUE_TYPE),
        'des A limit': 'in descending order of {0}' + 'with maximum {0} item(s)'.format(VALUE_TYPE),
    }
    RULE_TEMPLATES_WITHOUT_TABLE = copy.copy(RULE_TEMPLATES)
    RULE_TEMPLATES_WITHOUT_TABLE.update({
        'none C T': 'the {0}',
        'max C T': 'the maximum {0}',
        'min C T': 'the minimum {0}',
        'count C T': 'the number of {0}',
        'sum C T': 'the sum of {0}',
        'avg C T': 'the average {0}',
    })

    def __init__(self, node_id: int,
                 text: str,
                 statement: str,
                 children_tokens: List[str],
                 father=None,
                 depth=0):
        self.node_id = node_id
        self.text = text
        self.statement = statement.split(' -> ')[1].strip()  # align with irnet.context.grammar
        if text in Node.TYPE_DICT:
            self.type = Node.TYPE_DICT.get(text)
        elif text in Node.KEYWORD_LIST:
            self.type = Node.KEYWORD_TYPE
        else:
            raise Exception('Node type error')
        self._children_tokens = children_tokens
        self.children = []
        self.father = father
        self.depth = depth
        # father_text = 'None' if father is None else father.text
        # print(f'Created node: text={text}, children_tokens={children_tokens}, father={father_text}')
        self.checked: bool = True
        self.more_info = {}  # data container for outside operations

    def bfs(self, process_f=lambda x: x, node_only=True):
        ret_list = []
        queue = [self]
        while queue:
            node = queue[0]
            del queue[0]
            if isinstance(node, Node):
                ret_list.append(process_f(node))
            elif isinstance(node, str) and not node_only:
                ret_list.append(process_f(node))
            if isinstance(node, Node) and node.children:
                queue += node.children
        return ret_list

    def restatement(self, with_table=True):
        # 1. Find if Root node exists in subtree, if unchecked leaved, raise error  # todo: may be eliminated
        subtree_nodes = self.bfs()
        if False in [node.checked for node in subtree_nodes if not isinstance(node, str) and node.type == Node.ROOT_TYPE]:
            raise Exception('Unchecked Root node exists, check it first')

        # 2. Restate each child
        return self._restate_node(self, with_table=with_table)

    def restatement_with_tag(self):
        # 1. Find if Root node exists in subtree, if unchecked leaved, raise error  # todo: may be eliminated
        subtree_nodes = self.bfs()
        if False in [node.checked for node in subtree_nodes if
                     not isinstance(node, str) and node.type == Node.ROOT_TYPE]:
            raise Exception('Unchecked Root node exists, check it first')

        # 2. Restate each child
        return self._restate_node_with_tag(self)

    @staticmethod
    def _restate_node(node, with_table=True):
        if isinstance(node, str) and node not in Node.KEYWORD_LIST:
            raise Exception('WA!!!')
        node_statement = node.statement
        templates = Node.RULE_TEMPLATES if with_table else Node.RULE_TEMPLATES_WITHOUT_TABLE
        if node_statement in templates:
            rule_template = templates.get(node_statement)
            if isinstance(rule_template, List):
                rule_template = random.sample(rule_template, 1)[0]
            format_strings = []
            for child in node.children:
                if isinstance(child, str) and child in Node.KEYWORD_LIST:
                    continue
                format_strings.append(Node._restate_node(child, with_table=with_table))
            return rule_template.format(*format_strings)
        else:
            return ' '.join(node_statement.split('_')) if node_statement is not '*' else 'items'  # select *

    @staticmethod
    def _restate_node_with_tag(node, with_table=True):
        if isinstance(node, str) and node not in Node.KEYWORD_LIST:
            raise Exception('WA!!!')
        node_statement = node.statement
        templates = Node.RULE_TEMPLATES if with_table else Node.RULE_TEMPLATES_WITHOUT_TABLE
        if node_statement in templates:
            rule_template = templates.get(node_statement)
            if isinstance(rule_template, List):
                rule_template = random.sample(rule_template, 1)[0]
            sub_strings = []
            sub_string_tags = []
            for child in node.children:
                if isinstance(child, str) and child in Node.KEYWORD_LIST:
                    continue
                node_restatement_string, node_restatement_tag = Node._restate_node_with_tag(child)
                sub_strings.append(node_restatement_string)
                sub_string_tags.append(node_restatement_tag)
            restatement_string = rule_template.format(*sub_strings)
            restatement_tag = []
            nonterminal_children = [_ for _ in node.children if isinstance(_, Node)]
            for word in rule_template.split():
                if word.startswith('{') and word.endswith('}'):
                    placeholder_idx = int(word[1:-1])
                    if sub_string_tags[placeholder_idx] is not None:
                        restatement_tag += sub_string_tags[placeholder_idx]
                    else:
                        restatement_tag += nonterminal_children[placeholder_idx].text * len(sub_strings[placeholder_idx].split())
                else:
                    restatement_tag.append(node.text)
            return restatement_string, restatement_tag
        else:
            return ' '.join(node_statement.split('_')) if node_statement is not '*' else 'items', None
        # todo: tag of table and column with split char

    @staticmethod
    def print_subtree(node):
        def _print_subtree(node):
            print('   ' * node.depth + node.text)
            for child in node.children:
                if isinstance(child, Node):
                    _print_subtree(child)
                else:
                    print('   ' * (node.depth + 1) + child)

        _print_subtree(node)

    def clear_more_info_recursively(self, keys=None):
        if keys is None:
            def clear_more_info(node):
                node.more_info.clear()
        else:
            def clear_more_info(node):
                for key in keys:
                    if key in node.more_info:
                        del node.more_info[key]

        self.bfs(process_f=clear_more_info)

    def compare_node(self, node) -> bool:
        if self.type == node.type and self.children == node.children:
            return True
        else:
            return False

    def compare_tree(self, node) -> bool:
        if self.type != node.type or self.statement != node.statement:
            self.more_info['subtree_equal'] = node.more_info['subtree_equal'] = False
            self.bfs(lambda x: x.more_info.update({'subtree_equal': False}))
            node.bfs(lambda x: x.more_info.update({'subtree_equal': False}))
            return False
        else:
            assert len(self.children) == len(node.children)
            status = True
            for child1, child2 in zip(self.children, node.children):
                if isinstance(child1, str) or isinstance(child2, str):
                    if child1 != child2:
                        status = False
                elif child1.compare_tree(child2) is False:
                    status = False
            self.more_info['subtree_equal'] = node.more_info['subtree_equal'] = status
            self.bfs(lambda x: x.more_info.update({'subtree_equal': status}))
            node.bfs(lambda x: x.more_info.update({'subtree_equal': status}))
            return status

    @staticmethod
    def from_statements(statements):
        root, _ = parse_sql_tree(statements)
        return root


def is_nonterminal(token):
    letter = token[0]
    if ord('A') <= ord(letter) <= ord('Z'):
        return True
    else:
        return False


def parse_sql_tree(tree_statements):
    # print(tree_statements)
    max_depth = -1
    depth = 0
    stack = []
    root = Node(0, 'SQL', 'SQL -> Statement', ['Statement'], depth=0)
    stack.append(root)
    for state_id, statement in enumerate(tree_statements):
        assert statement.split(' -> ')[0] == stack[-1]._children_tokens[0]  # non-terminal match
        # print(f'statement = {statement}')
        nonterminal, children = statement.split('->')
        nonterminal = nonterminal.strip()
        children = [child.strip() for child in children.strip().split(' ')]
        node = Node(state_id, nonterminal, statement, children, father=stack[-1], depth=depth)
        stack[-1].children.append(node)
        del stack[-1]._children_tokens[0]
        stack.append(node)
        depth += 1
        max_depth = max(max_depth, depth)
        while stack:
            # move terminal tokens from children_tokens into children
            while stack[-1]._children_tokens and not is_nonterminal(stack[-1]._children_tokens[0]):
                stack[-1].children.append(stack[-1]._children_tokens[0])
                del stack[-1]._children_tokens[0]
            # layer up if no child waiting for process
            if len(stack[-1]._children_tokens) == 0:
                stack.pop()
                depth -= 1
                if len(stack) == 0:
                    return root, max_depth
            else:
                break
    return root, max_depth


if __name__ == '__main__':
    # sql_tree = ['Statement -> Root', 'Root -> Select Filter',
    #             'Select -> A', 'A -> count C T', 'C -> budget_in_billions', 'T -> department',
    #             'Filter -> > A', 'A -> none C T', 'C -> age', 'T -> head']
    sql_tree = ['Statement -> Root', 'Root -> Select Filter',
                    'Select -> A', 'A -> sum C T', 'C -> enr', 'T -> college',
                    'Filter -> not_in A Root', 'A -> none C T', 'C -> cname', 'T -> college',
                        'Root -> Select Filter',
                            'Select -> A', 'A -> none C T', 'C -> cname', 'T -> tryout',
                            'Filter -> = A', 'A -> none C T', 'C -> ppos', 'T -> tryout']
    root, max_depth = parse_sql_tree(sql_tree)
    Node.print_subtree(root)
    restatement = root.restatement_with_tag()
    print(restatement)
