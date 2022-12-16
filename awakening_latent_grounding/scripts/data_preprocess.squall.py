#%%
import os
import sys
import re
sys.path.append("..")
import json
from collections import defaultdict, Counter
from typing import Counter, Dict, List, Tuple
from transformers import BertTokenizer
from tqdm import tqdm
from utils import *
#%%
data_dir = r'../data/squall'
bert_version = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_version)
print('load Bert tokenizer over, vocab size = {}'.format(len(tokenizer)))
#%%
@dataclass
class Column:
    header: str
    data_type: str
    values: List[object]

@dataclass
class WTQTable:
    table_id: str
    headers: List[str]
    columns_internal: List[Column]
    internal_to_header: List[int]

    def get_schema(self):
        return WTQSchema(
            table_id=self.table_id,
            column_headers=self.headers,
            column_names_internal=[x.header for x in self.columns_internal],
            column_types_internal=[x.data_type for x in self.columns_internal],
            internal_to_header=self.internal_to_header)

    @classmethod
    def from_wqt_json(cls, obj: Dict):
        contents = obj['contents']
        assert contents[0][0]['col'] == 'id'
        assert contents[1][0]['col'] == 'agg'
        assert len(contents) == len(obj['headers'])

        headers = []
        columns_internal = []
        internal_to_header = []
        header_to_internals = []

        internal_to_header += [len(headers)]
        header_to_internals += [[len(columns_internal)]]
        headers += ['id']
        columns_internal += [Column(header='id', data_type='INTEGER', values=[])]

        for i, content in enumerate(contents):
            if i < 2:
                continue

            internal_ids = []
            for col in content:
                column = Column(header=col['col'], data_type=col['type'], values=col['data'])
                internal_ids.append(len(columns_internal))
                internal_to_header.append(len(headers))
                columns_internal.append(column)

            headers += [obj['headers'][i]]
            header_to_internals += [internal_ids]
        
        internal_to_header += [len(headers)]
        header_to_internals += [[len(columns_internal)]]
        headers += ['*']
        columns_internal += [Column(header='*', data_type='TEXT', values=[])]

        assert len(headers) == len(header_to_internals)
        assert len(columns_internal) == len(internal_to_header)

        for i in range(len(headers)):
            headers[i] = headers[i].replace('\n', ' ')
            if len(headers[i]) == 0:
                headers[i] = 'c{}'.format(i)

        return WTQTable(
            table_id=None,
            headers=headers,
            columns_internal=columns_internal,
            internal_to_header=internal_to_header)

def load_wtq_tables(data_dir: str) -> Dict[str, WTQTable]:
    tables = {}
    for file_name in os.listdir(data_dir):
        assert file_name.endswith('.json')
        path = os.path.join(data_dir, file_name)
        table = WTQTable.from_wqt_json(json.load(open(path, 'r', encoding='utf-8')))
        table.table_id = file_name.replace('.json', '')
        assert table.table_id not in tables
        tables[table.table_id] = table
    
    print('load {} tables from {} over.'.format(len(tables), data_dir))
    return tables

def column_type_statistics(tables: List[WTQTable]):
    types = {}
    for table in tables:
        for col in table.columns_internal:
            types[col.data_type] = types.get(col.data_type, 0) + 1
    
    for key, val in types.items():
        print(key, val)

# Download from https://github.com/tzshi/squall/tree/main/tables/json and copy folder to '../data/squall/'
tables = load_wtq_tables(r'../data/squall/json')
column_type_statistics(tables.values())
# %%
wtq_type_mappings = {
    'TEXT': 'text',
    'REAL': 'real',
    'INTEGER': 'integer',
    'LIST TEXT': 'text',
    'LIST REAL': 'real',
    'LIST INTEGER': 'integer',
    'EMPTY': 'text'
    }

def generate_identify_labels_from_align(align_labels):
    identify_labels: Dict[str, List[str]] = { str(SQLTokenType.column): [] }
    for align_type, align_value in align_labels:
        if align_type == 'None':
            continue
        if align_type == 'Column':
            assert isinstance(align_value, str)
            identify_labels[str(SQLTokenType.column)].add(align_value)
    
    for key, labels in identify_labels.items():
        identify_labels[key] = list(set(labels))
    return identify_labels

def generate_identify_labels_from_sql(sql: SQLExpression):
    identify_labels: Dict[str, List[str]] = { str(SQLTokenType.column): [] }
    for token in sql.tokens:
        if isinstance(token, ColumnToken):
            identify_labels[str(SQLTokenType.column)].append(token.column_name)
    
    for key, labels in identify_labels.items():
        identify_labels[key] = list(set(labels))
    return identify_labels

def _get_value_span(indices: List[int]) -> Tuple[int, int]:
    assert len(indices) > 0
    if len(indices) == 1:
        return indices[0], indices[0]
    
    assert len(indices) == 2, indices
    return indices[0], indices[-1]

def parse_squall_sql(sql: Dict, schema: WTQSchema) -> SQLExpression:
    tokens = []
    for item in sql:
        if item[0] == 'Keyword':
            if item[1] == 'w':
                tokens += [TableToken(table_name=schema.table_id)]
            elif item[1] == 'id':
                tokens += [ColumnToken(column_header='id', suffix_type='')]
            elif item[1] == '*':
                tokens += [ColumnToken(column_header='*', suffix_type='')]
            else:
                tokens += [KeywordToken(keyword=item[1])]
        elif item[0] == 'Column':
            column, suffix = schema.lookup_header_and_suffix(item[1])
            tokens += [ColumnToken(column_header=column, suffix_type=suffix)]

        elif item[0].startswith('Literal'):
            suffix = item[0].replace('Literal.', '')
            span = _get_value_span(item[2])
            if item[0] == 'Literal.String':
                value = item[1]
                tokens += [ValueToken(value=value, span=span, columns=None)]
            elif item[0] == 'Literal.Number':
                value = item[1]
                tokens += [ValueToken(value=value, span=span, columns=None)]
            else:
                raise NotImplementedError(item[0])
        else:
            raise NotImplementedError("Not supported SQL type: {}".format(item[0]))
    return SQLExpression(tokens=tokens)

ngram_matchers: Dict[str, NGramMatcher] = {}
def generate_masking_ngrams(question: Utterance, schema: WTQSchema) -> List[Tuple[int, int, str]]:
    if schema.table_id not in ngram_matchers:
        column_tokens = []
        for column in schema.column_headers:
            column_tokens.append((column, column.split(' ')))
        ngram_matchers[schema.table_id] = NGramMatcher(column_tokens)
    
    ngram_matcher = ngram_matchers[schema.table_id]
    masking_ngrams = []
    for tok_idx in range(len(question.tokens)):
        masking_ngrams.append((tok_idx, tok_idx, question.tokens[tok_idx].token))
    
    ngram_spans = set([])
    for q_i, q_j, _, _, _ in ngram_matcher.match([token.token for token in question.tokens]):
        ngram_spans.add((q_i, q_j))
    
    for q_i, q_j in sorted(list(ngram_spans), key=lambda x: x[1]-x[0], reverse=True):
        is_overlap = False
        for q_i2, q_j2, ngram in masking_ngrams:
            if q_i2 <= q_i and q_j2 >= q_j:
                is_overlap = True
                break
        if not is_overlap:
            ngram_ij = " ".join([x.token for x in question.tokens[q_i:q_j+1]])
            masking_ngrams.append((q_i, q_j, ngram_ij))

    return masking_ngrams

vocab = { 'keyword': Counter(), 'suffix_type': Counter() }

def process_squall_query(query: Dict) -> Dict:
    question_tokens = query['nl']
    question_utterance = generate_utterance(tokenizer, None, question_tokens, None)

    table = tables[query['tbl']]
    schema = table.get_schema()
    processed_columns_internal = []
    for i, column in enumerate(table.columns_internal):
        header, suffix = schema.lookup_header_and_suffix(column.header)
        true_col_name = header + ' :' + suffix
        vocab['suffix_type'][suffix] += 1
        column_utterance = generate_utterance(tokenizer, true_col_name)
        column_json = {
            'index': i,
            'id': column.header,
            'data_type': wtq_type_mappings[column.data_type],
            'utterance': column_utterance.to_json()
            }

        processed_columns_internal += [column_json]
    
    processed_columns = []
    for i, column in enumerate(table.headers):
        column_utterance = generate_utterance(tokenizer, column)
        column_json = {
            'index': i,
            'id': column,
            'data_type': 'text',
            'utterance': column_utterance.to_json()
        }
        processed_columns += [column_json]
    
    # Parse SQL
    sql = parse_squall_sql(query['sql'], schema)
    for term in sql.tokens:
        if isinstance(term, KeywordToken):
            vocab['keyword'][term.value] += 1

    identify_labels = generate_identify_labels_from_sql(sql)
    # identify_labels_internal = generate_identify_labels_from_align(query['nl_ralign'])

    return {
            'id': query['nt'],
            'question': question_utterance.to_json(),
            'columns': processed_columns,
            'columns_internal': processed_columns_internal,
            'identify_labels': identify_labels,
            'align_labels': query['nl_ralign'],
            'sql': sql.to_json(),
            'masking_ngrams': generate_masking_ngrams(question_utterance, schema),
            'schema': schema.to_json()
        }
#%%
dev_queries = json.load(open(r'../data/squall/dev-0.json', 'r', encoding='utf-8'))
print('load dev over, size = {}'.format(len(dev_queries)))

train_queries = json.load(open(r'../data/squall/train-0.json', 'r', encoding='utf-8'))
print('load train over, size = {}'.format(len(train_queries)))
# %%
process_squall_query(dev_queries[0])
process_squall_query(dev_queries[7])
# %%
dev_processed = []
for query in tqdm(dev_queries):
    dev_processed += [process_squall_query(query)]
save_json_objects(dev_processed, os.path.join(r'../data/squall/', 'dev.{}.json'.format(bert_version)))
print('process dev over')
# %%
dev_iter = load_wtq_data_iterator(os.path.join(data_dir, 'dev.{}.json'.format(bert_version)), tokenizer, 16, torch.device('cpu'), False, False, 300)
# %%
total_size, num_examples = 0, 0
for batch_input in dev_iter:
    bs, length = batch_input['input_token_ids'].size(0), batch_input['input_token_ids'].size(1)
    total_size += bs * length
    num_examples += bs
    #print(batch_input['input_token_ids'].size())
print(total_size, num_examples, total_size / num_examples)
#%%
def sampling_sqls(data_iter: DataLoader, count: int):
    while count > 0:
        for batch_input in data_iter:
            for example in batch_input['example']:
                sql = SQLExpression.from_json(example['sql'])
                print(sql)
                count -= 1

# %%
train_processed = []
for query in tqdm(train_queries):
    try:
        processed_query = process_squall_query(query)
        train_processed += [processed_query]
    except Exception as ex:
        print(ex)
save_json_objects(train_processed, os.path.join(r'../data/squall/', 'train.{}.json'.format(bert_version)))
print('process train over, {}/{}'.format(len(train_processed), len(train_queries)))
# %%
#%%
print('Keyword Vocab:')
print(vocab['keyword'].most_common(20))
keyword_saved_path = os.path.join(data_dir, 'keyword.vocab.txt')
with open(keyword_saved_path, 'w', encoding='utf-8') as fw:
    for keyword, cnt in sorted(vocab['keyword'].items(), key=lambda x: x[1], reverse=True):
        fw.write("{}\t{}\n".format(keyword, cnt))
print('save keyword vocab into {} over.'.format(keyword_saved_path))

#%%
print('Suffix Type Vocab:')
print(vocab['suffix_type'].most_common(20))
suffix_type_saved_path = os.path.join(data_dir, 'suffix_type.vocab.txt')
with open(suffix_type_saved_path, 'w', encoding='utf-8') as fw:
    for keyword, cnt in sorted(vocab['suffix_type'].items(), key=lambda x: x[1], reverse=True):
        fw.write("{}\t{}\n".format(keyword, cnt))
print('save suffix_type vocab into {} over.'.format(suffix_type_saved_path))

# %%
train_iter = load_wtq_data_iterator(os.path.join(data_dir, 'train.{}.json'.format(bert_version)), tokenizer, 16, torch.device('cpu'), True, True, 300)
total_size, num_examples = 0, 0
for batch_input in train_iter:
    bs, length = batch_input['input_token_ids'].size(0), batch_input['input_token_ids'].size(1)
    total_size += bs * length
    num_examples += bs
    #print(batch_input['input_token_ids'].size())
print(total_size, num_examples, total_size / num_examples)
# %%
train_iter2 = load_wtq_data_iterator(os.path.join(data_dir, 'train.{}.json'.format(bert_version)), tokenizer, 16, torch.device('cpu'), False, True, 300)
total_size, num_examples = 0, 0
for batch_input in train_iter2:
    bs, length = batch_input['input_token_ids'].size(0), batch_input['input_token_ids'].size(1)
    total_size += bs * length
    num_examples += bs
    #print(batch_input['input_token_ids'].size())
print(total_size, num_examples, total_size / num_examples)
# %%
