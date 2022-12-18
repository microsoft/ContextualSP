# %%
import sys

sys.path.append("..")
from utils import *

# %%
data_dir = 'data/wtq_grounding'
bert_version = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_version)
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print('load Bert tokenizer over, vocab size = {}'.format(len(tokenizer)))
number_type = ['<c, date>',
               '<c, number>',
               '<c, score(score)>',
               '<c, score>',
               '<c, time>',
               '<c, timespan(text)>',
               '<c, timespan>',
               '<c, unitnum>',
               '<c,number(text)>',
               '<c,number>',
               '<c,numberspan>',
               '<c,rank(text)>',
               '<c,score(number)>',
               '<c,score(score)>',
               '<c,score>',
               '<c,time(text)>',
               '<c,time>',
               '<c,timespan(text)>',
               '<c,timespan>',
               '<cn, timespan>',
               '<cn,numberspan(text)>', '<cs, timespan>', '<d, number>', '<d, unitnum(unitnum)>',
               '<d, unitnum>',
               '<d, unitnum>(<d, unitnum>)', '<f, unitnum>',
               '<f,date>',
               '<f,number>', '<m, number>',
               '<m,score>', '<n, date(text)>',
               '<n, number>', '<n, timespan>',
               '<n, unitnum(text)>',
               '<n, unitnum>',
               '<n,date>',
               '<n,number>',
               '<n,score(text)>', '<n,time(text)>',
               '<n,time>',
               '<n,timespan>',
               '<nc,rank(text)>', '<s,date>',
               '<s,score>', '<s,timespan>', '<t, number>', '<t,number>', '<t,unitnum>', '<w,score(number)>', ]


def get_ctype(raw_type: str):
    if raw_type.startswith("unitnum") \
            or raw_type.startswith("time") \
            or raw_type.startswith("score") \
            or raw_type.startswith("rank") \
            or raw_type.startswith("percentage") \
            or raw_type.startswith("number") \
            or raw_type.startswith("fraction") \
            or raw_type.startswith("date") \
            or raw_type in number_type:
        return "number"
    else:
        return "text"


# %%
@dataclass
class Column:
    header: str
    data_type: str
    values: List[object]


@dataclass
class WTQTable:
    table_id: str
    columns: List[str]
    types: List[str]
    columns_internal: List[Column]
    internal_to_header: List[int]
    header_to_internals: List[List[int]]

    def get_schema(self):
        return WTQSchema(
            table_id=self.table_id,
            column_headers=self.columns,
            column_names_internal=[x.header for x in self.columns_internal],
            column_types_internal=[x.data_type for x in self.columns_internal],
            internal_to_header=self.internal_to_header)

    @classmethod
    def from_dataset_json(cls, obj: Dict):
        contents = obj['contents']

        assert contents[0][0]['col'] == 'id'
        assert contents[1][0]['col'] == 'agg'

        headers = []
        columns_internal = []
        internal_to_header = []
        header_to_internals = []
        column_type = []
        for i, content in enumerate(contents):
            if i < 2:
                continue

            column_type.append(get_ctype(obj['types'][i]))
            internal_ids = []
            for col in content:
                column = Column(header=col['col'], data_type=col['type'], values=col['data'])
                internal_ids.append(len(columns_internal))
                internal_to_header.append(len(headers))
                columns_internal.append(column)

            headers += [obj['headers'][i]]
            header_to_internals += [internal_ids]

        return WTQTable(
            table_id=None,
            columns=headers,
            types=column_type,
            columns_internal=columns_internal,
            internal_to_header=internal_to_header,
            header_to_internals=header_to_internals)


def load_wtq_tables(data_dir: str) -> Dict[str, WTQTable]:
    wtq_tables = {}
    for file_name in os.listdir(data_dir):
        assert file_name.endswith('.json')
        path = os.path.join(data_dir, file_name)
        table = WTQTable.from_dataset_json(json.load(open(path, 'r', encoding='utf-8')))
        table.table_id = file_name.replace('.json', '')
        assert table.table_id not in wtq_tables
        wtq_tables[table.table_id] = table

    print('load {} tables from {} over.'.format(len(wtq_tables), data_dir))
    return wtq_tables


def column_type_statistics(wtq_tables: List[WTQTable]):
    types = {}
    for table in wtq_tables:
        for col in table.columns_internal:
            types[col.data_type] = types.get(col.data_type, 0) + 1

    for key, val in types.items():
        print(key, val)


# Download from https://github.com/tzshi/squall/tree/main/tables/json and copy folder to '../data/squall/'
tables = load_wtq_tables(f'{data_dir}/json')
column_type_statistics(list(tables.values()))
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


def _generate_labels_from_ralign(align_labels):
    labels = set([])
    for align_type, align_value, _ in align_labels:
        if align_type != 'Column':
            continue
        if align_type == 'Column':
            assert isinstance(align_value, str)
            labels.add(align_value)

    return list(labels)


def _generate_labels_from_sql_and_slign(query):
    result = [['None', ""] for _ in range(len(query['nl']))]
    for y_step, (ttype, value, span) in enumerate(query['sql']):
        if ttype == 'Column':
            for xs, ys in query['align']:
                if y_step in ys:
                    for x in xs:
                        result[x] = ['Column', value]
    return result


sql_value_suffix = defaultdict(int)


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
            else:
                tokens += [KeywordToken(keyword=item[1])]
        elif item[0] == 'Column':
            tokens += [ColumnToken(column_header=item[1], suffix_type="")]

        elif item[0].startswith('Literal'):
            suffix = item[0].replace('Literal.', '')
            sql_value_suffix[suffix] += 1
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
    return SQLExpression(db_id=schema.table_id, tokens=tokens)


ngram_matchers: Dict[str, NGramMatcher] = {}


def generate_masking_ngrams(question: Utterance, schema: WTQSchema) -> List[Tuple[int, int, str]]:
    if schema.table_id not in ngram_matchers:
        column_tokens = []
        for i, column in enumerate(schema.column_headers):
            column_tokens.append((column, column.split(' ')))
        ngram_matchers[schema.table_id] = NGramMatcher(column_tokens)

    ngram_matcher = ngram_matchers[schema.table_id]
    masking_ngrams = []
    for tok_idx in range(len(question.tokens)):
        masking_ngrams.append((tok_idx, tok_idx, question.tokens[tok_idx].token))

    ngram_spans = set([])
    for q_i, q_j, _, _, _ in ngram_matcher.match([token.token for token in question.tokens]):
        ngram_spans.add((q_i, q_j))

    for q_i, q_j in sorted(list(ngram_spans), key=lambda x: x[1] - x[0], reverse=True):
        is_overlap = False
        for q_i2, q_j2, ngram in masking_ngrams:
            if q_i2 <= q_i and q_j2 >= q_j:
                is_overlap = True
                break
        if not is_overlap:
            ngram_ij = " ".join([x.token for x in question.tokens[q_i:q_j + 1]])
            masking_ngrams.append((q_i, q_j, ngram_ij))

    return masking_ngrams


column_suffix_dict = defaultdict(int)


def process_squall_test_set(query: Dict) -> Dict:
    question_tokens = query['nl']
    question_utterance = generate_utterance(tokenizer, None, question_tokens, None)

    table = tables[query['tbl']]
    schema = table.get_schema()
    processed_columns_internal = []
    for i, column in enumerate(table.columns_internal):
        header = schema.column_headers[schema.internal_to_header[i]]
        assert '_' not in header
        suffix = re.sub('^c\d+', '', column.header).replace('_', ' ')
        true_col_name = header + ' :' + suffix
        column_suffix_dict[suffix] += 1
        column_utterance = generate_utterance(tokenizer, true_col_name)
        column_json = {
            'index': i,
            'id': column.header,
            'data_type': wtq_type_mappings[column.data_type],
            'utterance': column_utterance.to_json()
        }

        processed_columns_internal += [column_json]

    processed_columns = []
    for i, column in enumerate(table.columns):
        column_utterance = generate_utterance(tokenizer, column)
        column_json = {
            'index': i,
            'id': column,
            'data_type': table.types[i],
            'utterance': column_utterance.to_json()
        }
        processed_columns += [column_json]
    masking_ngrams = generate_masking_ngrams(question_utterance, schema)
    return {
        'id': query['nt'],
        'question': question_utterance.to_json(),
        'columns': processed_columns,
        'columns_internal': processed_columns_internal,
        'masking_ngrams': masking_ngrams,
        'schema': schema.to_json()
    }


def process_squall_query(query: Dict) -> Dict:
    question_tokens = query['nl']
    question_utterance = generate_utterance(tokenizer, None, question_tokens, None)

    table = tables[query['tbl']]
    schema = table.get_schema()
    processed_columns_internal = []
    for col_index, column in enumerate(table.columns_internal):
        header = schema.column_headers[schema.internal_to_header[col_index]]
        assert '_' not in header
        suffix = re.sub('^c\d+', '', column.header).replace('_', ' ')
        true_col_name = header + ' :' + suffix
        column_suffix_dict[suffix] += 1
        column_utterance = generate_utterance(tokenizer, true_col_name)
        column_json = {
            'index': col_index,
            'id': column.header,
            'data_type': wtq_type_mappings[column.data_type],
            'utterance': column_utterance.to_json()
        }

        processed_columns_internal += [column_json]

    processed_columns = []
    for col_index, column in enumerate(table.columns):
        column_utterance = generate_utterance(tokenizer, column)
        column_json = {
            'index': col_index,
            'id': column,
            'data_type': table.types[col_index],
            'utterance': column_utterance.to_json()
        }
        processed_columns += [column_json]

    identify_labels_internal = _generate_labels_from_ralign(query['sql'])
    identify_labels = {
        "column": []
    }

    for col_name in identify_labels_internal:
        header_id = schema.lookup_header_id_from_internal(col_name)
        identify_labels["column"].append(schema.column_headers[header_id])

    parsed_sql = parse_squall_sql(query['sql'], schema)
    masking_ngrams = generate_masking_ngrams(question_utterance, schema)
    return {
        'id': query['nt'],
        'question': question_utterance.to_json(),
        'columns': processed_columns,
        'columns_internal': processed_columns_internal,
        'identify_labels': identify_labels,
        'sql': parsed_sql.to_json(),
        'align_labels': _generate_labels_from_sql_and_slign(query),
        'masking_ngrams': masking_ngrams,
        'schema': schema.to_json()
    }


# %%
test_query = json.load(open(f'{data_dir}/test.json', 'r', encoding='utf-8'))
print('load test over, size = {}'.format(len(test_query)))
dev_queries = json.load(open(f'{data_dir}/dev.json', 'r', encoding='utf-8'))
print('load dev over, size = {}'.format(len(dev_queries)))

train_queries = json.load(open(f'{data_dir}/train.json', 'r', encoding='utf-8'))
print('load train over, size = {}'.format(len(train_queries)))
# %%
process_squall_query(dev_queries[0])
# %%
dev_processed = []
for query in tqdm(dev_queries):
    try:
        dev_processed += [process_squall_query(query)]
    except Exception as ex:
        print(ex)
save_json_objects(dev_processed, os.path.join(data_dir, 'dev.{}.json'.format(bert_version)))
print('process dev over')
# %%
train_processed = []
for query in tqdm(train_queries):
    try:
        processed_query = process_squall_query(query)
        train_processed += [processed_query]
    except Exception as ex:
        print(ex)
save_json_objects(train_processed,
                  os.path.join(data_dir, 'train.{}.json'.format(bert_version)))
print('process train over, {}/{}'.format(len(train_processed), len(train_queries)))

process_squall_test_set(test_query[0])
test_processed = []
for query in tqdm(test_query):
    try:
        test_processed += [process_squall_test_set(query)]
    except Exception as ex:
        print(ex)
save_json_objects(test_processed, os.path.join(data_dir, 'test.{}.json'.format(bert_version)))
print('process test over')
test_iter = load_wtq_data_iterator(f'{data_dir}/test.{bert_version}.json', tokenizer, 1,
                                   torch.device('cpu'), False, False, 300)
