import sys

sys.path.append("..")
from contracts import *
from utils.nlp_utils import *
from tqdm import tqdm

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(proj_dir, 'data', 'wikisql')
# %%

_Data_Type_Mappings = [DataType.Text, DataType.DateTime, DataType.Number, DataType.Number, DataType.Boolean]


def load_table_schemas(path: str) -> Dict[str, DBSchema]:
    schemas = {}
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            table_obj = json.loads(line)
            table_id = table_obj['table_id']
            columns = []
            for column_obj in table_obj['columns']:
                header, data_type = column_obj['name'], column_obj['data_type']
                header_tokens = [Token(token=token['token'], lemma=token['lemma']) for token in column_obj['tokens']]
                column = Column(name=header, tokens=header_tokens, data_type=_Data_Type_Mappings[data_type])
                columns += [column]

            schema = DBSchema(db_id=table_id, columns=columns)
            schemas[table_id] = schema
    print('load {} tables from {} over.'.format(len(schemas), path))
    return schemas


# dev_schemas = load_schemas(os.path.join(data_dir, 'dev.tables.json'))

# %%
ngram_matchers: Dict[str, NGramMatcher] = {}


def generate_erased_ngrams(question: Utterance, schema: DBSchema) -> List[Tuple[int, int, str]]:
    if schema.db_id not in ngram_matchers:
        column_tokens = []
        for i, column in enumerate(schema.columns):
            column_tokens.append((column.identifier, [x.token for x in column.tokens]))
        ngram_matchers[schema.db_id] = NGramMatcher(column_tokens)

    ngram_matcher = ngram_matchers[schema.db_id]
    erased_ngrams = []
    ngram_spans = set([])

    for tok_idx in range(len(question.tokens)):
        erased_ngrams.append((tok_idx, tok_idx, question.tokens[tok_idx].token))

    for q_i, q_j, _, _, _ in ngram_matcher.match([token.token for token in question.tokens]):
        ngram_spans.add((q_i, q_j))

    for q_i, q_j in sorted(list(ngram_spans), key=lambda x: x[1] - x[0], reverse=True):
        is_overlap = False
        for q_i2, q_j2, ngram in erased_ngrams:
            if q_i2 <= q_i and q_j2 >= q_j:
                is_overlap = True
                break
        if not is_overlap:
            ngram_ij = " ".join([x.token for x in question.tokens[q_i:q_j + 1]])
            erased_ngrams.append((q_i, q_j, ngram_ij))

    erased_ngrams = sorted(erased_ngrams, key=lambda x: x[0])
    return [Span(start=i, end=j) for i, j, _ in erased_ngrams]


# %%
# %%
def parse_ae_tokens(ae_tokens: List[Dict], schema: DBSchema) -> SQLExpression:
    sql_tokens: List[SQLToken] = []
    col_idx = -1
    expr_str = [token['value'] if token['value'] else '@null' for token in ae_tokens]
    expr_str = " ".join(expr_str)
    for idx, token in enumerate(ae_tokens):
        token_type = token['token_type']
        if token_type == 0:  # keyword
            sql_tokens += [SQLToken(type=SQLTokenType.Keyword, field=SQLFieldType.Select, value=token['value'])]

            if token['value'] == 'take':
                col_idx = idx

        elif token_type == 1:  # column
            sql_tokens += [SQLToken(type=SQLTokenType.Column, field=SQLFieldType.Select, value=token['value'])]
            col_idx = idx

        elif token_type == 2:  # value
            assert col_idx >= 0 and idx - col_idx <= 3, expr_str
            column = ae_tokens[col_idx]['value'] if ae_tokens[col_idx]['token_type'] == 1 else '*'
            value_str = token['value'] if token['value'] is not None else "@null"
            value = "{}::{}".format(column, value_str)
            sql_tokens += [SQLToken(type=SQLTokenType.Value, field=SQLFieldType.Select, value=value)]

        elif token_type in [4, 5, 6]:  # LiteralString, LiteralNumber, LiteralDatetime
            # assert col_idx >= 0 and idx - col_idx <= 3, expr_str
            column = ae_tokens[col_idx]['value'] if ae_tokens[col_idx]['token_type'] == 1 else '*'
            value_str = token['value'] if token['value'] is not None else "@null"
            value = "{}::{}".format(column, value_str)
            sql_tokens += [SQLToken(type=SQLTokenType.Value, field=SQLFieldType.Select, value=value)]

        else:
            raise NotImplementedError()

    return SQLExpression(db_id=schema.db_id, tokens=sql_tokens, sql_dict=None)


# %%
def process_dataset(dataset: str):
    assert dataset in ['train', 'dev', 'test']
    tables = load_table_schemas(os.path.join(data_dir, f"{dataset}.tables.jsonl"))

    examples, unresolved_cnt = [], 0
    with open(os.path.join(data_dir, f'{dataset}.json'), 'r', encoding='utf-8') as fr:
        for line in tqdm(fr.readlines(), desc=f'Preprocessing {dataset}'):
            obj = json.loads(line)
            schema = tables[obj['table_id']]

            question = obj['question'].strip()
            question_tokens = [Token(token=x['token'], lemma=x['lemma']) for x in obj['question_tokens']]
            question_utterance = Utterance(text=question, tokens=question_tokens)
            sql = parse_ae_tokens(obj['answer_expression']['answer_tokens'], schema)

            matched_values = []
            if obj['matched_values'] is not None:
                for value_obj in obj['matched_values']:
                    value = CellValue(
                        name=value_obj['name'],
                        tokens=[Token(token=value_str, lemma=value_str.lower()) for value_str in value_obj['tokens']],
                        span=Span(start=value_obj['start'], end=value_obj['end']),
                        column=value_obj['column'],
                        score=value_obj['confidence']
                    )
                    matched_values += [value]

            erased_ngrams = generate_erased_ngrams(question_utterance, schema)

            example = Text2SQLExample(
                dataset='wikisql',
                question=question_utterance,
                schema=schema,
                sql=sql,
                matched_values=matched_values,
                erased_ngrams=erased_ngrams,
                value_resolved=obj['resolved']
            )

            if not example.value_resolved:
                # print(example.schema.db_id, example.question.text, str(example.sql))
                unresolved_cnt += 1

            examples += [example]
    save_json_objects(examples, os.path.join(data_dir, f'{dataset}.preproc.json'))
    print('proprocess {} examples over, size = {}, unresolved = {}'.format(dataset, len(examples), unresolved_cnt))


# %%
process_dataset('dev')
# %%
process_dataset('test')
# %%
process_dataset('train')
