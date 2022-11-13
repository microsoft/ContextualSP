# %%
import sys

sys.path.append("..")
from contracts import *
from utils.nlp_utils import *
from tqdm import tqdm

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


# %%
def get_matched_values(question: str, values: List[Dict], schema: DBSchema) -> Tuple[Utterance, List[CellValue]]:
    question_tokens = [Token(token=token['text'], lemma=token['Value']) for token in values['question_tokens']]
    question_utterance = Utterance(text=question, tokens=question_tokens)

    cell_values = []
    for raw_value in values['matched_values']:
        start, end = raw_value['start'], raw_value['end']
        column: Column = schema.identifier_map[raw_value['column']]
        val_name = raw_value['value']
        if column.data_type == DataType.Number:
            val_tokens = question_utterance.tokens[start:end + 1]
            val_name = "".join([x.token for x in val_tokens])
        else:
            val_tokens = [Token(raw_value['value'], lemma=raw_value['value'].lower())]
        value = CellValue(name=val_name, tokens=val_tokens, span=Span(start=start, end=end), column=raw_value['column'],
                          score=raw_value['confidence'])
        cell_values += [value]
    return question_utterance, cell_values


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
        ngram_spans.add((tok_idx, tok_idx))

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
            # assert col_idx >= 0 and idx - col_idx <= 3, expr_str
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
def parse_text2sql_example(example_obj: Dict, schema: DBSchema) -> Text2SQLExample:
    question = example_obj['question']
    question_tokens = [Token(token=x['token'], lemma=x['lemma']) for x in example_obj['question_tokens']]
    question_utterance = Utterance(text=question, tokens=question_tokens)

    sql = parse_ae_tokens(example_obj['answer_expression']['answer_tokens'], schema)

    matched_values = []
    if example_obj['matched_values'] is not None:
        for value_obj in example_obj['matched_values']:
            value = CellValue(
                name=value_obj['name'],
                tokens=[Token(token=value_str, lemma=value_str.lower()) for value_str in value_obj['tokens']],
                span=Span(start=value_obj['start'], end=value_obj['end']),
                column=value_obj['column'],
                score=value_obj['confidence']
            )

            if len(value.column) > 0:
                matched_values += [value]

    erased_ngrams = generate_erased_ngrams(question_utterance, schema)

    example = Text2SQLExample(
        dataset='wikitq',
        question=question_utterance,
        schema=schema,
        sql=sql,
        matched_values=matched_values,
        erased_ngrams=erased_ngrams,
        value_resolved=example_obj['resolved'] and len(sql.tokens) > 0
    )

    return example


# %%
def preprocess_data(mode: str):
    data_dir = os.path.join(Proj_Abs_Dir, 'data', 'wikitq')
    schemas = load_table_schemas(os.path.join(data_dir, f'{mode}.tables.json'))
    preprocessed_path = os.path.join(data_dir, f'{mode}.preproc.json')

    unresolved_cnt = 0
    with open(os.path.join(data_dir, f'{mode}.json'), 'r', encoding='utf-8') as fr:
        examples = []
        for line in tqdm(fr.readlines(), desc='Preprocess {}'.format(mode)):
            try:
                obj = json.loads(line)
                schema = schemas[obj['table_id']]
                example = parse_text2sql_example(obj, schema)

                if not example.value_resolved:
                    # print("Database:", example.schema.db_id, ", Question:", example.question.text, ", SQL: ", str(example.sql))
                    unresolved_cnt += 1
                examples += [example]

            except Exception as e:
                print("Exception: {}".format(str(e)))

        save_json_objects(examples, preprocessed_path, ensure_ascii=False)
        print('proprocess {} examples over, size = {}, unresolved = {}/{:.2f}%'.format(mode, len(examples),
                                                                                       unresolved_cnt,
                                                                                       unresolved_cnt * 100.0 / len(
                                                                                           examples)))
    pass


# %%
preprocess_data('dev')
# %%
# preprocess_data('test')
# %%
preprocess_data('train')

# %%
import shutil

print('copy dev as test for squall')
shutil.copyfile(
    os.path.join(Proj_Abs_Dir, 'data', 'squall', 'dev.preproc.json'),
    os.path.join(Proj_Abs_Dir, 'data', 'squall', 'test.preproc.json')
)
