import os
import argparse
import torch
import json
from models import *
from utils import *
from tqdm import tqdm

def load_model_and_data_iter(args):
    ckpt_path = args.checkpoint
    device = torch.device(args.device)
    config = json.load(open(os.path.join(os.path.dirname(ckpt_path), 'config.json'), 'r', encoding='utf-8'))
    config['checkpoint'] = ckpt_path
    config['device'] = device

    model = load_model_from_checkpoint(**config)
    print('load model from {} over.'.format(ckpt_path))
    model.eval()

    print('-------------------Config-------------------')
    for key, val in config.items():
        print(key, val)
    print('load {} from {} over .'.format(config['model'], ckpt_path))
    
    bert_version = config['bert_version']
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    print('load {} tokenizer over'.format(bert_version))

    return config, model, tokenizer

# load schemas from database
def get_column_names_unique(column_names: List[Tuple[int, str]], table_names: List[str], primary_keys: List[int]) -> List[str]:
    column_names_dict = defaultdict(int)
    for tbl_idx, col_name in column_names:
        column_names_dict[col_name] += 1
    
    column_names_unique = []
    for c_idx, (tbl_idx, col_name) in enumerate(column_names):
        if tbl_idx == -1:
            column_names_unique.append(col_name)
            continue

        if column_names_dict[col_name] == 1:
            column_names_unique.append(col_name)
        elif c_idx in primary_keys:
            column_names_unique.append(col_name)
        else:
            tbl_name = table_names[tbl_idx]
            full_name = '{} . {}'.format(tbl_name, col_name)
            column_names_unique.append(full_name)
    assert len(column_names_unique) == len(column_names)
    return column_names_unique

def alt_tbl_name(tbl_name):
    tbl_name = tbl_name.split()
    if len(tbl_name) > 1 and tbl_name[0] == 'reference':
        tbl_name = tbl_name[1:]
    if len(tbl_name) > 1 and tbl_name[-1] == 'data':
        tbl_name = tbl_name[:-1]
    if len(tbl_name) > 1 and tbl_name[-1] == 'list':
        tbl_name = tbl_name[:-1]
    return ' '.join(tbl_name)

def remove_shared_prefix(col_name: str, tbl_name: str) -> str:
    col_tokens, tbl_tokens = col_name.split(), tbl_name.split()
    idx = 0
    while idx < len(col_tokens) and idx < len(tbl_tokens) and col_tokens[idx] == tbl_tokens[idx]:
        idx += 1
    return " ".join(col_tokens[idx:])

def get_column_name_normalized(column_lem_names: List[Tuple[int, str]], table_lem_names: List[str], verbose: bool = False):
    column_norm_names, table_norm_names = [], []
    for tbl_name in table_lem_names:
        table_norm_names.append(alt_tbl_name(tbl_name))

    for col_idx, (tbl_idx, col_name) in enumerate(column_lem_names):
        if col_name == '*':
            column_norm_names.append('*')
            continue
        col_norm_name = remove_shared_prefix(col_name, table_norm_names[tbl_idx])
        if col_norm_name != col_name and verbose:
            logging.info(" {}\t{}\t{}".format(table_norm_names[tbl_idx], col_name, col_norm_name))
        column_norm_names.append(col_norm_name)

    return column_norm_names, table_norm_names

def load_schema(obj: Dict) -> SpiderSchema:
    column_names_lemma = obj['column_names_lemma']
    table_names_lemma = obj['table_names_lemma']
    column_names_original = [x[1] for x in obj['column_names_original']]
    column_to_table, table_to_columns = {}, {}
    for col_idx, (tbl_idx, _) in enumerate(obj['column_names']):
        if tbl_idx not in table_to_columns:
            table_to_columns[tbl_idx] = []
        table_to_columns[tbl_idx].append(col_idx)
        column_to_table[col_idx] = tbl_idx
    
    col_norm_names, tbl_norm_names = get_column_name_normalized(column_names_lemma, table_names_lemma, True)
    return SpiderSchema(
        db_id=obj['db_id'],
        column_names=col_norm_names,
        column_types=obj['column_types'],
        column_names_lemma=[x[1] for x in column_names_lemma],
        column_names_original=column_names_original,
        table_names=tbl_norm_names,
        table_names_lemma=table_names_lemma,
        table_names_original=obj['table_names_original'],
        table_to_columns=table_to_columns,
        column_to_table=column_to_table,
        primary_keys=obj['primary_keys'],
        foreign_keys=obj['foreign_keys'])

def load_schemas(path: str):
    databases = json.load(open(path, 'r', encoding='utf-8'))
    schemas = {}
    for database in databases:
        schema = load_schema(database)
        schemas[schema.db_id] = schema
    return schemas

def process_examples(input_path: str, database_dir: str, tokenizer: BertTokenizer, table_path: str, output_path):
    schemas = load_schemas(table_path)
    print('load schemas over.')

    value_matchers = load_value_matchers(database_dir, schemas)
    print('load value matchers over')

    processed_examples = []
    for raw_example in tqdm(json.load(open(input_path, 'r', encoding='utf-8'))):
        db_id = raw_example['db_id']
        assert db_id in schemas
        processed_example = process_slsql_example(raw_example, tokenizer, schemas[db_id], value_matchers[db_id])
        processed_examples += [processed_example]
    
    save_json_objects(processed_examples, output_path)
    print('process examples over, save into {}'.format(output_path))

def fix_tok(tok):
    tok = tok.lower()
    if tok == '-lrb-':
        tok = '('
    elif tok == '-rrb-':
        tok = ')'
    elif tok == '\"':
        tok = '\''
    return tok

spider_type_mappings = {
    'text': 'text', 
    'time': 'time',
    'number': 'number',
    'boolean': 'boolean',
    'others': 'text'
    }

def get_data_type(db_data_type: str):
    if db_data_type.startswith("int") or db_data_type.startswith("bigint") or db_data_type.startswith("mediumint"):
        return "int"
    if db_data_type.startswith("smallint") or db_data_type.startswith("tinyint") or db_data_type.startswith("bit") or db_data_type.startswith("bool") :
        return "int"

    if db_data_type.startswith("real") or db_data_type.startswith("numeric") or db_data_type.startswith("number"):
        return "real"
    
    if db_data_type.startswith("double") or db_data_type.startswith("decimal") or db_data_type.startswith("float"):
        return "real"

    if db_data_type.startswith("text") or db_data_type.startswith("varchar") or db_data_type.startswith("char"):
        return "text"
    if db_data_type.startswith("timestamp") or db_data_type.startswith("date") or db_data_type.startswith("year"):
        return "datetime"
    
    if len(db_data_type) == 0 or db_data_type.startswith("blob"):
        return "text"

    return 'text'
    #raise ValueError("not support data type: " + db_data_type)

def get_column_with_values(path: str):
    column_values = defaultdict(list)
    try:
        conn = sqlite3.connect(path)
        conn.text_factory = lambda b: b.decode(errors = 'ignore')
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables =[x[0] for x in cur.fetchall()]
        for table in tables:
            col_results = cur.execute("PRAGMA table_info('%s')" % table).fetchall()
            columns = []
            for col in col_results:
                col_name = col[1]
                data_type = get_data_type(col[2].lower())
                columns.append((col_name, data_type))

            assert len(columns) > 0
            # get rows
            cur.execute("SELECT * FROM " + table + ";")
            row_results = cur.fetchall()
            rows = []
            for row in row_results:
                assert len(row) == len(columns)
                rows.append(row)
            
            for i, (col_name, col_type) in enumerate(columns):
                values = [row[i] for row in rows]
                unique_name = '{}.{}'.format(table, col_name).lower()
                column_values[unique_name] = (unique_name, col_type, values)
    except:
        pass

    return column_values.values()

def load_value_matchers(database_dir: str, schemas: Dict[str, SpiderSchema]):
    db_matchers = {}
    for schema in schemas.values():
        db_id = schema.db_id
        column_with_values = get_column_with_values(os.path.join(database_dir, db_id, f'{db_id}.sqlite'))
        db_matchers[db_id] = ValueMatcher(column_with_values)
    return db_matchers

def process_slsql_example(query: Dict, tokenizer: BertTokenizer, schema: SpiderSchema, value_matcher: ValueMatcher) -> Dict:
    question = query['question']
    assert len(query['toks']) == len(query['lemma'])
    question_utterance = generate_utterance(tokenizer, question, [fix_tok(x) for x in query['toks']], [fix_tok(x) for x in query['lemma']])

    # Step 2: process tables & columns   
    processed_tables = []
    for tbl_idx, col_indices in schema.table_to_columns.items():
        # special column *
        if tbl_idx == -1:
            table_json = {
                'index': -1,
                'utterance': Utterance('*', tokens=[]).to_json(),
                'columns': None
                }
            processed_tables += [table_json]
            continue
        tbl_name = schema.table_names[tbl_idx]
        table_utterance = generate_utterance(tokenizer, tbl_name)

        processed_columns = []
        for col_idx in col_indices:
            column_type = schema.column_types[col_idx]
            column_utterance = generate_utterance(tokenizer, schema.column_names[col_idx])
            column_json = {
                'index': col_idx,
                'utterance': column_utterance.to_json(),
                'data_type': spider_type_mappings.get(column_type, 'text')
                }
            processed_columns += [column_json]
        
        table_json = {
            'index': tbl_idx,
            'utterance': table_utterance.to_json(),
            'columns': processed_columns
            }
        processed_tables += [table_json]

    matched_values = value_matcher.match(question_utterance.text_tokens, 0.8, 3)

    processed_query = {
        'question': question_utterance.to_json(),
        'tables': processed_tables,
        'schema': schema.to_json(),
        'values': [v.to_json() for v in matched_values]
        }

    return processed_query

def predict_alignments(model: nn.Module, data_iter: DataLoader, saved_path: str, threshold: float):
    slsql_align_labels = []
    model.eval()
    with torch.no_grad():
        for model_input in data_iter:
            model_output = model(**model_input)
            example = model_input['example'][0]
            meta_index: MetaIndex = model_input['meta_index'][0]
            question: Utterance = Utterance.from_json(example['question'])
            schema: SpiderSchema = SpiderSchema.from_json(example['schema'])
            values = [ValueMatch.from_json(v) for v in example['values']]

            identify_logits = { SQLTokenType.table: model_output['table_logits'][0], SQLTokenType.column: model_output['column_logits'][0], SQLTokenType.value: model_output['value_logits'][0] }
            tbl_align_weights, col_align_weights, val_align_weights = meta_index.split(model_output['alignment_weights'][0])
            align_weights = { SQLTokenType.table: tbl_align_weights, SQLTokenType.column: col_align_weights, SQLTokenType.value: val_align_weights }
            
            pred_align_labels = greedy_link_spider(identify_logits, align_weights, question, schema, values, threshold=threshold)
            assert len(pred_align_labels) == len(question.tokens)

            sql_align_label = [label.to_slsql(schema) for label in pred_align_labels]
            slsql_align_labels += [sql_align_label]
    
    save_json_objects(slsql_align_labels, saved_path)
    print('predict alignments over, saved into {}'.format(saved_path))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', default='baseline/slsql/codalab/saved_models_large/align_model.bin')
    parser.add_argument('-data', '--data_dir', default='baseline/slsql/codalab/dev_data')
    parser.add_argument('-db_dir', '--database_dir', default='baseline/slsql/data/database')
    parser.add_argument('-threshold', '--threshold', default=0.4, type=float)
    parser.add_argument('-output_dir', '--output_dir', default='output')
    parser.add_argument('-gpu', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config, model, tokenizer = load_model_and_data_iter(args)
    print('loading data iterator ...')

    processed_path = os.path.join(args.output_dir, 'dev.val_processed.json')
    process_examples(
        input_path=os.path.join(args.output_dir, 'dev.processed.json'),
        table_path=os.path.join(args.output_dir, 'tables.processed.json'),
        database_dir=args.database_dir,
        tokenizer=tokenizer,
        output_path=processed_path
    )

    data_iter = get_data_iterator_func(config['model'])(processed_path, tokenizer, 1, config['device'], False, False, 512, None)

    predict_alignments(model, data_iter, os.path.join(args.output_dir, 'dev.align.json'), args.threshold)
    print('Run Alignment Over')