import math
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler, SubsetRandomSampler
from logging import warning, info
from utils.data_types import *
from utils.nlp_utils import *
from tqdm import tqdm


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
    Example:
        >>> list(SortedSampler(range(10)))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.sort_key = lambda x: x
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.
    """

    def __init__(self, sampler, batch_size, drop_last, bucket_size_multiplier=100) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.bucket_sampler = BatchSampler(sampler,
                                           min(batch_size * bucket_size_multiplier, len(sampler)),
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket)
            for batch in SubsetRandomSampler(list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


@dataclass
class MetaIndex:
    question_spans: List[Tuple[Token, int, int]]  # (token, start, end)
    entity_spans: List[Tuple[str, int, int, int]]  # (type, entity_id, start, end)

    column2table_indices: List[int]  # table index for each column

    table_spans: List[Tuple[int, int, int]] = field(init=False)
    column_spans: List[Tuple[int, int, int]] = field(init=False)
    value_spans: List[Tuple[int, int, int]] = field(init=False)

    def __post_init__(self):
        self.table_spans = [(e_id, start, end) for e_type, e_id, start, end in self.entity_spans if e_type == 'tbl']
        self.column_spans = [(e_id, start, end) for e_type, e_id, start, end in self.entity_spans if e_type == 'col']
        self.value_spans = [(e_id, start, end) for e_type, e_id, start, end in self.entity_spans if e_type == 'val']

    @property
    def num_question_tokens(self):
        return len(self.question_spans)

    @property
    def num_columns(self):
        return len(self.col_encode_indices)

    @property
    def num_values(self):
        return len(self.val_encode_indices)

    @property
    def num_tables(self):
        return len(self.tbl_encode_indices)

    @property
    def question_sep_index(self):
        return self.question_spans[-1][1] + 1

    @property
    def question_encode_indices(self) -> List[int]:
        return [start for _, start, _ in self.question_spans]

    @property
    def tbl_encode_indices(self) -> List[int]:
        return [start for _, start, _ in self.table_spans]

    @property
    def col_encode_indices(self) -> List[int]:
        return [start for _, start, _ in self.column_spans]

    @property
    def val_encode_indices(self) -> List[int]:
        return [start for _, start, _ in self.value_spans]

    @property
    def col_tbl_encode_indices(self) -> List[int]:
        col_tbl_enc_indices = []
        for col_idx, tbl_idx in enumerate(self.column2table_indices):
            if tbl_idx == -1:
                col_tbl_enc_indices.append(self.column_spans[col_idx][1])
            else:
                col_tbl_enc_indices.append(self.table_spans[tbl_idx][1])

        assert len(col_tbl_enc_indices) == len(self.column_spans)
        return col_tbl_enc_indices

    def split(self, outputs, dim=0):
        return torch.split(outputs, [self.num_tables, self.num_columns, self.num_values], dim=dim)

    def lookup_entity_id(self, e_type: str, type_encode_idx: int) -> int:
        count = 0
        for t, e_idx, _, _ in self.entity_spans:
            if t != e_type:
                continue
            if count == type_encode_idx:
                return e_idx
            count += 1
        raise ValueError("Index {} of range for type {}".format(type_encode_idx, e_type))


class WTQDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: BertTokenizer, device: torch.device, max_enc_length: int,
                 sort_by_length: bool) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.max_enc_length = max_enc_length
        self.examples = self._encode_examples(examples, sort_by_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        return self.examples[index]

    def _encode_examples(self, examples: List[Dict], sort_by_length: bool) -> List[Dict]:
        new_examples = []
        t_examples = tqdm(examples) if len(examples) > 100 else examples
        for example in t_examples:
            new_example = self._encode_example(example)
            if new_example is None:
                continue
            new_examples += [new_example]

        if len(new_examples) < len(examples):
            warning('Ignore {} examples >= max encode length {}'.format(len(examples) - len(new_examples),
                                                                        self.max_enc_length))

        if not sort_by_length:
            return new_examples

        sorted_examples = sorted(new_examples, key=lambda x: x['input_token_ids'].size(0))
        return list(sorted_examples)

    def _encode_example(self, example: Dict) -> Dict:
        question: Utterance = Utterance.from_json(example['question'])
        input_tokens = [self.tokenizer.cls_token]
        question_spans = []
        for token in question.tokens:
            start = len(input_tokens)
            input_tokens += token.pieces
            question_spans += [(token, start, len(input_tokens) - 1)]
        input_tokens += [self.tokenizer.sep_token]
        input_token_types = [0] * len(input_tokens)

        schema: WTQSchema = WTQSchema.from_json(example['schema'])
        entity_spans = []
        for c_idx, column in enumerate(example['columns']):
            column_utterance: Utterance = Utterance.from_json(column['utterance'])
            start = len(input_tokens)
            input_tokens += [Bert_Special_Tokens[column['data_type']]] + column_utterance.pieces
            assert column['index'] == c_idx
            entity_spans += [('col', column['index'], start, len(input_tokens) - 1)]
            input_tokens += [self.tokenizer.sep_token]

        if len(input_tokens) > self.max_enc_length:
            return None

        input_token_types += [1] * (len(input_tokens) - len(input_token_types))
        input_token_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        assert len(input_tokens) == len(input_token_types)

        meta_index = MetaIndex(question_spans=question_spans, entity_spans=entity_spans, column2table_indices=None)
        column_labels = [0 for _ in entity_spans]
        assert len(entity_spans) == len(schema.column_headers)
        if 'identify_labels' in example:
            for col_name in example['identify_labels'][str(SQLTokenType.column)]:
                col_id = schema.column_header_to_id[col_name]
                column_labels[col_id] = 1

        return {
            'input_token_ids': torch.tensor(input_token_ids, dtype=torch.long, device=self.device),
            'input_token_types': torch.tensor(input_token_types, dtype=torch.long, device=self.device),
            'column_labels': torch.tensor(column_labels, dtype=torch.long, device=self.device),
            'meta_index': meta_index,
            'input_tokens': input_tokens,
            'example': example
        }


class SpiderDataset(WTQDataset):
    def __init__(self, examples: List[Dict], tokenizer: BertTokenizer, device: torch.device, max_enc_length: int,
                 sort_by_length: bool) -> None:
        super().__init__(examples, tokenizer, device, max_enc_length, sort_by_length)

    def fix_span(self, span: Tuple[str, int, int, int]):
        start, end = span[2], span[3]
        if start >= self.max_enc_length:
            start = 0
        if end >= self.max_enc_length:
            end = 0

        return (span[0], span[1], start, end)

    def get_matched_values(self, example: Dict, schema: SpiderSchema, threshold: float = 0.82) -> List[ValueMatch]:
        values = [ValueMatch.from_json(v) for v in example['values']]
        filter_values = []
        for value in values:
            if value.score < threshold and value.score > 0.51:  # 0.5 is matched by sub string
                if value.label:
                    warning('Ignore gold value with confidence < {}, {} ({})'.format(threshold, str(value),
                                                                                     example['question']['text']))
                continue
            filter_values.append(value)

        sorted_values = list(sorted(filter_values, key=lambda v: schema.id_map[v.column]))

        example['values'] = [v.to_json() for v in sorted_values]
        return sorted_values

    def build_relations(self, schema: SpiderSchema, values: List[ValueMatch]):
        col_idx_offset, val_idx_offset = schema.num_tables, schema.num_tables + schema.num_columns

        relations = {}

        def update_relation(i, j, r):
            if (i, j) not in relations:
                relations[(i, j)] = int(r)

        # primary key
        for col_idx in schema.primary_keys:
            tbl_idx = schema.column_to_table[col_idx]
            update_relation(tbl_idx, col_idx_offset + col_idx, SchemaRelation.table_column_pk)
            update_relation(col_idx_offset + col_idx, tbl_idx, SchemaRelation.column_table_pk)

        # foreign key
        for col_idx1, col_idx2 in schema.foreign_keys:
            update_relation(col_idx1 + col_idx_offset, col_idx2 + col_idx_offset, SchemaRelation.column_column_fk_fw)
            update_relation(col_idx2 + col_idx_offset, col_idx1 + col_idx_offset, SchemaRelation.column_column_fk_bw)

        for col_idx in range(schema.num_columns):
            tbl_idx = schema.column_to_table[col_idx]
            update_relation(tbl_idx, col_idx_offset + col_idx, SchemaRelation.table_column)
            update_relation(col_idx_offset + col_idx, tbl_idx, SchemaRelation.column_table)

        for tbl_idx in range(schema.num_tables):
            for tbl_idx2 in range(schema.num_tables):
                update_relation(tbl_idx, tbl_idx2, SchemaRelation.table_table)

            col_indices = schema.table_to_columns
            for col_idx1 in col_indices:
                for col_idx2 in col_indices:
                    update_relation(col_idx1 + col_idx_offset, col_idx2 + col_idx_offset, SchemaRelation.column_column)

        for val_idx in range(len(values)):
            col_idx = schema.id_map[values[val_idx].column]
            update_relation(col_idx + col_idx_offset, val_idx + val_idx_offset, SchemaRelation.column_value)
            update_relation(val_idx + val_idx_offset, col_idx + col_idx_offset, SchemaRelation.value_column)

        relations_tensor = torch.zeros((schema.num_tables + schema.num_columns + len(values),
                                        schema.num_tables + schema.num_columns + len(values)), dtype=torch.long,
                                       device=self.device)
        for (i, j), r in relations.items():
            relations_tensor[i, j] = r

        return relations_tensor

    def _encode_example(self, example: Dict) -> Dict:
        question: Utterance = Utterance.from_json(example['question'])
        input_tokens = [self.tokenizer.cls_token]
        question_spans = []
        for token in question.tokens:
            start = len(input_tokens)
            input_tokens += token.pieces
            question_spans += [(token, start, len(input_tokens) - 1)]
        input_tokens += [self.tokenizer.sep_token]
        input_token_types = [0] * len(input_tokens)

        assert len(input_tokens) < self.max_enc_length

        schema: SpiderSchema = SpiderSchema.from_json(example['schema'])
        values = self.get_matched_values(example, schema)
        grouped_values = defaultdict(list)
        for i, value in enumerate(values):
            column_idx = schema.id_map[value.column]
            grouped_values[column_idx].append(i)

        entity_spans, idx2spans = [], {}
        for table in example['tables']:
            table_utterance: Utterance = Utterance.from_json(table['utterance'])
            if table_utterance.text == '*':
                start = len(input_tokens)
                input_tokens += [Bert_Special_Tokens['*']]
                idx2spans[('col', 0)] = len(entity_spans)
                entity_spans += [('col', 0, start, len(input_tokens) - 1)]

                for value_idx in grouped_values[0]:
                    start = len(input_tokens)
                    input_tokens += [Col_Val_Sep]
                    input_tokens += self.tokenizer.tokenize(str(values[value_idx].value))
                    idx2spans[('val', value_idx)] = len(entity_spans)
                    entity_spans += [('val', value_idx, start, len(input_tokens) - 1)]

                input_tokens += [self.tokenizer.sep_token]
                continue

            start = len(input_tokens)
            input_tokens += [TBL_Token] + table_utterance.pieces
            idx2spans[('tbl', table['index'])] = len(entity_spans)
            entity_spans += [('tbl', table['index'], start, len(input_tokens) - 1)]

            for column in table['columns']:
                column_utterance: Utterance = Utterance.from_json(column['utterance'])
                start = len(input_tokens)
                col_db_key = schema.get_column_key_code(column['index'])
                input_tokens += [Tbl_Col_Sep, DB_Col_Keys[col_db_key], column['data_type']]

                # If column name is not unique, append table name 
                # assert column_utterance.text.lower() in column2ids
                # if len(column2ids[column_utterance.text.lower()]) > 1 and col_db_key == 0:
                #     input_tokens += table_utterance.pieces

                col_pieces = column_utterance.pieces
                if len(col_pieces) == 0:  # column share same same with table
                    col_pieces = table_utterance.pieces

                input_tokens += col_pieces
                idx2spans[('col', column['index'])] = len(entity_spans)
                entity_spans += [('col', column['index'], start, len(input_tokens) - 1)]

                for value_idx in grouped_values[column['index']]:
                    start = len(input_tokens)
                    input_tokens += [Col_Val_Sep]
                    input_tokens += self.tokenizer.tokenize(str(values[value_idx].value))
                    idx2spans[('val', value_idx)] = len(entity_spans)
                    entity_spans += [('val', value_idx, start, len(input_tokens) - 1)]

            input_tokens += [self.tokenizer.sep_token]

        if len(input_tokens) > self.max_enc_length:
            # warning("Length out of max: {}\t{}".format(schema.db_id, question.text))
            input_tokens = input_tokens[:self.max_enc_length]

        input_token_types += [1] * (len(input_tokens) - len(input_token_types))
        input_token_ids = self.tokenizer.convert_tokens_to_ids(
            [x if x not in Bert_Special_Tokens else Bert_Special_Tokens[x] for x in input_tokens])

        ordered_spans = []
        for tbl_idx in range(len(schema.table_names_original)):
            ordered_spans.append(self.fix_span(entity_spans[idx2spans[('tbl', tbl_idx)]]))
        for col_idx in range(len(schema.column_names_original)):
            ordered_spans.append(self.fix_span(entity_spans[idx2spans[('col', col_idx)]]))
        for val_idx in range(len(values)):
            ordered_spans.append(self.fix_span(entity_spans[idx2spans[('val', val_idx)]]))

        column2table = [schema.column_to_table[c_idx] for c_idx in range(len(schema.column_names))]
        meta_index = MetaIndex(question_spans=question_spans, entity_spans=ordered_spans,
                               column2table_indices=column2table)
        assert meta_index.num_tables == len(schema.table_names)
        assert meta_index.num_columns == len(schema.column_names)

        table_labels = [0] * meta_index.num_tables
        column_labels = [0] * meta_index.num_columns
        if 'identify_labels' in example:
            for table_name in example['identify_labels'][str(SQLTokenType.table)]:
                table_idx = schema.id_map[table_name.lower()]
                assert table_idx < meta_index.num_tables
                table_labels[table_idx] = 1

            for column_name in example['identify_labels'][str(SQLTokenType.column)]:
                column_idx = schema.id_map[column_name.lower()]
                assert column_idx < meta_index.num_columns
                column_labels[column_idx] = 1

        value_labels = [0] * meta_index.num_values
        for i, value in enumerate(values):
            value_labels[i] = int(value.label)

        # relations = self.build_relations(schema, values)

        return {
            'input_token_ids': torch.tensor(input_token_ids, dtype=torch.long, device=self.device),
            'input_token_types': torch.tensor(input_token_types, dtype=torch.long, device=self.device),
            'table_labels': torch.tensor(table_labels, dtype=torch.long, device=self.device),
            'column_labels': torch.tensor(column_labels, dtype=torch.long, device=self.device),
            'value_labels': torch.tensor(value_labels, dtype=torch.long, device=self.device),
            # 'relations': relations,
            'meta_index': meta_index,
            'input_tokens': input_tokens,
            'example': example
        }


def tensor_collate_fn(inputs: List[Dict], is_training: bool) -> Dict:
    assert len(inputs) > 0
    collated = {}
    for key in inputs[0]:
        values = [x[key] for x in inputs]
        if key == 'input_token_ids':
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key == 'input_token_types':
            collated[key] = pad_sequence(values, batch_first=True, padding_value=1)
        else:
            collated[key] = values
    collated['is_training'] = is_training
    return collated


def load_wtq_data_iterator(paths, tokenizer: BertTokenizer, batch_size: int, device: torch.device,
                           bucket: bool, shuffle: bool, max_enc_length: int, sampling_size: int = None) -> DataLoader:
    all_examples = []
    if isinstance(paths, list):
        for path in paths:
            examples = json.load(open(path, 'r', encoding='utf-8'))
            if sampling_size is not None:
                all_examples += examples[:sampling_size]
                info("Sampling {}/{} examples from {} over.".format(sampling_size, len(examples), path))
            else:
                all_examples += examples
                info("Load {} examples from {} over.".format(len(examples), path))
    elif isinstance(paths, str):
        all_examples = json.load(open(paths, 'r', encoding='utf-8'))
        if sampling_size is not None:
            info("Sampling {}/{} examples from {} over.".format(sampling_size, len(all_examples), paths))
            all_examples += all_examples[:sampling_size]
        else:
            info("Load {} examples from {} over.".format(len(all_examples), paths))
    else:
        raise ValueError("Invalid path input: {}".format(paths))

    if bucket:
        dataset = WTQDataset(all_examples, tokenizer, device, max_enc_length, True)
        data_loader = DataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(SequentialSampler(list(range(len(dataset)))), batch_size=batch_size,
                                             drop_last=False),
            collate_fn=lambda x: tensor_collate_fn(x, shuffle))

        return data_loader
    else:
        dataset = WTQDataset(all_examples, tokenizer, device, max_enc_length, False)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: tensor_collate_fn(x, shuffle))

        return data_loader


def load_spider_data_iterator(paths: List[str], tokenizer: BertTokenizer, batch_size: int, device: torch.device,
                              bucket: bool, shuffle: bool, max_enc_length: int,
                              sampling_size: int = None) -> DataLoader:
    all_examples = []
    if isinstance(paths, list):
        for path in paths:
            examples = json.load(open(path, 'r', encoding='utf-8'))
            if sampling_size is not None:
                all_examples += examples[:sampling_size]
                info("Sampling {}/{} examples from {} over.".format(sampling_size, len(examples), path))
            else:
                all_examples += examples
                info("Load {} examples from {} over.".format(len(examples), path))
    elif isinstance(paths, str):
        all_examples = json.load(open(paths, 'r', encoding='utf-8'))
        if sampling_size is not None:
            info("Sampling {}/{} examples from {} over.".format(sampling_size, len(all_examples), paths))
            all_examples += all_examples[:sampling_size]
        else:
            info("Load {} examples from {} over.".format(len(all_examples), paths))
    else:
        raise ValueError("Invalid path input: {}".format(paths))

    if bucket:
        dataset = SpiderDataset(all_examples, tokenizer, device, max_enc_length, True)
        data_loader = DataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(SequentialSampler(list(range(len(dataset)))), batch_size=batch_size,
                                             drop_last=False),
            collate_fn=lambda x: tensor_collate_fn(x, shuffle))

        return data_loader
    else:
        dataset = SpiderDataset(all_examples, tokenizer, device, max_enc_length, False)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: tensor_collate_fn(x, shuffle))

        return data_loader
