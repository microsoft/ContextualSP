from logging import info
from typing import Dict, List
import multiprocessing as mp
from datetime import datetime

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizer

from contracts import *
from .data_encoder import DataEncoder


def multiprocessing_func(input_with_func):
    return input_with_func[1](input_with_func[0])


def multiprocessing_run(inputs: List, func, n_processes: int = None):
    n_processes = n_processes if n_processes is not None else mp.cpu_count()
    with mp.Pool(n_processes) as pool:
        mp_inputs = [(x, func) for x in inputs]
        mp_outputs = pool.map(multiprocessing_func, mp_inputs)
        return mp_outputs


class Text2SQLDataset(Dataset):
    def __init__(
            self,
            examples: List[Text2SQLExample],
            tokenizer: PreTrainedTokenizer,
            max_enc_length: int,
            sort_by_length: bool = False,
            n_processes: int = None
    ) -> None:

        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = DataEncoder(tokenizer=tokenizer)
        self.max_enc_length = min(max_enc_length, self.tokenizer.model_max_length)

        start = datetime.now()
        self._encoded_examples = self._encode_examples(examples, sort_by_length=sort_by_length, n_processes=n_processes)
        cost = (datetime.now() - start).seconds
        info('Setup Text2SQLDataset over, cost = {}s'.format(cost))

    def __len__(self) -> int:
        return len(self._encoded_examples)

    def __getitem__(self, index) -> Dict:
        example = self._encoded_examples[index]
        example['id'] = torch.tensor(index, dtype=torch.long)
        return example

    def get_example_by_id(self, idx: int) -> Dict:
        return self._encoded_examples[idx]

    def get_data_loader(self, batch_size: int, is_training: bool = True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=is_training,
            collate_fn=self._batch_collate
        )

    def _batch_collate(self, inputs):
        assert len(inputs) > 0, "collate inputs size must be greater than 0."
        collated = {}
        for key in inputs[0]:
            values = [x[key] for x in inputs]

            # Ignore all non-tensor values when collating
            if not isinstance(values[0], torch.Tensor):
                # continue
                collated[key] = values

            elif key == 'input_token_ids':
                collated[key] = pad_sequence(values, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            elif key.endswith('indices'):
                collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
            elif key.endswith('labels'):
                collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
            elif key.endswith('mask'):
                collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
            elif key == "id":
                collated[key] = torch.stack(values, dim=0)
            else:
                raise NotImplementedError(key)

        # collated['is_training'] = is_training
        return collated

    def _encode_example(self, example: Text2SQLExample) -> Dict:
        input_tokens, input_token_ids, indices, blocks = self.encoder.encode(
            example.question,
            example.schema,
            example.matched_values,
            lang=example.language,
        )

        if len(input_token_ids) > self.max_enc_length:
            return None

        raw_concept_labels = example.get_concept_labels()
        raw_question_labels = example.get_question_labels()
        column_labels = [x > 0 for x in raw_concept_labels[SQLTokenType.Column]]
        value_labels = [x > 0 for x in raw_concept_labels[SQLTokenType.Value]]
        keyword_labels = [x > 0 for x in raw_concept_labels[SQLTokenType.Keyword]]

        erased_indices = []
        for span in example.erased_ngrams:
            p_start, p_end = indices['question'][span.start].start, indices['question'][span.end].end
            erased_indices += [(span.start, span.end, p_start, p_end)]

        return {
            'example': example,
            'input_tokens': input_tokens,
            'input_token_ids': input_token_ids,
            'question_indices': indices['question'],
            'entity_indices': indices['column'] + indices['value'],
            'question_labels': raw_question_labels,
            'concept_labels': keyword_labels + column_labels + value_labels,
            'erased_indices': erased_indices,
            'blocks': blocks
        }

    def _to_tensor(self, data_input: Dict):
        def _convert_all_question_index_spans_to_tensor(spans: List[Span]) -> torch.Tensor:
            # drop last sep token
            index_list = [span.start for span in spans[:-1]]
            return torch.tensor(index_list, dtype=torch.long)

        def _convert_index_spans_to_tensor(spans: List[Span]) -> torch.Tensor:
            index_list = [span.start for span in spans]
            return torch.tensor(index_list, dtype=torch.long)

        def _convert_integers_to_tensor(values: List[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long)

        data_input['input_token_ids'] = _convert_integers_to_tensor(data_input['input_token_ids'])

        data_input['raw_question_indices'] = _convert_all_question_index_spans_to_tensor(data_input['question_indices'])
        data_input['raw_question_mask'] = torch.ones_like(data_input['raw_question_indices'], dtype=torch.bool)

        data_input['question_indices'] = _convert_index_spans_to_tensor(data_input['question_indices'])
        data_input['question_mask'] = torch.ones_like(data_input['question_indices'], dtype=torch.bool)

        data_input['entity_indices'] = _convert_index_spans_to_tensor(data_input['entity_indices'])
        data_input['entity_mask'] = torch.ones_like(data_input['entity_indices'], dtype=torch.bool)

        data_input['concept_labels'] = _convert_integers_to_tensor(data_input['concept_labels'])
        data_input['concept_mask'] = torch.ones_like(data_input['concept_labels'], dtype=torch.bool)

        data_input['question_labels'] = _convert_integers_to_tensor(data_input['question_labels'])
        data_input['question_labels_mask'] = torch.ones_like(data_input['question_labels'], dtype=torch.bool)

        return data_input

    def _encode_examples(self, examples: List[Text2SQLExample], sort_by_length: bool, n_processes: int = None):
        info("Encode {} examples with {} processes ...".format(len(examples), n_processes))
        if n_processes > 1:
            encoded_examples = multiprocessing_run(examples, self._encode_example, n_processes)
        else:
            encoded_examples = [self._encode_example(ex) for ex in examples]

        encoded_examples = [self._to_tensor(ex) for ex in encoded_examples if ex is not None]

        if sort_by_length:
            encoded_examples = list(sorted(encoded_examples, key=lambda x: len(x['input_token_ids'])))

        info("Ignore {} examples with encode length > {}".format(len(examples) - len(encoded_examples),
                                                                 self.max_enc_length))

        return encoded_examples


def tensor_collate_func(inputs: List[Dict], is_training: bool, pad_token_id=0) -> Dict:
    assert len(inputs) > 0
    collated = {}
    for key in inputs[0]:
        values = [x[key] for x in inputs]
        if not isinstance(values[0], torch.Tensor):
            collated[key] = values
        elif key == 'input_token_ids':
            collated[key] = pad_sequence(values, batch_first=True, padding_value=pad_token_id)
        elif key.endswith('indices'):
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key.endswith('labels'):
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key.endswith('mask'):
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        else:
            raise NotImplementedError(key)
    collated['is_training'] = is_training
    return collated


def to_device(inputs, device: torch.device):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)

    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]

    if isinstance(inputs, dict):
        return {key: to_device(val, device) for key, val in inputs.items()}

    return inputs


def load_data_loader(
        examples: List[Text2SQLExample],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        is_training: bool,
        max_enc_length: int,
        n_processes: int
) -> DataLoader:
    dataset = Text2SQLDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_enc_length=max_enc_length,
        sort_by_length=False,
        n_processes=n_processes
    )

    return dataset.get_data_loader(batch_size, is_training=is_training)
