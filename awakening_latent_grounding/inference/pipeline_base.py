"""
NLBinding pipeline: tokenization, model prediction, post-prcessing.
"""
import abc
import os
import json
import traceback
from typing import List, Dict
from datetime import datetime
from transformers import AutoTokenizer

from .bind_types import StatusCode, NLBindingRequest, NLBindingResult, NLBindingTermResult, NLBindingType, NLModelError
from .data_encoder import NLDataEncoder
from .greedy_linker import greedy_link

_All_Agg_Op_Keywords = ["Max", "Min", "Sum", "Avg", "Count", "!=", ">", ">=", "<", "<=" ]

class NLBindingInferencePipeline:
    def __init__(self, model_dir:str, greedy_linking: bool, threshold: float=0.2) -> None:
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.data_encoder = NLDataEncoder(self.tokenizer, spm_phrase_path=os.path.join(model_dir, "spm.phrase.txt"))
        self.threshold = threshold
        self.greedy_linking = greedy_linking

    @abc.abstractmethod
    def run_model(self, **inputs) -> Dict[str, List]:
        raise NotImplementedError()

    def predict(self, data: str) -> str:
        start = datetime.now()

        try:
            try:
                request_obj = json.loads(data)
            except: # pylint: disable=bare-except
                raise NLModelError( # pylint: disable=raise-missing-from
                    error_code=StatusCode.invalid_input,
                    message="Input String Json Decoder Failed: {}".format(traceback.format_exc())
                )

            try:
                request = NLBindingRequest.from_json(request_obj)
            except:
                raise NLModelError( # pylint: disable=raise-missing-from
                    error_code=StatusCode.invalid_input,
                    message="Request Json Deserialization Failed: {}".format(traceback.format_exc())
                )

            result = self.infer(request)

        except NLModelError as e:
            result = NLBindingResult(
                status_code=e.error_code,
                message=e.message,
                inference_ms=(datetime.now() - start).microseconds // 1000,
                term_results=[],
                binding_tokens=None
            )

        except: # pylint: disable=bare-except
            result = NLBindingResult(
                status_code=StatusCode.internal_error,
                message="Unknown Internal Error: {}".format(traceback.format_exc()),
                inference_ms=(datetime.now() - start).microseconds // 1000,
                term_results=[],
                binding_tokens=None
            )

        return json.dumps(result.to_json())

    def infer(self, request: NLBindingRequest) -> NLBindingResult:
        start = datetime.now()

        model_inputs = self.encode_request(request)
        model_outputs = self.run_model(**model_inputs)
        term_results = self.get_term_results(model_outputs, request)

        binding_tokens = None
        if self.greedy_linking:
            binding_tokens = greedy_link(request=request, term_results=term_results, threshold=self.threshold)

        inference_ms = (datetime.now() - start).microseconds // 1000
        return NLBindingResult(
            status_code=StatusCode.succeed,
            message=None,
            inference_ms=inference_ms,
            term_results=term_results,
            binding_tokens=binding_tokens
        )

    def encode_request(self, request: NLBindingRequest):
        _, input_token_ids, spm_idx_mappings, indices = \
            self.data_encoder.encode(request.question_tokens, request.columns, request.matched_values, request.language)

        if len(input_token_ids) > self.tokenizer.model_max_length:
            raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: Input Token Length Out of {}".format(self.tokenizer.model_max_length),
                )

        if len(indices['question']) == 0 :
            raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: Empty Question"
                )

        if len(indices['column']) == 0:
            raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: Empty Columns"
                )

        for span in indices['value']:
            if span is None:
                raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: None Value Span"
                )

        column_indices = [start for start, _ in indices['column']]
        value_indices = [start for start, _ in indices['value']]
        question_indices = [start for start, _ in indices['question']]

        return {
            'input_token_ids': input_token_ids,
            'entity_indices': column_indices + value_indices,
            'question_indices': question_indices,
            'spm_idx_mappings': spm_idx_mappings,
        }

    def get_term_results(self, model_outputs, request: NLBindingRequest):
        cp_scores, grounding_scores = model_outputs['cp_scores'], model_outputs['grounding_scores']
        if len(cp_scores) != len(grounding_scores) or len(request.question_tokens) != len(grounding_scores[0]):
            raise NLModelError(
                    error_code= StatusCode.invalid_output,
                    message="NLBinding Output Error: Invalid Model Predictions"
                )

        term_results: List[NLBindingTermResult] = []
        num_columns, num_values = len(request.columns), len(request.matched_values)
        num_keywords = len(cp_scores) - num_columns - num_values

        for idx, term_score in enumerate(cp_scores):
            term_score = cp_scores[idx]
            if term_score < self.threshold:
                continue

            term_type, term_index, term_value = None, None, None
            if idx < num_keywords:
                term_type = NLBindingType.Keyword
                term_index = idx
                term_value = _All_Agg_Op_Keywords[idx]
            elif idx < num_columns + num_keywords:
                term_type = NLBindingType.Column
                term_index = idx - num_keywords
                term_value = request.columns[term_index].name
            else:
                term_type = NLBindingType.Value
                term_index = idx - num_columns - num_keywords
                term_value = str(request.matched_values[term_index])

            term_result = NLBindingTermResult(
                term_type=term_type,
                term_index=term_index,
                term_value=term_value,
                term_score=term_score,
                grounding_scores=grounding_scores[idx]
            )

            term_results += [term_result]

        return term_results
