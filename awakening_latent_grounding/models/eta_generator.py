from collections import defaultdict
from typing import Dict, List
from logging import info

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from contracts import Text2SQLExample, All_Agg_Op_Keywords
from models.nn_layers import generate_mask_matrix


class ETAGroundingGenerator:
    def __init__(self, cp_model: nn.Module, tokenizer: PreTrainedTokenizer, cp_batch_size: int, neg_penalty_weight: float=None, verbose: bool=False) -> None:
        self.concept_predict_model = cp_model
        self.tokenizer = tokenizer
        self.erased_token_id = tokenizer.unk_token_id
        self.cp_batch_size = cp_batch_size
        self.neg_penalty_weight = neg_penalty_weight
        self.verbose = verbose

    def generate(self, inputs: Dict[str, torch.Tensor], is_training: bool=True) -> torch.Tensor:
        """
        Generate pseudo grounding alignment through erasing
        """
        if not is_training:
            return torch.zeros((inputs['concept_labels'].size(0), inputs['concept_labels'].size(1), inputs['question_mask'].size(1)), device=inputs['concept_labels'].device)

        training_mode = self.concept_predict_model.training
        self.concept_predict_model.eval()

        with torch.no_grad():
            input_token_ids, question_indices, entity_indices, erased_question_indices = \
                 inputs['input_token_ids'], inputs['question_indices'], inputs['entity_indices'], inputs['erased_indices']

            base_concept_scores = self._concept_predict(
                input_token_ids=input_token_ids,
                question_indices=question_indices,
                entity_indices=entity_indices
            )

            erased_inputs = self._erase(
                input_token_ids=input_token_ids,
                entity_indices=entity_indices,
                question_indices=question_indices,
                erased_indices=erased_question_indices
            )

            erased_concept_scores = self._concept_predict(
                input_token_ids=erased_inputs['input_token_ids'],
                question_indices=erased_inputs['question_indices'],
                entity_indices=erased_inputs['entity_indices']
            )

            pseudo_alignment = self._get_pseudo_alignments(
                examples=inputs['example'],
                concept_labels=inputs['concept_labels'],
                base_concept_scores=base_concept_scores,
                erased_concept_scores=erased_concept_scores,
                erased_question_spans=erased_inputs['erased_question_spans'],
                concept_mask=inputs['concept_mask'],
                question_mask=inputs['question_mask']
            )

            if self.verbose:
                self._verbose_erasing_results(
                    inputs,
                    base_concept_scores=base_concept_scores,
                    erased_concept_scores=erased_concept_scores,
                    erased_question_spans=erased_inputs['erased_question_spans'],
                    pseudo_alignment=pseudo_alignment
                )

        # Reset model to previous mode
        self.concept_predict_model.train(training_mode)
        return pseudo_alignment

    def _verbose_erasing_results(self, inputs, base_concept_scores, erased_concept_scores, erased_question_spans, pseudo_alignment):
        erased_results = {}
        for erased_idx, (batch_idx, q_start, q_end) in enumerate(erased_question_spans):
            if batch_idx not in erased_results:
                erased_results[batch_idx] = {}
            erased_results[batch_idx][(q_start, q_end)] = erased_concept_scores[erased_idx]

        for batch_idx in range(len(base_concept_scores)):
            example: Text2SQLExample = inputs['example'][batch_idx]
            info("Question: {}".format(example.question.text))
            info("Schema: {}".format(str(example.schema.to_string())))
            info("SQL: {}".format(str(example.sql)))
            concepts = example.get_grounding_concepts()
            scores = base_concept_scores[batch_idx, :len(concepts)].cpu().tolist()
            info("{:>10} Results: {}".format("Base" ," ".join(["{}={:.3f}".format(str(c), s) for c, s in zip(concepts, scores)])))

            for (q_start, q_end), erased_scores in erased_results[batch_idx].items():
                scores = erased_scores[:len(concepts)]
                erased_tokens = example.question.text_tokens[q_start:q_end+1]
                info("{:>10} Result: {}".format("_".join(erased_tokens), " ".join(["{}={:.3f}".format(str(c), s) for c, s in zip(concepts, scores)])))

            info("Pseudo Alignment:")
            question_tokens = example.question.text_tokens
            for c_idx in range(len(concepts)):
                scores = pseudo_alignment[batch_idx, c_idx, :len(question_tokens)]
                info("{:>10}: {}".format(str(concepts[c_idx]), " ".join(["{}={:.3f}".format(str(c), s) for c, s in zip(question_tokens, scores)])))
            info("="*100 + '\n')
        pass

    def _erase(self, input_token_ids: torch.Tensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor, erased_indices: List) -> torch.Tensor:
        batch_size = len(input_token_ids)
        erased_input_token_ids, erased_question_indices, erased_entity_indices, erased_question_spans = [], [], [], []
        for i in range(batch_size):
            for q_start, q_end, p_start, p_end in erased_indices[i]:
                new_input_token_ids = input_token_ids[i].detach().clone()
                new_input_token_ids[p_start:p_end+1] = self.erased_token_id
                erased_input_token_ids += [new_input_token_ids]

                erased_question_indices += [question_indices[i]]
                erased_entity_indices += [entity_indices[i]]

                erased_question_spans += [(i, q_start, q_end)]

        return {
            'input_token_ids': torch.stack(erased_input_token_ids, dim=0),
            'question_indices': torch.stack(erased_question_indices, dim=0),
            'entity_indices': torch.stack(erased_entity_indices, dim=0),
            'erased_question_spans': erased_question_spans
        }

    def _get_pseudo_alignments(self, examples: List[Text2SQLExample], concept_labels, base_concept_scores, erased_concept_scores, erased_question_spans, concept_mask: torch.BoolTensor, question_mask: torch.BoolTensor) -> torch.Tensor:
        cp_diff_dict = defaultdict(list)
        for erased_idx, (batch_idx, q_start, q_end) in enumerate(erased_question_spans):
            base_scores = base_concept_scores[batch_idx]
            erased_scores = erased_concept_scores[erased_idx]
            score_diff = (base_scores - erased_scores).clamp(0, 1)
            for q_idx in range(q_start, q_end + 1):
                cp_diff_dict[(batch_idx, q_idx)].append(score_diff)

        pseudo_alignment = torch.zeros((concept_labels.size(0), concept_labels.size(1), question_mask.size(1)), device=concept_labels.device)
        for (batch_idx, q_idx), diff_scores in cp_diff_dict.items():
            cp_score_diff = torch.mean(torch.stack(diff_scores, dim=0), dim=0)
            pseudo_alignment[batch_idx, :, q_idx] = cp_score_diff

        # This tries to remove dependency between columns and values
        for batch_idx in range(concept_labels.size(0)):
            example: Text2SQLExample = examples[batch_idx]
            base_column_index = len(All_Agg_Op_Keywords)
            base_value_index = len(All_Agg_Op_Keywords) + example.schema.num_columns

            for col_idx in range(example.schema.num_columns):
                if concept_labels[batch_idx, col_idx + base_column_index].item() == 0:
                    continue

                val_indices = example.get_value_indices(column=col_idx)
                mask = torch.zeros_like(question_mask[batch_idx], dtype=torch.bool)
                for val_idx in val_indices:
                    if concept_labels[batch_idx, val_idx + base_value_index].item() == 0:
                        continue
                    cell_value = example.matched_values[val_idx]
                    mask[cell_value.start:cell_value.end + 1] = 1
                pseudo_alignment[batch_idx, base_column_index + col_idx].masked_fill_(mask == 1, 0.0)

            for val_idx in range(len(example.matched_values)):
                if concept_labels[batch_idx, val_idx + base_value_index].item() == 0:
                    continue

                cell_value = example.matched_values[val_idx]
                mask = torch.zeros_like(question_mask[batch_idx], dtype=torch.bool)
                mask[cell_value.start:cell_value.end+1] = 1
                pseudo_alignment[batch_idx, base_value_index + val_idx].masked_fill_(mask == 0, 0.0)

        # Mask
        if self.neg_penalty_weight is not None:
             neg_concept_mask = concept_labels[:, :, None]
             pseudo_alignment.masked_fill_(neg_concept_mask == 0, self.neg_penalty_weight)

        grounding_mask = generate_mask_matrix(concept_mask, question_mask)
        pseudo_alignment.masked_fill_(grounding_mask == 0, 0.0)
        return pseudo_alignment

    def _concept_predict(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor):
        concept_scores ,index = [], 0
        while index < len(input_token_ids):
            tmp_cp_scores = self.concept_predict_model.forward(
                input_token_ids=input_token_ids[index:index + self.cp_batch_size],
                entity_indices=entity_indices[index:index + self.cp_batch_size],
                question_indices=question_indices[index:index + self.cp_batch_size],
                concept_predict_only=True
            )['concept_scores']

            concept_scores += [tmp_cp_scores]
            index += self.cp_batch_size

        # info("Run concept prediction over, batch size = {}".format(len(input_token_ids)))
        return torch.cat(concept_scores, dim=0)
