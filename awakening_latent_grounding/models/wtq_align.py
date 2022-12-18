from utils.data_types import SQLTokenType
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from collections import defaultdict
from typing import Dict, List
from utils.data_iter import MetaIndex
from models.nn_utils import *


class WTQAlignmentModel(nn.Module):
    def __init__(self, bert_version: str, dropout_prob: float) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_version)
        self.hidden_size = get_bert_hidden_size(bert_version)

        self.linear_out_col = nn.Linear(self.hidden_size, 2)
        self.align_pointer = AttentivePointer(self.hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, **inputs) -> Dict:
        bert_outputs = self.bert(
            inputs['input_token_ids'],
            token_type_ids=inputs['input_token_types'],
            attention_mask=inputs['input_token_ids'].ne(0))["last_hidden_state"]

        bert_outputs = self.dropout(bert_outputs)

        batched_col_logits = []
        batched_align_weights = []
        batched_question_outputs, batched_entity_outputs = [], []
        for batch_idx in range(len(bert_outputs)):
            meta_index: MetaIndex = inputs['meta_index'][batch_idx]
            question_outputs = bert_outputs[batch_idx][1: meta_index.question_sep_index + 1]
            batched_question_outputs += [question_outputs[:-1]]

            col_outputs = bert_outputs[batch_idx][meta_index.col_encode_indices]
            col_logits = self.linear_out_col(col_outputs)
            batched_col_logits += [col_logits]

            entity_outputs = col_outputs
            _, alignment_weights = self.align_pointer.forward(
                entity_outputs.unsqueeze(0),
                question_outputs.unsqueeze(0),
                question_outputs.unsqueeze(0))

            batched_align_weights += [alignment_weights.squeeze(0)[:, :-1]]
            batched_entity_outputs.append({SQLTokenType.column: col_outputs})

        outputs = {
            'column_logits': batched_col_logits,
            'alignment_weights': batched_align_weights,
            'question_outputs': batched_question_outputs,
            'entity_outputs': batched_entity_outputs,
        }

        return outputs

    def compute_loss(self, **inputs):
        outputs = self.forward(**inputs)
        column_logits = outputs['column_logits']
        total_loss = 0
        identify_loss = self._calculate_identification_loss(column_logits, **inputs)
        total_loss += identify_loss
        outputs['identify_loss'] = identify_loss

        alignment_loss_weight = inputs['align_loss_weight'] if 'align_loss_weight' in inputs else 0.0

        if alignment_loss_weight > 1e-3:
            align_loss = self._calculate_alignment_loss(column_logits, outputs['alignment_weights'], **inputs)
            total_loss += align_loss * alignment_loss_weight
            outputs['align_loss'] = align_loss

        outputs['loss'] = total_loss
        return outputs

    def _calculate_identification_loss(self, col_logits, **inputs):
        col_labels = inputs['column_labels']
        assert len(col_labels) == len(col_logits)

        total_loss = 0
        criterion = LabelSmoothingLoss(0.05) if inputs['label_smoothing'] else nn.CrossEntropyLoss()
        for batch_idx in range(len(col_labels)):
            total_loss += criterion(col_logits[batch_idx], col_labels[batch_idx])

        return total_loss / len(col_labels)

    def _calculate_alignment_loss(self, col_logits, align_weights, **inputs):
        assert len(col_logits) == len(align_weights)

        total_alignment_loss = 0
        for batch_idx in range(len(col_logits)):
            meta_index: MetaIndex = inputs['meta_index'][batch_idx]
            # question_length = meta_index.num_question_tokens
            question_length = meta_index.question_sep_index - 1
            col_labels = inputs['column_labels'][batch_idx]

            with torch.no_grad():
                masking_inputs = self._generate_masking_inputs(
                    input_token_ids=inputs['input_token_ids'][batch_idx].detach(),
                    input_token_types=inputs['input_token_types'][batch_idx].detach(),
                    meta_index=meta_index,
                    example=inputs['example'][batch_idx])

                masking_scores = self._run_masking_outputs(
                    input_token_ids=masking_inputs['input_token_ids'],
                    input_token_types=masking_inputs['input_token_types'],
                    meta_index=meta_index)

                masking_rewards = self._calculate_masking_rewards(
                    labels={'col': col_labels},
                    base_scores={'col': F.softmax(col_logits[batch_idx], dim=-1)},
                    masking_scores=masking_scores,
                    masking_spans=masking_inputs['masking_spans'],
                    meta_index=meta_index)

            total_alignment_loss += F.binary_cross_entropy(align_weights[batch_idx],
                                                           col_labels.to(torch.float).repeat_interleave(
                                                               question_length).view(-1, question_length),
                                                           weight=masking_rewards['col'])

        return total_alignment_loss / len(col_logits)

    @staticmethod
    def _generate_masking_inputs(input_token_ids: torch.Tensor, input_token_types: torch.Tensor, meta_index: MetaIndex,
                                 example: Dict):
        all_masking_input_token_ids, all_masking_spans = [], []
        for i, j, _ in example['masking_ngrams']:
            p_start, p_end = meta_index.question_spans[i][1], meta_index.question_spans[j][2]
            masking_input_token_ids = input_token_ids.clone()
            masking_input_token_ids[p_start:p_end + 1] = 100  # unk
            all_masking_input_token_ids += [masking_input_token_ids]
            all_masking_spans += [(i, j)]

        return {
            'input_token_ids': torch.stack(all_masking_input_token_ids, dim=0),
            'input_token_types': torch.stack([input_token_types for _ in all_masking_spans]),
            'meta_index': [meta_index for _ in all_masking_spans],
            'masking_spans': all_masking_spans
        }

    def _run_masking_outputs(self, input_token_ids: torch.Tensor, input_token_types: torch.Tensor,
                             meta_index: MetaIndex):
        bert_outputs = \
            self.bert(input_token_ids, attention_mask=input_token_ids.ne(0), token_type_ids=input_token_types)[
                "last_hidden_state"]
        col_outputs = bert_outputs[:, meta_index.col_encode_indices]
        col_scores = F.softmax(self.linear_out_col(col_outputs), dim=-1)

        return {'col': col_scores}

    def _calculate_masking_rewards(self,
                                   labels: Dict[str, torch.LongTensor],
                                   base_scores: Dict[str, torch.Tensor],
                                   masking_scores: Dict[str, torch.Tensor],
                                   masking_spans: List[Tuple[int, int]],
                                   meta_index: MetaIndex,
                                   default_weight: float = 0.1,
                                   ):
        num_question_subword = meta_index.question_sep_index - 1
        masking_rewards = {}
        for e_type in ['col']:
            e_labels, e_base_scores, e_masking_scores = labels[e_type], base_scores[e_type], masking_scores[e_type]
            reward = torch.zeros((len(e_labels), num_question_subword), device=e_labels.device)
            for idx in range(len(e_labels)):
                label = e_labels[idx].item()
                if label == 0:
                    reward[idx] = default_weight
                    continue
                ngram_rewards = defaultdict(list)
                for m_i, (start, end) in enumerate(masking_spans):
                    score_diff = (e_base_scores[idx, label] - e_masking_scores[m_i, idx, label]).clamp(0, 1).item()
                    for j in range(start, end + 1):
                        ngram_rewards[j].append(score_diff)

                for q_idx in range(num_question_subword):
                    reward[idx, q_idx] = sum(ngram_rewards[q_idx]) / len(
                        ngram_rewards[q_idx]) if q_idx in ngram_rewards else 0.0

            masking_rewards[e_type] = reward
        return masking_rewards
