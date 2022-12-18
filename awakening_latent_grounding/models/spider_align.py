import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from models.nn_layers import RelationalEncoder
from models.nn_utils import *
from collections import defaultdict
from typing import Dict, List
from utils.data_iter import MetaIndex


class SpiderAlignmentModel(nn.Module):
    def __init__(self, bert_version: str, dropout_prob: float) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_version)
        self.hidden_size = get_bert_hidden_size(bert_version)
        # self.rat_encoder = RelationalEncoder(num_layers=2, hidden_size=self.hidden_size, num_relations=len(SchemaRelation), num_heads=8, dropout_prob=dropout_prob)

        self.linear_out_tbl = nn.Linear(self.hidden_size, 2)
        self.linear_out_col = nn.Linear(self.hidden_size, 2)
        self.linear_out_val = nn.Linear(self.hidden_size, 2)

        self.align_pointer = AttentivePointer(self.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, **inputs) -> Dict:
        bert_outputs = self.bert(
            inputs['input_token_ids'],
            token_type_ids=inputs['input_token_types'],
            attention_mask=inputs['input_token_ids'].ne(0))["last_hidden_state"]

        bert_outputs = self.dropout(bert_outputs)

        batched_tbl_logits, batched_col_logits, batched_val_logits = [], [], []
        batched_align_weights = []
        for batch_idx in range(len(bert_outputs)):
            meta_index: MetaIndex = inputs['meta_index'][batch_idx]
            question_outputs = bert_outputs[batch_idx][
                meta_index.question_encode_indices + [meta_index.question_sep_index]]

            tbl_outputs = bert_outputs[batch_idx][meta_index.tbl_encode_indices]
            col_outputs = bert_outputs[batch_idx][meta_index.col_encode_indices]
            val_outputs = bert_outputs[batch_idx][meta_index.val_encode_indices]

            alignment_outputs, alignment_weights = self.align_pointer.forward(
                torch.cat((tbl_outputs, col_outputs, val_outputs), dim=0).unsqueeze(0),
                question_outputs.unsqueeze(0),
                question_outputs.unsqueeze(0))

            # Use attentive outputs to do identification
            # tbl_outputs, col_outputs, val_outputs = meta_index.split(alignment_outputs.squeeze(0))

            batched_tbl_logits += [self.linear_out_tbl(tbl_outputs)]
            batched_col_logits += [self.linear_out_col(col_outputs)]
            batched_val_logits += [self.linear_out_val(val_outputs)]
            batched_align_weights += [alignment_weights.squeeze(0)[:, :-1]]

        return {
            'table_logits': batched_tbl_logits,
            'column_logits': batched_col_logits,
            'value_logits': batched_val_logits,
            'alignment_weights': batched_align_weights
        }

    def compute_loss(self, **inputs):
        outputs = self.forward(**inputs)
        table_logits, column_logits, value_logits = outputs['table_logits'], outputs['column_logits'], outputs[
            'value_logits']

        total_loss = 0
        identify_loss = self._calculate_identification_loss(table_logits, column_logits, value_logits, **inputs)
        total_loss += identify_loss
        outputs['identify_loss'] = identify_loss

        alignment_loss_weight = inputs['align_loss_weight'] if 'align_loss_weight' in inputs else 0.0

        if alignment_loss_weight > 1e-3:
            align_loss = self._calculate_alignment_loss(table_logits, column_logits, value_logits,
                                                        outputs['alignment_weights'], **inputs)
            total_loss += align_loss * alignment_loss_weight
            outputs['align_loss'] = align_loss

        outputs['loss'] = total_loss
        return outputs

    def _calculate_identification_loss(self, tbl_logits, col_logits, val_logits, **inputs):
        tbl_labels, col_labels, val_labels = inputs['table_labels'], inputs['column_labels'], inputs['value_labels']
        assert len(tbl_logits) == len(col_logits)
        assert len(tbl_labels) == len(tbl_logits)
        assert len(col_labels) == len(col_logits)

        total_loss = 0
        criterion = LabelSmoothingLoss(0.05) if inputs['label_smoothing'] else nn.CrossEntropyLoss()
        for batch_idx in range(len(tbl_labels)):
            total_loss += criterion(tbl_logits[batch_idx], tbl_labels[batch_idx])
            total_loss += criterion(col_logits[batch_idx], col_labels[batch_idx])
            if len(val_labels[batch_idx]) > 0:
                total_loss += criterion(val_logits[batch_idx], val_labels[batch_idx])

        return total_loss / len(tbl_labels) / 3

    def _calculate_alignment_loss(self, tbl_logits, col_logits, val_logits, align_weights, **inputs):
        assert len(tbl_logits) == len(col_logits)
        assert len(tbl_logits) == len(align_weights)

        total_alignment_loss = 0
        for batch_idx in range(len(tbl_logits)):
            meta_index: MetaIndex = inputs['meta_index'][batch_idx]
            tbl_labels, col_labels, val_labels, = inputs['table_labels'][batch_idx], \
                                                  inputs['column_labels'][batch_idx], \
                                                  inputs['value_labels'][batch_idx]

            with torch.no_grad():
                masking_inputs = self._generate_masking_inputs(
                    input_token_ids=inputs['input_token_ids'][batch_idx].detach(),
                    input_token_types=inputs['input_token_types'][batch_idx].detach(),
                    meta_index=meta_index,
                    example=inputs['example'][batch_idx])

                masking_scores = None
                if 'masking_infer_func' not in inputs:
                    masking_scores = self._run_masking_outputs(
                        input_token_ids=masking_inputs['input_token_ids'],
                        input_token_types=masking_inputs['input_token_types'],
                        meta_index=meta_index,
                        batch_size=len(tbl_logits))
                else:
                    masking_scores = inputs['masking_infer_func'](masking_inputs)

                masking_rewards = self._calculate_masking_rewards(
                    labels={'tbl': tbl_labels, 'col': col_labels, 'val': val_labels, },
                    base_scores={'tbl': F.softmax(tbl_logits[batch_idx], dim=-1),
                                 'col': F.softmax(col_logits[batch_idx], dim=-1),
                                 'val': F.softmax(val_logits[batch_idx], dim=-1)},
                    masking_scores=masking_scores,
                    masking_spans=masking_inputs['masking_spans'],
                    question_length=meta_index.num_question_tokens)

            tbl_align_weights, col_align_weights, val_align_weights = meta_index.split(align_weights[batch_idx], dim=0)
            question_length = meta_index.num_question_tokens

            total_alignment_loss += F.binary_cross_entropy(
                tbl_align_weights,
                tbl_labels.to(torch.float).repeat_interleave(question_length).view(-1, question_length),
                weight=masking_rewards['tbl'])
            total_alignment_loss += F.binary_cross_entropy(col_align_weights,
                                                           col_labels.to(torch.float).repeat_interleave(
                                                               question_length).view(-1, question_length),
                                                           weight=masking_rewards['col'])
            if len(val_align_weights) > 0:
                total_alignment_loss += F.binary_cross_entropy(val_align_weights,
                                                               val_labels.to(torch.float).repeat_interleave(
                                                                   question_length).view(-1, question_length),
                                                               weight=masking_rewards['val'])

        return total_alignment_loss / len(tbl_logits) / 3

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
                             meta_index: MetaIndex, batch_size: int):
        index = 0
        batched_tbl_scores, batched_col_scores, batched_val_scores = [], [], []
        while index < len(input_token_ids):
            bert_outputs = self.bert(
                input_token_ids[index:index + batch_size],
                attention_mask=input_token_ids[index:index + batch_size].ne(0),
                token_type_ids=input_token_types[index:index + batch_size])["last_hidden_state"]

            tbl_outputs = bert_outputs[:, meta_index.tbl_encode_indices]
            tbl_scores = F.softmax(self.linear_out_tbl(tbl_outputs), dim=-1)

            col_outputs = bert_outputs[:, meta_index.col_encode_indices]
            col_scores = F.softmax(self.linear_out_col(col_outputs), dim=-1)

            val_outputs = bert_outputs[:, meta_index.val_encode_indices]
            val_scores = F.softmax(self.linear_out_val(val_outputs), dim=-1)

            index += batch_size

            batched_tbl_scores.append(tbl_scores)
            batched_col_scores.append(col_scores)
            batched_val_scores.append(val_scores)

        return {
            'tbl': torch.cat(batched_tbl_scores, dim=0),
            'col': torch.cat(batched_col_scores, dim=0),
            'val': torch.cat(batched_val_scores, dim=0)
        }

    def _calculate_masking_rewards(self,
                                   labels: Dict[str, torch.LongTensor],
                                   base_scores: Dict[str, torch.Tensor],
                                   masking_scores: Dict[str, torch.Tensor],
                                   masking_spans: List[Tuple[int, int]],
                                   question_length: int,
                                   default_weight: float = 0.01,
                                   ):

        masking_rewards = {}
        for e_type in ['tbl', 'col', 'val']:
            e_labels, e_base_scores, e_masking_scores = labels[e_type], base_scores[e_type], masking_scores[e_type]
            reward = torch.zeros((len(e_labels), question_length), device=e_labels.device)
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

                for q_idx in range(question_length):
                    reward[idx, q_idx] = sum(ngram_rewards[q_idx]) / len(
                        ngram_rewards[q_idx]) if q_idx in ngram_rewards else 0.0
            masking_rewards[e_type] = reward
        return masking_rewards
