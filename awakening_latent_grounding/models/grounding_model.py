import abc
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nn_layers import BahdanauAttention, ScaledDotProductAttention, generate_mask_matrix
from contracts import load_pretrained_model, load_pretrained_tokenizer
from inference.data_adapter import scores_to_labels


class BaseGroundingModel(nn.Module):
    def __init__(self, pretrain_model_name: str, label_smoothing: float= 0.0, torch_script: bool=False):
        super().__init__()
        self.tokenizer = load_pretrained_tokenizer(pretrain_model_name)
        self.encoder = load_pretrained_model(pretrain_model_name, torchscript=torch_script)

        self.label_smoothing = label_smoothing
        self.crossentropyloss_fuc = nn.CrossEntropyLoss()

    @abc.abstractmethod
    def infer(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'concept_labels' in inputs, 'No concept labels to compute loss'
        outputs['cp_loss']  = F.binary_cross_entropy_with_logits(
            input=torch.logit(outputs['concept_scores'], eps=1e-4),
            target=inputs['concept_labels'].to(torch.float) * (1.0 - self.label_smoothing),
            weight=inputs['concept_mask'].to(torch.float)
        )

        if 'grounding_labels' in inputs:
            outputs['grounding_loss'] = self._compute_grounding_loss(
                concept_labels=inputs['concept_labels'],
                grounding_scores=outputs['grounding_scores'],
                grounding_labels=inputs['grounding_labels'],
                concept_mask=inputs['concept_mask'],
                question_mask=inputs['question_mask']
            )
        else:
            outputs.pop('grounding_scores', None)
        
        if 'sequence_labeling_start' in inputs:
            # scores_to_labels and to_tensor
            # todo: if from evaluation, question label should be gold (as well as loss computing)
            batch_labels = scores_to_labels(outputs['concept_scores'], outputs['grounding_scores'], inputs['example'])
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(outputs['grounding_scores'].device)
            outputs['sequence_labeling_loss'] = self._compute_sequence_labeling_loss(
                question_labels=batch_labels,
                question_label_scores=outputs['question_label_scores'],
                question_labels_mask=inputs['question_labels_mask'],
            )
        else:
            outputs.pop('question_label_scores', None)

        outputs['loss'] = outputs['cp_loss'] + outputs.get('grounding_loss', 0.0) * inputs.get('grounding_loss_weight', 1.0) + outputs.get('sequence_labeling_loss', 0.0) * inputs.get('sequence_labeling_loss_weight', 1.0)
        return outputs

    def _compute_sequence_labeling_loss(self,
        question_labels: torch.Tensor,
        question_label_scores: torch.Tensor,
        question_labels_mask: torch.BoolTensor,
        ) -> torch.Tensor:

        total_loss = 0.1

        """
        question_labels shape: torch.Size([8, 29])
        question_label_scores shape: torch.Size([8, 29, 7])
        question_labels_mask shape: torch.Size([8, 29])
        """

        # todo: loss function
        batch_size = question_labels.size(0)
        total_loss = 0
        for bix in range(batch_size):
            # should use mask to cut padding part.
            one_mask = question_labels_mask[bix]
            q_length = sum(one_mask)
            total_loss += self.crossentropyloss_fuc(question_label_scores[bix][:q_length], question_labels[bix][:q_length])

        return total_loss / batch_size
    
    def _compute_grounding_loss(self,
                                concept_labels: torch.Tensor,
                                grounding_scores: torch.Tensor,
                                grounding_labels: torch.Tensor,
                                concept_mask: torch.BoolTensor,
                                question_mask: torch.BoolTensor
                                ) -> torch.Tensor:

        total_loss = 0
        concept_lengths, question_lengths = torch.sum(concept_mask.to(torch.long), dim=1), torch.sum(question_mask.to(torch.long), dim=1) - 1
        for b_idx in range(concept_labels.size(0)):
            concept_length, question_length = concept_lengths[b_idx], question_lengths[b_idx]
            one_hot_labels = concept_labels[b_idx][:concept_length].unsqueeze(1).expand(-1, question_length)
            total_loss += F.binary_cross_entropy_with_logits(
                input=torch.logit(grounding_scores[b_idx, :concept_length, :question_length], eps=1e-6),
                target=one_hot_labels.to(torch.float),
                weight=grounding_labels[b_idx, :concept_length, :question_length])

        return total_loss / concept_labels.size(0)

    def fetch_outputs_by_index(self, outputs: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
        assert len(outputs) == len(indices), 'batch size must be equal'
        fetched_outputs = []
        for i in range(len(outputs)):
            fetched_outputs += [outputs[i, indices[i]]]
        return torch.stack(fetched_outputs, dim=0)

    def get_erasing_token_id(self):
        pass

class UniGroundingModel(BaseGroundingModel):
    def __init__(self, config: Dict):
        super().__init__(
            pretrain_model_name=config['pretrained_model'],
            label_smoothing=config['label_smoothing']
        )

        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_keywords = config['num_keywords']

        self.num_labels = config['num_labels']

        ##### sequence labeling module #####
        ## just use softmax ##
        self.linear_softmax_classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels),
            nn.Softmax(dim=-1)
        )

        self.linear_enc_out = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['dropout'])
        )

        self.cls_to_keywords = nn.Sequential(
            nn.Linear(self.hidden_size, config['num_keywords'] * self.hidden_size),
            nn.Tanh()
        )

        self.concept_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.grounding_scorer = ScaledDotProductAttention(self.hidden_size)

    def forward(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, **kwargs):
        encoder_outputs = self.encoder(
            input_token_ids,
            attention_mask=input_token_ids.ne(self.tokenizer.pad_token_id) if 'attention_mask' not in kwargs else kwargs['attention_mask']
        )[0]

        entity_outputs = self.linear_enc_out(self.fetch_outputs_by_index(encoder_outputs, entity_indices))
        keyword_outputs = self.cls_to_keywords(self.linear_enc_out(encoder_outputs[:, 0])).view(encoder_outputs.size(0), -1, self.hidden_size)
        concept_outputs = torch.cat([keyword_outputs, entity_outputs], dim=1)

        # Concept Prediction
        concept_scores = self.concept_scorer(concept_outputs).squeeze(2)
        outputs = { 'concept_scores': concept_scores }

        if kwargs.get('concept_predict_only', False):
            return outputs
        
        # Grounding
        question_indices = kwargs['question_indices']
        part_question_outputs = self.fetch_outputs_by_index(encoder_outputs, question_indices)
        question_outputs = self.linear_enc_out(part_question_outputs)
        concept2question_mask = generate_mask_matrix(kwargs['concept_mask'], kwargs['question_mask'])

        grounding_scores = self.grounding_scorer.forward(
            query=concept_outputs,
            key=question_outputs,
            mask=concept2question_mask
        )

        outputs['grounding_scores'] = grounding_scores

        raw_question_indices = kwargs['raw_question_indices']
        raw_question_outputs = self.fetch_outputs_by_index(encoder_outputs, raw_question_indices)
        question_logits = self.linear_softmax_classifier(raw_question_outputs)
        outputs['question_label_scores'] = question_logits

        if 'concept_labels' in kwargs:
            kwargs['input_token_ids'] = input_token_ids
            kwargs['entity_indices'] = entity_indices
            outputs = self.compute_loss(kwargs, outputs)

        return outputs
