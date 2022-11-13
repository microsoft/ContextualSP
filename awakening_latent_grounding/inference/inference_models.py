import os
import json
from typing import Dict, Tuple
import torch
import torch.nn as nn
from models import BaseGroundingModel, ScaledDotProductAttention

class GroundingInferenceModel(BaseGroundingModel):
    def __init__(self, config: Dict):
        super().__init__(pretrain_model_name=config['pretrained_model'], torch_script=True)
        self.config = config
        self.hidden_size = config['hidden_size']

        self.num_labels = config['num_labels']

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

    def forward(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_token_ids = input_token_ids.unsqueeze(0)
        attention_mask = input_token_ids.ne(self.tokenizer.pad_token_id)

        encoder_outputs = self.encoder.forward(
            input_token_ids,
            position_ids=torch.cumsum(attention_mask, dim=1) + 1,
            attention_mask=attention_mask,
            token_type_ids=(~attention_mask).long(),
        )[0]

        entity_outputs = self.linear_enc_out(encoder_outputs[:, entity_indices])
        keyword_outputs = self.cls_to_keywords(self.linear_enc_out(encoder_outputs[:, 0])).view(encoder_outputs.size(0), -1, self.hidden_size)
        concept_outputs = torch.cat([keyword_outputs, entity_outputs], dim=1)

        concept_scores = self.concept_scorer.forward(concept_outputs).squeeze(2).squeeze(0)
        question_outputs = self.linear_enc_out(encoder_outputs[:, question_indices])

        grounding_scores = self.grounding_scorer.forward(query=concept_outputs, key=question_outputs, mask=None).squeeze(0)

        return concept_scores, grounding_scores[:, :-1]

    @classmethod
    def from_trained(cls, ckpt_path: str):
        config_path = os.path.join(os.path.dirname(ckpt_path), 'model_config.json')
        print("Load model config from {} ...".format(config_path))
        config = json.load(open(config_path, 'r', encoding='utf-8'))

        model = cls(config)
        print('load model from checkpoint {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        model.eval()
        return model

class GroundingInferenceModelForORT(GroundingInferenceModel):
    def forward(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_token_ids = input_token_ids.unsqueeze(0)
        attention_mask = input_token_ids.ne(self.tokenizer.pad_token_id).to(torch.long)

        encoder_outputs = self.encoder.forward(
            input_token_ids,
            attention_mask=attention_mask,
        )[0]

        entity_outputs = self.linear_enc_out(encoder_outputs[:, entity_indices])
        keyword_outputs = self.cls_to_keywords(self.linear_enc_out(encoder_outputs[:, 0])).view(encoder_outputs.size(0), -1, self.hidden_size)
        concept_outputs = torch.cat([keyword_outputs, entity_outputs], dim=1)

        concept_scores = self.concept_scorer.forward(concept_outputs).squeeze(2).squeeze(0)
        question_outputs = self.linear_enc_out(encoder_outputs[:, question_indices])

        grounding_scores = self.grounding_scorer.forward(query=concept_outputs, key=question_outputs, mask=None).squeeze(0)

        return concept_scores, grounding_scores[:, :-1]