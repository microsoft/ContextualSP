import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils import *


def get_bert_hidden_size(bert_version: str) -> int:
    if bert_version in ['bert-base-uncased', 'bert-base-chinese', 'bert-base-multilingual-cased',
                        'hfl/chinese-bert-wwm', 'hfl/chinese-bert-wwm-ext', 'hfl/chinese-roberta-wwm-ext']:
        return 768
    if bert_version in ['bert-large-cased', 'bert-large-uncased', 'bert-large-uncased-whole-word-masking',
                        'hfl/chinese-roberta-wwm-ext-large']:
        return 1024
    raise NotImplementedError(f"not supported bert version: {bert_version}")


class AttentivePointer(nn.Module):
    def __init__(self, hidden_size: int):
        super(AttentivePointer, self).__init__()
        self.hidden_size = hidden_size

        self.linear_query = nn.Linear(hidden_size, hidden_size)
        self.linear_key = nn.Linear(hidden_size, hidden_size)
        self.linear_value = nn.Linear(hidden_size, hidden_size)

        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.LayerNorm(hidden_size))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.BoolTensor = None) -> \
    Tuple[torch.Tensor, torch.Tensor]:
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)

        attn_logits = torch.matmul(query, key.transpose(-2, -1))
        attn_logits /= math.sqrt(self.hidden_size)

        if mask is not None:
            attn_logits.masked_fill_(mask == 0, float('-inf'))

        # [batch_size, query_length, key_length]
        attn_weights = F.softmax(attn_logits, dim=-1)

        attn_outputs = torch.matmul(attn_weights, value)
        attn_outputs = self.linear_out(torch.cat((attn_outputs, query), dim=-1))

        return attn_outputs, attn_weights


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        n_classes = output.size(-1)
        log_logits = F.log_softmax(output, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_logits)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_logits, dim=-1))
