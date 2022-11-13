import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_mask_matrix(mask1: torch.BoolTensor, mask2: torch.BoolTensor) -> torch.BoolTensor:
    assert len(mask1) == len(mask2)
    mask1_expand = mask1.unsqueeze(2).expand(-1, -1, mask2.size(1))
    mask2_expand = mask2.unsqueeze(1).expand(-1, mask1.size(1), -1)
    return mask1_expand & mask2_expand

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_query = nn.Linear(hidden_size, hidden_size)
        self.linear_key = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        query = self.linear_query(query)
        key = self.linear_key(key)
        
        attn_logits = torch.matmul(query, key.transpose(-2, -1))
        attn_logits /= math.sqrt(self.hidden_size)

        if mask is not None:
            attn_logits.masked_fill_(mask == 0, -1000)
        
        # [batch_size, query_length, key_length]
        attn_weights = F.softmax(attn_logits, dim=-1)
        return attn_weights

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int, attn_size: int=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_size = attn_size

        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, attn_size)
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_expanded = query.unsqueeze(2).expand(-1, -1, key.size(1), -1)
        key_expanded = key.unsqueeze(1).expand(-1, query.size(1), -1, -1)

        # batch_size * query_size * attn_size
        attn_logits = self.scorer.forward(
            torch.cat((query_expanded, key_expanded), dim=3)
        )

        if mask is not None:
            attn_logits.masked_fill_(mask[:, :, :, None] == 0, -1000)
        
        # [batch_size, query_length, key_length, attn_size]
        attn_weights = F.softmax(attn_logits, dim=2)
        return attn_weights