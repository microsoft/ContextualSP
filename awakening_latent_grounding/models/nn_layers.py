import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from The Annotated Transformer
class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout):
        super(MultiHeadedAttentionWithRelations, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, relation_k, relation_v, mask=None):
        # query shape: [batch_size, query_length, hidden_size]
        # key shape: [batch_size, kv_length, hidden_size]
        # value shape: [batch_size, kv_length, hidden_size]
        # relation_k shape: [batch_size, query_length, kv_length, hidden_size // num_heads]
        # relation_v shape: [batch_size, query_length, kv_length, hidden_size // num_heads]
        batch_size = query.size(0)

        # [batch_size, num_heads, query_length, head_dim]
        query = self.linears[0](query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linears[1](key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linears[2](value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_outputs, attn_weights = self.attention_with_relations(query, key, value, relation_k, relation_v, mask=mask)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        attn_outputs = self.linears[-1](attn_outputs)

        return attn_outputs #, attn_weights
    
    def attention_with_relations(self, query, key, value, relation_k, relation_v, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = self.relative_attention_logits(query, key, relation_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1000)

        p_attn_orig = F.softmax(scores, dim=-1)
        # print(p_attn_orig.shape, value.shape, relation_v.shape)
        #if self.dropout is not None:
        p_attn = self.dropout(p_attn_orig)
        return self.relative_attention_values(p_attn, value, relation_v), p_attn_orig
    
    def relative_attention_logits(self, query, key, relation):
        # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
        # In this version, relation vectors are shared across heads.
        # query: [batch, heads, num queries, depth].
        # key: [batch, heads, num kvs, depth].
        # relation: [batch, num queries, num kvs, depth].

        # qk_matmul is [batch, heads, num queries, num kvs]
        qk_matmul = torch.matmul(query, key.transpose(-2, -1))

        # q_t is [batch, num queries, heads, depth]
        q_t = query.permute(0, 2, 1, 3)

        # r_t is [batch, num queries, depth, num kvs]
        r_t = relation.transpose(-2, -1)

        #   [batch, num queries, heads, depth]
        # * [batch, num queries, depth, num kvs]
        # = [batch, num queries, heads, num kvs]
        # For each batch and query, we have a query vector per head.
        # We take its dot product with the relation vector for each kv.
        q_tr_t_matmul = torch.matmul(q_t, r_t)

        # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

        # [batch, heads, num queries, num kvs]
        return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

    def relative_attention_values(self, weight, value, relation):
        # In this version, relation vectors are shared across heads.
        # weight: [batch, heads, num queries, num kvs].
        # value: [batch, heads, num kvs, depth].
        # relation: [batch, num queries, num kvs, depth].

        # wv_matmul is [batch, heads, num queries, depth]
        wv_matmul = torch.matmul(weight, value)

        # w_t is [batch, num queries, heads, num kvs]
        w_t = weight.permute(0, 2, 1, 3)

        #   [batch, num queries, heads, num kvs]
        # * [batch, num queries, num kvs, depth]
        # = [batch, num queries, heads, depth]
        w_tr_matmul = torch.matmul(w_t, relation)

        # w_tr_matmul_t is [batch, heads, num queries, depth]
        w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

        return wv_matmul + w_tr_matmul_t

# Adapted from The Annotated Transformer
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class RelationalEncoderLayer(nn.Module):
    def __init__(self, num_heads, hidden_size, num_relations, dropout):
        super(RelationalEncoderLayer, self).__init__()
        assert hidden_size % num_heads == 0

        self.self_attn = MultiHeadedAttentionWithRelations(num_heads, hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size * 4, dropout)

        self.sub_layers = nn.ModuleList([SublayerConnection(hidden_size, dropout)  for _ in range(2)])

        self.relation_k_embeddings = nn.Embedding(num_relations, hidden_size // num_heads)
        self.relation_v_embeddings = nn.Embedding(num_relations, hidden_size // num_heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, relation_ids, mask=None):
        # Map relation id to embedding
        relation_k = self.dropout(self.relation_k_embeddings(relation_ids))
        relation_v = self.dropout(self.relation_v_embeddings(relation_ids))

        inputs = self.sub_layers[0](inputs, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask))
        return self.sub_layers[1](inputs, self.feed_forward)

class RelationalEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, num_relations, dropout_prob):
        super(RelationalEncoder, self).__init__()

        self.encode_layers = nn.ModuleList([RelationalEncoderLayer(num_heads, hidden_size, num_relations, dropout_prob) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, inputs, relations, mask=None) -> torch.Tensor:
        for layer in self.encode_layers:
            inputs = layer(inputs, relations, mask)
        return self.layer_norm(inputs)