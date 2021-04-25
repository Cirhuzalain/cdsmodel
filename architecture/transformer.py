import math

import torch
import torch.nn as nn


def positional_encoding(dim, max_len=5000):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float)* -(math.log(10000.0)/ dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class Embedding(nn.Module):


    def __init__(self, embedding_dim, vocab_size, padding_idx, dropout):
        super(Embedding, self).__init__()
        self.word_padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        pe = positional_encoding(embedding_dim)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.embedding_dim ** -0.5)
    
    @property
    def padding_idx(self):
        return self.word_padding_idx

    def forward(self, x, timestep=0):
        embedding = self.embedding(x) * math.sqrt(self.embedding_dim) + self.pe[timestep:timestep + x.size(1)]
        return self.dropout(embedding)

class FeedForward(nn.Module):

    def __init__(self, hidden_size, inner_size, dropout):
        super(FeedForward, self).__init__()
        self.linear_in = nn.Linear(hidden_size, inner_size, bias=False)
        self.linear_out = nn.Linear(inner_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, x):
        y = self.linear_in(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_out(y)
        return y


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_count, model_dim, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.dim_per_head = model_dim // head_count
        self.head_count = head_count

        self.linear_q = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_k = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_v = nn.Linear(model_dim, model_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, query, memory=None, mask=None):
        memory = query if memory is None else memory

        def split_head(x):
            return x.view(x.size(0), -1, self.head_count, self.dim_per_head).transpose(1, 2)

        def combine_head(x):
            return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.head_count * self.dim_per_head)

        q = split_head(self.linear_q(query))
        k = split_head(self.linear_k(memory))
        v = split_head(self.linear_v(memory))

        q = q / math.sqrt(self.dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))

        mask = mask.unsqueeze(1).expand_as(scores).bool()

        scores.masked_fill_(mask, -1e18)

        weights = self.dropout(self.softmax(scores))
        context = combine_head(torch.matmul(weights, v))

        return self.final_linear(context)

class EncoderLayer(nn.Module):
    
    def __init__(self, hidden_size, dropout, head_count, ff_size):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(2)])

    def forward(self, x, mask):
        y = self.self_attn(self.norm[0](x), mask=mask)
        x = x + self.dropout[0](y)

        y = self.feed_forward(self.norm[1](x))
        x = x + self.dropout[1](y)
        return x

class Encoder(nn.Module):

    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size, embedding):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = embedding
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, src, src_pad):
        src_mask = src_pad.unsqueeze(1)
        output = self.embedding(src)

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask)

        return self.norm(output)

class DecoderLayer(nn.Module):
    
    def __init__(self, hidden_size, dropout, head_count, ff_size):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout)
        self.src_attn = MultiHeadedAttention(head_count, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(3)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, tgt_mask, previous=None):
        all_input = x if previous is None else torch.cat((previous, x), dim=1)

        y = self.self_attn(self.norm[0](x), self.norm[0](all_input), mask=tgt_mask)
        x = x + self.dropout[0](y)

        y = self.src_attn(self.norm[1](x), enc_out, mask=src_mask)
        x = x + self.dropout[1](y)

        y = self.feed_forward(self.norm[2](x))
        x = x + self.dropout[2](y)
        return x, all_input

class Decoder(nn.Module):
    
    def __init__(self, num_layers, num_heads, hidden_size, dropout, ff_size, embedding):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
       
        self.embedding = embedding
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, dropout, num_heads, ff_size) for _ in range(num_layers)])
        self.register_buffer("upper_triangle", torch.triu(torch.ones(1000, 1000), diagonal=1).byte())
        self.register_buffer("zero_mask", torch.zeros(1).byte())
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, tgt, enc_out, src_pad, tgt_pad, previous=None, timestep=0):

        output = self.embedding(tgt, timestep)
        tgt_len = tgt.size(1)

        src_mask = src_pad.unsqueeze(1)
        tgt_mask = tgt_pad.unsqueeze(1)
        upper_triangle = self.upper_triangle[:tgt_len, :tgt_len]

        tgt_mask = torch.gt(tgt_mask + upper_triangle, 0)
        save_inputs = []

        for i in range(self.num_layers):
            prev_layer = None if previous is None else previous[:, i]
            tgt_mask = tgt_mask if previous is None else self.zero_mask
            
            output, all_input = self.layers[i](output, enc_out, src_mask, tgt_mask, prev_layer)
            save_inputs.append(all_input)

        return self.norm(output), torch.stack(save_inputs, dim=1)