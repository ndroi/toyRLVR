from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch import nn

D_MODEL = 64
N_LAYER = 8
N_HEAD = 4

KVCache = Tuple[Tensor, Tensor]


class MHA(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.wv = nn.Linear(in_features=d_model, out_features=d_model)
        self.wo = nn.Linear(in_features=d_model, out_features=d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = self.d_model // self.n_head
        self.scaler = torch.tensor(1 / (self.d_k ** 0.5), dtype=torch.float32)

    def split_heads(self, a: Tensor) -> Tensor:
        # a: [bs, seq_len, d_model]
        bs = a.size(0)
        seq_len = a.size(1)
        out = a.view(bs, seq_len, self.n_head, self.d_k)  # [bs, seq_len, n_head, d_k]
        out = out.transpose(1, 2)  # [bs, n_head, seq_len, d_k]
        return out

    def forward(self, x: Tensor, mask: Tensor = None, past_kv: KVCache = None) -> (Tensor, KVCache):
        assert len(x.shape) == 3
        bs = x.size(0)
        # x: [bs, seq_len_x, d_model]
        q = self.wq(x)  # [bs, seq_len_x, d_model]
        k = self.wk(x)  # [bs, seq_len_x, d_model]
        v = self.wv(x)  # [bs, seq_len_x, d_model]
        if past_kv:
            k = torch.cat([past_kv[0], k], dim=1)  # [bs, seq_len_kv, d_model]
            v = torch.cat([past_kv[1], v], dim=1)  # [bs, seq_len_kv, d_model]
        cur_kv = (k, v)
        seq_len_x = x.size(1)
        seq_len_kv = k.size(1)
        q = self.split_heads(q)  # [bs, n_head, seq_len_x, d_k]
        k = self.split_heads(k)  # [bs, n_head, seq_len_kv, d_k]
        v = self.split_heads(v)  # [bs, n_head, seq_len_kv, d_k]
        k_t = k.transpose(-2, -1)  # [bs, n_head, d_k, seq_len_kv]
        scores = torch.matmul(q, k_t)  # [bs, n_head, seq_len_x, seq_len_kv]
        _mask = mask[seq_len_kv - seq_len_x:seq_len_kv, :seq_len_kv].view(1, 1, seq_len_x, seq_len_kv)
        scores = scores * self.scaler
        scores = scores + _mask
        weights = torch.softmax(scores, dim=-1)  # [bs, n_head, seq_len_x, seq_len_kv]
        out = torch.matmul(weights, v)  # [bs, n_head, seq_len_x, d_k]
        out = out.transpose(1, 2)  # [bs, seq_len_x, n_head, d_k]
        out = out.contiguous().view(bs, seq_len_x, self.d_model)  # [bs, seq_len_x, d_model]
        out = self.wo(out)  # [bs, seq_len_x, d_model]
        return out, cur_kv


class FFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, 4 * d_model)
        self.linear_2 = nn.Linear(4 * d_model, d_model)
        self.active = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear_1(x)
        out = self.active(out)
        out = self.linear_2(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.scaler = nn.Parameter(torch.ones(size, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs, seq_len, d_model]
        squared_sum = torch.sum(torch.square(x), dim=2, keepdim=True)  # [bs, seq_len, 1]
        mean = squared_sum / x.size(2)  # [bs, seq_len, 1]
        root = torch.sqrt(mean)  # [bs, seq_len, 1]
        x = x / root  # [bs, seq_len, d_model]
        x = x * self.scaler
        return x


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.mha = MHA(d_model=d_model, n_head=n_head)
        self.ffn = FFN(d_model=d_model)
        self.norm_1 = RMSNorm(size=d_model)
        self.norm_2 = RMSNorm(size=d_model)

    def forward(self, x: Tensor, mask: Tensor = None, past_kv: KVCache = None) -> (Tensor, KVCache):
        mha_out, cur_kv = self.mha(self.norm_1(x), mask=mask, past_kv=past_kv)
        x = x + mha_out
        x = x + self.ffn(self.norm_2(x))
        return x, cur_kv


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_layer: int, max_len: int = 1024):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, n_head=n_head) for _ in range(n_layer)
        ])
        mask = np.zeros((max_len, max_len))
        for i in range(max_len):
            mask[i][i + 1:] = float('-inf')
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))

    def forward(self, x: Tensor, past_kvs: List[KVCache] = None) -> (Tensor, List[KVCache]):
        past_kvs = past_kvs or [None] * len(self.blocks)
        cur_kvs = []
        for i, block in enumerate(self.blocks):
            x, cur_kv = block(x, mask=self.mask, past_kv=past_kvs[i])
            cur_kvs.append(cur_kv)
        return x, cur_kvs


def generate_pe(max_in_len: int, d_model: int) -> Tensor:
    pe = np.zeros(shape=(max_in_len, d_model))  # [max_in_len, d_model]
    positions = np.arange(max_in_len).reshape(-1, 1)  # [max_in_len, 1]
    i = np.arange(d_model // 2)  # [d_model // 2,]
    rates = 1 / np.power(10000, 2 * i / d_model)  # [d_model // 2,]
    angles = positions * rates  # [max_in_len, d_model // 2]
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return torch.tensor(data=pe, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, vocab_size: int, max_in_len: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_MODEL)
        self.register_buffer('pe', generate_pe(max_in_len=max_in_len, d_model=D_MODEL))
        self.decoder = Decoder(d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER)
        self.policy_out_linear = nn.Linear(in_features=D_MODEL, out_features=vocab_size, bias=True)
        self.value_out_linear = nn.Linear(in_features=D_MODEL, out_features=1, bias=True)

    def forward(self, tokens: Tensor, past_kvs: List[KVCache] = None) -> (Tensor, KVCache):
        in_len = tokens.shape[1]
        embedding = self.embed(tokens)
        embedding *= (D_MODEL ** 0.5)
        past_len = past_kvs[0][0].size(1) if past_kvs else 0
        pe = self.pe[past_len:past_len + in_len]
        embedding += pe
        embedding, cur_kvs = self.decoder(embedding, past_kvs=past_kvs)  # [bs, in_len, d_model]
        policy_out = self.policy_out_linear(embedding)  # [bs, in_len, vocab_size]
        value_out = self.value_out_linear(embedding)[:, :, 0]  # [bs, in_len]
        return policy_out, value_out, cur_kvs
