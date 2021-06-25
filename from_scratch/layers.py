import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, features_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features_size))
        self.b_2 = nn.Parameter(torch.zeros(features_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean)  / (std + self.eps) + self.b_2

class SubLayer(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        attn = ()
        attn = sublayer(self.norm(x))
        return x + self.dropout(attn[0])

class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)