import math
from util import *
from constants import *
from torch import Tensor
import torch.nn.functional as F
from  torch.nn import Transformer as tf

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_emb = torch.zeros((max_len, emb_size))
        div = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos_emb[:, 0::2] = torch.sin(pos * div)
        pos_emb[:, 1::2] = torch.cos(pos * div)
        pos_emb = pos.unsqueeze(-2)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_emb[:token_embedding.size(0), :])

class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers, num_decoder_layers, 
                 emb_size, nhead,
                 src_vocab_size, tgt_vocab_size,
                 dim_feedforward=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.trnasformer = tf(emb_size, nhead,
                              num_encoder_layers, num_decoder_layers,
                              dim_feedforward, dropout)
        self.generator = Generator(emb_size, tgt_vocab_size)
        self.src_tok_emb = Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        memory_key_padding_mask = src_padding_mask
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.trnasformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.trnasformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.trnasformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)