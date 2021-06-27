import math
from util import *
from torch import Tensor
import torch.nn.functional as F
from  torch.nn import Transformer as tf

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_embedding = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len).reshape(max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        positional_embedding[:, 0::2] = torch.sin(position * div)
        positional_embedding[:, 1::2] = torch.cos(position * div)
        positional_embedding = positional_embedding.unsqueeze(-2)

        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, x):
        x = x + self.positional_embedding[:x.size(0),:].requires_grad_(False)
        return self.dropout(x)

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
        self.transformer = tf(emb_size, nhead,
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
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
