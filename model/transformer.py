from copy import deepcopy
import torch.nn as nn
from model.embed import Embedding
from model.attention import MultiHeadAttention
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.position import PositionalEncoding, PositionWiseFeedForward

class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 device = None):
        super(Transformer, self).__init__()
        self.attention = MultiHeadAttention(emb_size, nhead, dropout, device)
        self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.encoder = Encoder(
            EncoderLayer(emb_size, deepcopy(self.attention), deepcopy(self.ff), dropout),
            num_encoder_layers
        )
        self.decdoer = Decoder(
            DecoderLayer(emb_size, deepcopy(self.attention), deepcopy(self.ff), dropout, device),
            num_decoder_layers
        )

    def forward(self,
                src,
                tgt,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        return self.decode(tgt, self.encode(src, src_mask, src_padding_mask), tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)

    def encode(self, src, src_mask, src_padding_mask):
        return self.encoder(
            self.positional_encoding(self.src_tok_emb(src)), 
            src_mask,
            src_padding_mask
            )

    def decode(self, tgt, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_key_padding_mask):
        return self.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), 
            memory, 
            tgt_mask,
            memory_mask, 
            tgt_padding_mask,
            memory_key_padding_mask
            )