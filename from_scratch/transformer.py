from torch import nn
import torch.nn as nn
from copy import deepcopy as dc
from pytorch_encoder import *
from pytorch_decoder import *
from attentions import *
from position import *
from util import *

class o_transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(o_transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class encoder_decoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, N):
        super(encoder_decoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, N)
        self.decoder_layers = clones(decoder_layer, N)

    def forward(self, src, src_mask, tgt, tgt_mask):
        pass


class p_transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(p_transformer, self).__init__()
        self.encoder = encoder
        self.deocder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

def build_model(vocabs, nhead, d_model, d_ff, N, device, dropout=0.1, variation=False, load=False):
    attn = nn.MultiheadAttention(d_model, nhead,dropout)
    feedforward = PositionWiseFeedForward(d_model, d_ff)
    position = PositionalEncoding(d_model, dropout)
    if not variation:
        model = o_transformer(Encoder(EnocderLayer(d_model, dc(attn), dc(feedforward), dropout), N), 
                              Decoder(DecoderLayer(d_model, dc(attn), dc(attn), dc(feedforward), dropout), N),
                              nn.Sequential(Embeddings(d_model, len(vocabs['src_lang'])), dc(position)),
                              nn.Sequential(Embeddings(d_model, len(vocabs['tgt_lang'])), dc(position)),
                              Generator(d_model, len(vocabs['tgt_lang']))
                              )

    if load:
        state_dict = torch.load('checkpoints/script_checkpoint_inf.pth')
        model.load_state_dict(state_dict)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model.to(device)

    return model