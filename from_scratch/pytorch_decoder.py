from ... import util
from torch.nn import Dropout
from torch.nn import Module

class Decoder(Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = util.clones(layer, N)
        self.norm = util.LayerNorm(layer.size)

    def forward(self, tgt, memory, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(tgt, memory, tgt_mask, src_mask)
        return self.norm(x)

class DecoderLayer(Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = util.clones(util.SubLayer(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, tgt_mask, src_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(memory, memory, x, src_mask))
        return self.sublayer[2](x, self.feed_forward)