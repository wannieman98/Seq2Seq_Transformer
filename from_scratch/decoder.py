class DecoderLayer(nn.Module):
    '''디코더 레이어'''
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(params)
        self.layer_norm1 = nn.LayerNorm(params['hidden_dim'])

        self.enc_dec_attn = MultiHeadAttention(params)
        self.layer_norm2 = nn.LayerNorm(params['hidden_dim'])
        
        self.feed_forward = PositionwiseFeedForward(params)
        self.layer_norm3 = nn.LayerNorm(params['hidden_dim'])
        
        self.dropout = nn.Dropout(params['dropout'])
        
    def forward(self, x, tgt_mask, enc_output, src_mask):
        " x = [배치 사이즈, 문장 길이, 은닉 차원] "
        
        residual = x
        x, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm1(x)
        
        residual = x
        x, attn_map = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm2(x)
        
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm3(x)
        
        return x, attn_map


class Decoder(nn.Module):
    '''트랜스포머 디코더'''
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(params['vocab_size'], params['hidden_dim'], padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(params)
        self.layers = nn.ModuleList([DecoderLayer(params) for _ in range(params['num_layers'])])
        
    def forward(self, tgt, src, enc_out):
        " tgt = [배치 사이즈, 타겟 문장 길이] "

        src_mask, tgt_mask = create_tgt_mask(src, tgt)
        tgt = self.tok_embedding(tgt) + self.pos_embedding(tgt)
        
        for layer in self.layers:
            tgt, attn_map = layer(tgt, tgt_mask, enc_out, src_mask)
            
        tgt = torch.matmul(tgt, self.tok_embedding.weight.transpose(0, 1))
        # tgt = [배치 사이즈, 타겟 문장 길이, 은닉 차원]

        return tgt, attn_map