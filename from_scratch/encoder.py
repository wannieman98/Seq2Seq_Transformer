class EncoderLayer(nn.Module):
    '''인코더 레이어'''
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(params)
        self.layer_norm1 = nn.LayerNorm(params['hidden_dim'])
        self.feed_forward = PositionwiseFeedForward(params)
        self.layer_norm2 = nn.LayerNorm(params['hidden_dim'])
        self.dropout = nn.Dropout(params['dropout'])
        
    def forward(self, x, src_mask):
        " x = [배치 사이즈, 문장 길이, 은닉 차원] "
        
        residual = x
        x, _ = self.self_attn(x, x, x, src_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm1(x)
        
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm2(x)
        
        return x


class Encoder(nn.Module):
    '''트랜스포머 인코더'''
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.tok_embedding = nn.Embedding(params['vocab_size'], params['hidden_dim'], padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(params)
        self.layers = nn.ModuleList([EncoderLayer(params) for _ in range(params['num_layers'])])
        
    def forward(self, src):
        " src = [배치 사이즈, 소스 문장 길이] "

        src_mask = create_src_mask(src)
        src = self.tok_embedding(src) + self.pos_embedding(src)
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        # src = [배치 사이즈, 소스 문장 길이, 은닉 차원]
        return src