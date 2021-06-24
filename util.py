import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src_sentence, vocabs, text_transform, device):
    model.eval()
    src = text_transform['src_lang'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_IDX, device=device).flatten()
    return " ".join(vocabs['tgt_lang'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos> ", "").replace(" <eos>", "")

def epoch_time(time, curr_epoch, total_epochs):
    minutes = int(time / 60)
    seconds = int(time % 60)

    epoch_left = total_epochs - curr_epoch
    time_left = epoch_left * time
    time_left_min = int(time_left / 60)
    time_left_sec = int(time_left % 60)

    return minutes, seconds, time_left_min, time_left_sec    
    
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
        return x + self.dropout(sublayer(self.norm(x)))

class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# def create_src_mask(src, device):
#     " source = [배치 사이즈, 소스 문장 길이] "

#     src_len = src.size(1)
    
#     src_mask = (src == PAD_IDX)
#     # src_mask = [배치 사이즈, 소스 문장 길이]
    
#     src_mask = src_mask.unsqueeze(1).repeat(1, src_len, 1)
#     # src_mask = [배치 사이즈, 소스 문장 길이, 소스 문장 길이]

#     return src_mask.to(device)


def create_mask(src, tgt, device):
    " src = [배치 사이즈, 소스 문장 길이] "
    " tgt = [배치 사이즈, 타겟 문장 길이] "
    
    batch_size, tgt_len = tgt.size()
    
    subsequent_mask = generate_square_subsequent_mask(tgt, device)
    
    enc_dec_mask = (src == PAD_IDX)
    tgt_mask = (tgt == PAD_IDX)
    # src_mask = [배치 사이즈, 소스 문장 길이]
    # tgt_mask = [배치 사이즈, 타겟 문장 길이]
    
    enc_dec_mask = enc_dec_mask.unsqueeze(1).repeat(1, tgt_len, 1).to(device)
    tgt_mask = tgt_mask.unsqueeze(1).repeat(1, tgt_len, 1).to(device)
    # src_mask = [배치 사이즈, 타겟 문장 길이, 소스 문장 길이]
    # tgt_mask = [배치 사이즈, 타겟 문장 길이, 타겟 문장 길이]

    tgt_mask = tgt_mask | subsequent_mask
    
    return enc_dec_mask, tgt_mask