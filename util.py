import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def epoch_time(time, curr_epoch, total_epochs):
    minutes = int(time / 60)
    seconds = int(time % 60)

    epoch_left = total_epochs - curr_epoch
    time_left = epoch_left * time
    time_left_min = int(time_left / 60)
    time_left_sec = int(time_left % 60)

    return minutes, seconds, time_left_min, time_left_sec    

def attention(query, key, value, mask=None, dropout=None):
    q_k = torch.matmul(query, key.transpose(-2, -1))
    scores = torch.div(q_k, math.sqrt(query.size(-1)))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    output = F.softmax(scores, dim=-1)
    if dropout is not None:
        output = dropout(output)
    
    return torch.matmul(output, value), output



class ScheduledOptim:
    def __init__(self, optimizer, warmup_steps, hidden_dim):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.step_num = 0
        self.warmup_steps = warmup_steps
    
    def step(self):
        self.step_num += 1
        lr = self.init_lr * self.get_scale()
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr
            
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_scale(self):
        return np.min([
            np.power(self.step_num, -0.5),
            self.step_num * np.power(self.warmup_steps, -1.5)
        ])
    
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