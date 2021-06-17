import time
import random

from util import *
from data_utils import *
from constants import *

import torch
import torch.nn as nn
import torch.optim as optim

random.seed(5)
torch.manual_seed(5)


class Train:
    def __init__(self, file_path='Default',
                 emb_size=EMB_SIZE, nhead=NHEAD, ffn_hid_dim=FFN_HID_DIM, batch_size=BATCH_SIZE, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                 token_type=1):
        self.params = {'emb_size': emb_size,
                       'nhead': nhead,
                       'ffn_hid_dim': ffn_hid_dim,
                       'batch_size': batch_size,
                       'num_encoder_layers': num_encoder_layers,
                       'num_decoder_layers': num_decoder_layers}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kor, eng = get_kor_eng_sentences(file_path)
        self.sentences = {'src_lang': kor, 'tgt_lang': eng}
        self.tokens = get_tokens(self.sentences, token_type)
        self.vocabs = build_vocabs(self.sentencesm, self.tokens)
        self.train_iter = get_train_iter(self.sentences, self.tokens, self.vocabs, self.params['batch_size'])

        self.params['src_vocab_size'] = len(self.vocabs['src_lang'])
        self.params['tgt_vocab_size'] = len(self.vocabs['tgt_lang'])
        
        self.