import time
import random
from util import *
from constants import *
from data_utils import *
import torch.optim as optim
from pytorch_transformer import *

random.seed(5)
torch.manual_seed(5)

class Trainer:
    def __init__(self, file_path='Default', num_epoch=NUM_EPOCH,
                 emb_size=EMB_SIZE, nhead=NHEAD, ffn_hid_dim=FFN_HID_DIM, batch_size=BATCH_SIZE, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                 token_type=1, load=False):
        self.params = {'num_epoch': num_epoch,
                       'emb_size': emb_size,
                       'nhead': nhead,
                       'ffn_hid_dim': ffn_hid_dim,
                       'batch_size': batch_size,
                       'num_encoder_layers': num_encoder_layers,
                       'num_decoder_layers': num_decoder_layers}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kor, eng = get_kor_eng_sentences(file_path)
        self.sentences = {'src_lang': kor, 'tgt_lang': eng}
        self.tokens = get_tokens(self.sentences, token_type)
        self.vocabs = build_vocabs(self.sentences, self.tokens)
        self.train_iter = get_train_iter(self.sentences, self.tokens, self.vocabs, self.params['batch_size'])

        self.params['src_vocab_size'] = len(self.vocabs['src_lang'])
        self.params['tgt_vocab_size'] = len(self.vocabs['tgt_lang'])

        self.transformer = Transformer(self.params['num_encoder_layers'], self.params['num_decoder_layers'],
                                       self.params['emb_size'], self.params['nhead'], 
                                       self.params['src_vocab_size'], self.params['tgt_vocab_size'],
                                       self.params['ffn_hid_dim']).to(self.device)
        
        self.optimizer = ScheduledOptim(
            optim.Adam(self.transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
            warmup_steps=4000,
            hidden_dim=self.params['ffn_hid_dim']
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        if load == True:
            state_dict = torch.load('checkpoints/checkpoint.pth')
            self.transformer.load_state_dict(state_dict)
        else:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def train(self):
        self.transformer.train()

        train_loss = 0
        print("\nbegin training...")

        for epoch in range(self.params['num_epoch']):
            epoch_loss = 0
            start_time = time.time()

            for src, tgt in self.train_iter:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]

                self.optimizer.zero_grad()

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.device)

                logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

                tgt_out = tgt[1:, :]

                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))                    
                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.train_iter)

            end_time = time.time()

            print("Epoch: {}, Train loss: {}, Epoch time: {}".format(epoch, epoch_loss, end_time-start_time))
            
        torch.save(self.transformer.state_dict(), 'checkpoints/checkpoint.pth')