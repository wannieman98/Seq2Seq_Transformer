import time
import random

from catalogue import create

from util import *
from data_utils import *
import torch.optim as optim
from pytorch_transformer import *

random.seed(5)
torch.manual_seed(5)    


class Trainer:
    def __init__(self, file_path, num_epoch,
                 emb_size, nhead, ffn_hid_dim, batch_size, 
                 n_layers, dropout, token_type, load, variation):
        self.params = {'num_epoch': num_epoch,
                       'emb_size': emb_size,
                       'nhead': nhead,
                       'ffn_hid_dim': ffn_hid_dim,
                       'batch_size': batch_size,
                       'n_layers': n_layers,
                       'dropout': dropout}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kor, eng = get_kor_eng_sentences(file_path)
        self.sentences = {'src_lang': kor, 'tgt_lang': eng}
        self.tokens = get_tokens(self.sentences, token_type)
        self.vocabs = build_vocabs(self.sentences, self.tokens)
        self.train_iter = get_train_iter(self.sentences, self.tokens, self.vocabs, self.params['batch_size'])


        self.params['src_vocab_size'] = len(self.vocabs['src_lang'])
        self.params['tgt_vocab_size'] = len(self.vocabs['tgt_lang'])


  
        self.transformer = build_model(vocabs=self.vocabs, nhead=self.params['nhead'], N=self.params['n_layers'],
                                       d_model= self.params['emb_size'], d_ff=self.params['ffn_hid_dim'],
                                       device=self.device, dropout=self.params['dropout'], load=load, variation=variation)
        
        self.optimizer = ScheduledOptim(
            optim.Adam(self.transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
            warmup_steps=4000,
            hidden_dim=self.params['ffn_hid_dim']
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


    def train(self):
        self.transformer.train()

        print("\nbegin training...")

        for epoch in range(self.params['num_epoch']):
            epoch_loss = 0
            start_time = time.time()

            for src, tgt in self.train_iter:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]

                self.optimizer.zero_grad()

                src_mask, tgt_mask = create_target_mask(src, tgt)

                logits = self.transformer(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

                tgt_out = tgt[1:, :]

                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))                    
                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.train_iter)

            end_time = time.time()

            minutes, seconds, time_left_min, time_left_sec = epoch_time(end_time-start_time, epoch, self.params['num_epoch'])

            print("Epoch: {}, Train_loss: {}".format(epoch, epoch_loss))
            print("Epoch time: {}m {}s, Time left for training: {}m {}s".format(minutes, seconds, time_left_min, time_left_sec))
            
        torch.save(self.transformer.state_dict(), 'checkpoints/new_script_checkpoint_inf.pth')
        torch.save(self.transformer, 'checkpoints/new_script_checkpoint_mod.pt')


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