from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from fetch_tokenizers import *
from typing import List
from constants import *
import pandas as pd
import numpy as np
import torch
import os

def convert_to_csv(root, files):
  count = 1
  csv_files = []
  for filepath in files:
    if filepath[-4:] == "xlsx":
      xls = pd.read_excel(os.path.join(root, filepath), index_col=None)
      destination = 'data/spoken' + str(count) + '.csv'
      xls.to_csv(destination, encoding='utf-8', index=False)
      csv_files.append(destination)
  
  return csv_files

def convert_to_sentences(csv_files):
  dataframes = [ pd.read_csv(filepath) for filepath in csv_files ]
  data = pd.concat(dataframes, ignore_index=True)

  kor_sentences = []
  eng_sentences = []
  for index, sent in data.iterrows():
    _, kor, eng = sent
    kor_sentences.append(kor)
    eng_sentences.append(eng)
  for kor, eng in zip(kor_sentences[:5], eng_sentences[:5]):
      print(f'[KOR]: {kor}')
      print(f'[ENG]: {eng}\n')
  print(f'[KOR LEN]: {len(kor_sentences)}')
  print(f'[ENG LEN]: {len(eng_sentences)}')
  
  return kor_sentences, eng_sentences

def get_kor_eng_sentences(file_path="data"):
  root = "data"
  files = os.listdir("./data")
  csv_files = convert_to_csv(root, files)
  return convert_to_sentences(csv_files)
    

def divide_files(sentences):
  train_mask = np.random.rand(len(sentences[0])) < 0.8

  train = [ sentence[train_mask] for sentence in sentences ]
  remaining_data = [ sentence[~train_mask] for sentence in sentences ]

  val_test_mask = np.random.rand(len(remaining_data[0])) < 0.5

  val = [ remaining_data[val_test_mask] for data in remaining_data ]
  test = [ remaining_data[~val_test_mask] for data in remaining_data ]

  print("train data length: {}".format(len(train)))
  print("validation data length: {}".format(len(val)))
  print("test data length: {}".format(len(test)))

  return train, val, test

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def get_text_transform(tokens, vocabs):
  text_transform = {}
  for ln in ['src_lang', 'tgt_lang']:
      text_transform[ln] = sequential_transforms(tokens[ln], 
                                                vocabs[ln], 
                                                tensor_transform)
  return text_transform

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def data_process(sentences, vocabs, tokens):
  output = []
  text_transform = get_text_transform(tokens, vocabs)
  for kor, eng in zip(sentences['src_lang'], sentences['tgt_lang']):
    ko_tensor = text_transform['src_lang'](kor.rstrip("\n"))
    en_tensor = text_transform['tgt_lang'](eng.rstrip("\n"))
    output.append((ko_tensor, en_tensor))
  print("data size: {}".format( len(output)))
  return output

def get_train_iter(sentences, tokens, vocabs, batch_size):
  train_data = data_process(sentences, vocabs, tokens)
  return DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)