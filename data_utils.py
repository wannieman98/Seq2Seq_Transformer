import pandas as pd
import numpy as np
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

def get_kor_eng_sentences():
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