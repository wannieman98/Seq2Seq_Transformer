from soynlp import tokenizer
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from tokenizers import SentencePieceBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from typing import Iterable, List


def get_ko_words(sentences):
  word_extractor = WordExtractor(min_frequency=100,
      min_cohesion_forward=0.05, 
      min_right_branching_entropy=0.0
  )
  word_extractor.train(sentences) # list of str or like
  words = word_extractor.extract()
  return words

def get_ko_tokenizer(sentences):
  words = get_ko_words(sentences)
  cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
  return LTokenizer(scores=cohesion_score)

def build_vocab(sentences, tokenizer):
  counter = Counter()
  for sentence in sentences:
    counter.update(tokenizer.encode(sentence).tokens)
  return Vocab(counter, specials=['<pad>', '<sos>', '<eos>', '<unk>'])

def get_BPE_tokenizer(sentences, vocab_size=20000, min_frequency=3):
  tokenizer = SentencePieceBPETokenizer()
  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train_from_iterator(sentences, 
                               vocab_size=vocab_size,
                               min_frequency=min_frequency,
                               special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]'])

  return tokenizer

def build_vocabs(sentences: List[str], src_tokenizer, tgt_tokenizer):
  tokens = {'SRC_LANGUAGE': src_tokenizer, 'TGT_LANGUAGE': tgt_tokenizer}
  vocabs = {}

  PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

  special_symbols = ['<pad>', '<sos>', '<eos>', '<unk>']

  for ln in ['SRC_LANGUAGE', 'TGT_LANGUAGE']:
    vocabs[ln] = build_vocab_from_iterator(iterator=yield_tokens(sentences, ln, tokens))

  for ln in ['SRC_LANGUAGE', 'TGT_LANGUAGE']:
    vocabs[ln].set_default_index(UNK_IDX)

  return vocabs

  
def yield_tokens(data_iter: Iterable, language: str, tokens: dict) -> List[str]:
  language_index = {'SRC_LANGUAGE': 0, 'TGT_LANGUAGE': 1}
  
  data_iter = data_iter[language_index[language]]

  for data_sample in data_iter:
      yield tokens[language](data_sample)