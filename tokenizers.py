from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from collections import Counter
from torchtext.vocab import Vocab
from tokenizers import SentencePieceBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

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

def get_BPE_tokenizers(sentences, vocab_size=20000, min_frequency=3):
  ko_tokenizer = SentencePieceBPETokenizer()
  ko_tokenizer.pre_tokenizer = Whitespace()
  ko_tokenizer.train_from_iterator(sentences, 
                               vocab_size=vocab_size,
                               min_frequency=min_frequency,
                               special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]'])
