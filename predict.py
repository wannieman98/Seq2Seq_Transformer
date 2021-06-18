from constants import *
from data_utils import *
from util import *
import argparse

def trans(sent):
    kor, eng = get_kor_eng_sentences()
    sentences = {'src_lang': kor, 'tgt_lang': eng}
    tokens = get_tokens(sentences)
    vocabs = build_vocabs(sentences, tokens)
    model = trainer.Trainer(file_path=config.f, token_type=config.t, load=config.l)
    model.train()
    os.remove("constants.pyc")
    os.remove("constants.util.pyc")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Korean to English Translation')
    parser.add_argument('--input', type=str, default="나는 배고프다.", help="YOUR_INPUT")
    sent = parser.parse_args()
    trans(sent)