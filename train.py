from constants import *
import trainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    mymodel = trainer.Trainer()
    mymodel.train()
    os.remove("constants.pyc")
    os.remove("constants.util.pyc")