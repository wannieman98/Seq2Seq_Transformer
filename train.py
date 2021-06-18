from constants import *
import trainer
import argparse

def main(config):
    model = trainer.Trainer(file_path=config.f, token_type=config.t, load=config.l)
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Machine Translation')
    parser.add_argument('--t', type=int, default=1, choices=[1, 2])
    parser.add_argument('--f', type=str, default="Default")
    parser.add_argument('--l', type=bool, default="False")
    args = parser.parse_args()
    main(args)
