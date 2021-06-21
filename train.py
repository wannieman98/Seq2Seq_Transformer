from constants import *
import trainer
import argparse

def main(config):
    model = trainer.Trainer(file_path=config.f, token_type=config.t, load=config.l,
                            emb_size=config.emb_size, num_epoch=config.num_epoch, 
                            ffn_hid_dim=config.ffn_hid_dim, batch_size=config.batch_size,
                            num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers)
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Machine Translation')
    parser.add_argument('--t', type=int, default=1, choices=[1, 2])
    parser.add_argument('--f', type=str, default="Default")
    parser.add_argument('--l', type=bool, default="False", choices=[True, False])
    parser.add_argument('--num_epoch', type=int, default=15)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--ffn_hid_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    args = parser.parse_args()
    main(args)