import os
import argparse

import torch
import random
import pickle
import pandas as pd
import torch.nn as nn

from data import SpeechDataset, AudioDataLoader
from listener import Listener
from attend_and_spell import AttendAndSpell
from seq2seq import Seq2Seq


def parse_args():
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument("--load_dir")
    parser.add_argument("--epoch", type=int)
    return parser.parse_args()

def load_model(load_dir, epoch, vocab_size, device):
    with open(os.path.join(load_dir, 'info.txt'), 'rb') as f:
        hparams = pickle.load(f)    # load model info

    encoder = Listener(hparams['input_size'], hparams['hidden_dim'], 
                        hparams['num_layers'], dropout=hparams['dropout'], 
                        layer_norm=hparams['layer_norm'])

    decoder = AttendAndSpell(hparams['embed_dim'], hparams['hid_sz'], 
                            encoder.output_size, vocab_size)

    criterion = nn.CrossEntropyLoss()
    model = Seq2Seq(encoder, decoder, criterion, tf_ratio = 1.0, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join(load_dir, f'las_model_{epoch}')))
    return model


def decode_pred_sent(out):
    pred_sent = []
    for t in out:
        lol = t.max(dim=1)[1].item()
        pred_sent.append(token_to_char[lol])
    return ''.join(pred_sent)


def decode_true_sent(y):
    sent = []
    for t in y:
        sent.append(token_to_char[t.item()])
    return ''.join(sent)


if __name__ == "__main__":

    args = parse_args()
    dataset_dir = '../../../Dataset/data_aishell'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train_df.csv'), names=['id', 'sent'])
    train_df = train_df.dropna(how='any')
    # test_df = pd.read_csv('test_df.csv', names=['id', 'sent'])
    
    save_file = os.path.join('train_utils', 'chars')
    with open(save_file, 'rb') as f:
            chars = pickle.load(f) # load file
    char_to_token = {c:i for i,c in enumerate(chars)} 
    token_to_char = {i:c for c,i in char_to_token.items()}
    sos_token = char_to_token['<sos>']
    eos_token = char_to_token['<eos>']
   
    # #test_dataset = SpeechDataset(test_df, dataset_dir)
    # #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = load_model(args.load_dir, args.epoch, vocab_size=len(chars), device=DEVICE)
    
    num_sent = 10
    model.eval()
    model.tf_ratio = 0.9

    for _ in range(num_sent):
        
        idx = random.randint(0, train_df.shape[0])
        trial_dataset = SpeechDataset(train_df, dataset_dir, sos_token, char_to_token, eos_token)

        x, y = trial_dataset.__getitem__(idx)
        # plt.imshow(x[0,:,:].detach())

        # Model output
        target = y.unsqueeze(dim=0).to(DEVICE)
        data = x.permute(0, 2, 1).to(DEVICE)
        loss, output = model(data, target)
        print("True sent : ", decode_true_sent(y))
        print("Pred sent : ", decode_pred_sent(output))
        print("Loss :", loss.item())    
        print("\n")