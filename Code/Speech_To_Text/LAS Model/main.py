import os
import datetime
import torch
import random
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import SpeechDataset, AudioDataLoader
from listener import Listener
from attend_and_spell import AttendAndSpell
from seq2seq import Seq2Seq
from utils import  train



def get_chars(lang, train_df=None):

    if lang=='eng':
        chars = ['<sos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', \
                'y', 'z', ' ', "'", '<eos>', '<pad>']
    elif lang=='chinese':
        chars = [' ', '<sos>']
        for idx in range(train_df.shape[0]):
            _, sent = train_df.iloc[idx]
            for c in sent:
                if c not in chars:
                    chars.append(c)
        chars = chars + ['<eos>', '<pad>', '<unk>']        
    else:
        raise NotImplementedError
    
    print('Number of chars', len(chars))
    return chars


if __name__ == "__main__":
    
    
    dataset_dir = '../../../Dataset/data_aishell'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('DEVICE :', DEVICE)
    
    # data = []
    # files = os.listdir(dataset_dir)
    # for f in files:
    #     if '.txt' in f:
    #         with open(os.path.join(dataset_dir, f), 'r') as text:
    #             data.append((f.replace('.txt', ''), text.readline()))
                
    # train_df = pd.DataFrame(data, columns=['id', 'sent'])
    # train_df.to_csv(os.path.join(dataset_dir, 'train_df.csv'), header=None)
    # print(train_df.head())

    # read the transcript
    # transcript_dir = '../../../Dataset'
    # with open(os.path.join(transcript_dir, 'aishell_transcript_v0.8.txt')) as f:
    #     data_list = f.readlines()

    # data = []
    # for example in data_list:
    #     id_, sent = str(example.split(' ')[0]), str(' '.join(example.split(' ')[1:-1])) # -1 to remove '\n'
    #     data.append((id_, sent))

    # print('Num examples:', len(data))
    # data_df = pd.DataFrame(data, columns=['id', 'sent'])
    # data_df.to_csv(os.path.join(transcript_dir, 'data_aishell', 'train_df.csv'))


     
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train_df.csv'), names=['id', 'sent'])
    train_df = train_df.dropna(how='any')
    print(train_df.head())
    # test_df = pd.read_csv('test_df.csv', names=['id', 'sent'])
    
    
    chars = get_chars('chinese', train_df)
    char_to_token = {c:i for i,c in enumerate(chars)} 
    token_to_char = {i:c for c,i in char_to_token.items()}
    sos_token = char_to_token['<sos>']
    eos_token = char_to_token['<eos>']
    pad_token = char_to_token['<pad>']
   
    tensorboard_dir = os.path.join('tb_summary')
    train_dataset = SpeechDataset(train_df, dataset_dir, sos_token, char_to_token, eos_token)
    train_loader = AudioDataLoader(pad_token, train_dataset, batch_size=32, shuffle=True, drop_last=True)

    #test_dataset = SpeechDataset(test_df, dataset_dir)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    input_size = 128    # num rows in instagram
    hidden_dim = 64    # 256*2 nodes in each LSTM
    num_layers = 3
    dropout = 0.1
    layer_norm = False   
    encoder = Listener(input_size, hidden_dim, num_layers, dropout=dropout, layer_norm=layer_norm)

    hid_sz = 64
    embed_dim = 15
    vocab_size = len(chars)
    decoder = AttendAndSpell(embed_dim, hid_sz, encoder.output_size, vocab_size)

    criterion = nn.CrossEntropyLoss()
    model = Seq2Seq(encoder, decoder, criterion, tf_ratio = 1.0, device=DEVICE).to(DEVICE)


    # Let's start training
    epochs = 20
    # optimizer = optim.ASGD(model.parameters(), lr=0.2)  # lr = 0.2 used in paper
    optimizer = optim.Adadelta(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    log_interval = 5
    print_interval = 40

    time = str(datetime.datetime.now())
    save_dir = os.path.join('Trained Models', f'Training_{time}')
    os.mkdir(save_dir)
    summary_dir = os.path.join(tensorboard_dir, time)
    writer = SummaryWriter(summary_dir)

    for epoch in range(epochs):
        print("\nTeacher forcing ratio:", model.tf_ratio)
        train(model, DEVICE, train_loader, optimizer, epoch, print_interval, writer, log_interval)
        scheduler.step()                                    # Decrease learning rate
        torch.save(model.state_dict(), os.path.join(save_dir, f'las_model_{epoch}'))
        model.tf_ratio = max(model.tf_ratio - 0.01, 0.8)    # Decrease teacher force ratio                       