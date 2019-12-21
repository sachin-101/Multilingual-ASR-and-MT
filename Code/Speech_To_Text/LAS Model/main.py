import os
import datetime
import torch
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import SpeechDataset, CHARS
from listener import Listener
from attend_and_spell import AttendAndSpell
from seq2seq import Seq2Seq
from utils import collate_fn, train



if __name__ == "__main__":
    
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    train_df = pd.read_csv('train_df.csv', names=['id', 'sent'])
    test_df = pd.read_csv('test_df.csv', names=['id', 'sent'])
    dataset_dir = 'dataset'

    tensorboard_dir = os.path.join('tb_summary')
    train_dataset = SpeechDataset(train_df, dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_dataset = SpeechDataset(test_df, dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    input_size = 128    # num rows in instagram
    hidden_dim = 64    # 256*2 nodes in each LSTM
    num_layers = 3
    dropout = 0.1
    layer_norm = False   
    encoder = Listener(input_size, hidden_dim, num_layers, dropout=dropout, layer_norm=layer_norm)

    hid_sz = 64
    embed_dim = 15
    vocab_size = len(CHARS)
    decoder = AttendAndSpell(embed_dim, hid_sz, encoder.output_size, vocab_size)

    criterion = nn.CrossEntropyLoss()
    model = Seq2Seq(encoder, decoder, criterion, tf_ratio = 1.0, device=DEVICE) 


    # Let's start training
    epochs = 10
    # optimizer = optim.ASGD(model.parameters(), lr=0.2)  # lr = 0.2 used in paper
    optimizer = optim.Adadelta(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    log_interval = 5
    print_interval = 40

    summary_dir = os.path.join(tensorboard_dir, str(datetime.datetime.now()))
    print("summary dir:", summary_dir)
    writer = SummaryWriter(summary_dir)

    for epoch in range(epochs):
        print("\nTeacher forcing ratio:", model.tf_ratio)
        train(model, DEVICE, train_loader, optimizer, epoch, print_interval, writer, log_interval)
        scheduler.step()                                    # Decrease learning rate
        torch.save(model.state_dict(), f'las_model_small_{epoch}')
        model.tf_ratio = max(model.tf_ratio - 0.01, 0.8)    # Decrease teacher force ratio                       