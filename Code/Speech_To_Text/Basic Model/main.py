import os
import random
import numpy as np 
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data import SpeechDataset
from utils import train, test, check_on_personal
from model import BasicASR
from utils import train, test, check_on_personal

if __name__ == "__main__":
    

    chars = ['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    tokens = [i for i in range(len(chars))]
    tokenize_dict = {c:t for c,t in zip(chars, tokens)}
    token_to_char = {t:c for c,t in tokenize_dict.items()}

    print(tokenize_dict)
    print(token_to_char)


    # make dirs
    root_dir = '.'
    train_dir = os.path.join(root_dir, 'train_data')
    test_dir = os.path.join(root_dir, 'test_data')

    train_data_df = pd.read_csv(os.path.join(train_dir, 'train_data.csv'), 
                                skiprows=[0], 
                                header=None, 
                                names=['index', 'clip', 'sentence'])
    test_data_df = pd.read_csv(os.path.join(test_dir, 'test_data.csv'), 
                                skiprows=[0], 
                                header=None, 
                                names=['index', 'clip', 'sentence'])

    max_data_len = 2500
    max_sent_len = 100

    bs = int(input("Enter batch_size:"))

    train_dataset = SpeechDataset(train_data_df, train_dir, max_data_len, max_sent_len)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    test_dataset = SpeechDataset(test_data_df, test_dir, max_data_len, max_sent_len)
    test_loader = DataLoader(test_dataset, batch_size=bs)


    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu' 
    print('device:',device)
    input_len = 201
    hidden_size = 50
    num_layers = 3
    output_shape = 28
    bidirectional = True
    model = BasicASR(input_len, hidden_size, num_layers, output_shape, bidirectional).to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    epochs = 10
    log_interval = 20

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        val_loss = test(model, device, test_loader, log_interval)
        print('Validation loss', val_loss)
        scheduler.step(val_loss)
        print('-'*10)
        check_on_personal(model, device, 
                        random.randint(0, test_data_df.shape[0]), 
                        train_dataset, token_to_char)





