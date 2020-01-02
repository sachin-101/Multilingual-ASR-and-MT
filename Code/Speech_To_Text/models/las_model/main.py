import os
import datetime
import torch
import random
import pickle
import pandas as pd

#os.chdir(os.path.join(os.getcwd(), 'LAS Model'))
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import SpeechDataset, AudioDataLoader
from listener import Listener
from attend_and_spell import AttendAndSpell
from seq2seq import Seq2Seq
from utils import  train



def get_chars(lang, save_file=None, train_df=None):
    if lang=='eng':
        chars = ['<sos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', \
                'y', 'z', ' ', "'", '<eos>', '<pad>']
    elif lang=='chinese':            
        try:
            with open(save_file, 'rb') as f:
                chars = pickle.load(f) # load file
        except FileNotFoundError:
            chars = [' ', '<sos>']
            for idx in range(train_df.shape[0]):
                _, sent = train_df.iloc[idx]
                for c in sent:
                    if c not in chars:
                        chars.append(c)
            chars = chars + ['<eos>', '<pad>']
            with open(save_file, 'wb') as f:
                pickle.dump(chars, f) # save file
    else:
        raise NotImplementedError
    
    print('Number of chars', len(chars))
    return chars


# Used when each sentence is in a separate text file
def make_train_df(dataset_dir):
    data = []
    files = os.listdir(dataset_dir)
    for f in files:
        if '.txt' in f:
            with open(os.path.join(dataset_dir, f), 'r') as text:
                data.append((f.replace('.txt', ''), text.readline()))
                
    train_df = pd.DataFrame(data, columns=['id', 'sent'])
    train_df.to_csv(os.path.join(dataset_dir, 'train_df.csv'), header=None)
    print(train_df.head())


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


# Used for ai_shell dataset, when all sentences are in a single text file
def read_transcript(transcript_dir):
    transcript_dir = '../../../Dataset'
    with open(os.path.join(transcript_dir, 'aishell_transcript_v0.8.txt')) as f:
        data_list = f.readlines()

    data = []
    for example in data_list:
        id_, sent = str(example.split(' ')[0]), str(' '.join(example.split(' ')[1:-1])) # -1 to remove '\n'
        data.append((id_, sent))

    print('Num examples:', len(data))
    data_df = pd.DataFrame(data, columns=['id', 'sent'])
    data_df.to_csv(os.path.join(transcript_dir, 'train_df.csv'))
    data_df.head()

if __name__ == "__main__":

    dataset_dir = '../../../../Dataset/data_aishell'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('DEVICE :', DEVICE)
    
  
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train_df.csv'), names=['path', 'sent'])
    train_df = train_df.dropna(how='any')
    print(train_df.head())
    # test_df = pd.read_csv('test_df.csv', names=['id', 'sent'])
    
    save_file = os.path.join('save', 'chars')
    chars = get_chars('chinese', save_file, train_df)
    char_to_token = {c:i for i,c in enumerate(chars)} 
    token_to_char = {i:c for c,i in char_to_token.items()}
    sos_token = char_to_token['<sos>']
    eos_token = char_to_token['<eos>']
    pad_token = char_to_token['<pad>']
   
    train_dataset = SpeechDataset(train_df, dataset_dir, char_to_token)
    train_loader = AudioDataLoader(pad_token, train_dataset, batch_size=32, shuffle=True, drop_last=True)

    # #test_dataset = SpeechDataset(test_df, dataset_dir)
    # #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

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

    hyperparams = {'input_size':input_size, 'hidden_dim':hidden_dim, 'num_layers':num_layers,
                    'dropout':dropout, 'layer_norm':layer_norm, 'hid_sz':hid_sz, 'embed_dim':embed_dim}
                        
    model = Seq2Seq(encoder, decoder, tf_ratio=1.0, device=DEVICE).to(DEVICE)
    
    # optimizer = optim.ASGD(model.parameters(), lr=0.2)  # lr = 0.2 used in paper
    optimizer = optim.Adadelta(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    log_interval = 5
    print_interval = 40

    
    epochs = 20
    # load = False
    # if load:
    #     saved_file = 'Trained_models/Training_2019-12-25 00:09:23.921978/las_model_6'
    #     model.load_state_dict(torch.load(saved_file))
    #     start_epoch = int(saved_file[-1]) + 1
    #     time = os.listdir(tensorboard_dir)[-1]  # use the last one
    # else:
    #     start_epoch = 0
    #     time = str(datetime.datetime.now())
    
    time = str(datetime.datetime.now())
    save_dir = os.path.join('save', f'Training_{time}')
    try:    
        os.mkdir(save_dir);
    except FileExistsError:
        pass

    TRAIN = True
    
    if TRAIN:
        writer = SummaryWriter(save_dir)

        eg_x, eg_y = next(iter(train_loader))
        writer.add_graph(model, (eg_x.to(DEVICE), eg_y.to(DEVICE)))
        # Saving hyperparmas
        with open(os.path.join(save_dir, 'info.txt'), 'wb') as f:
            pickle.dump(hyperparams, f)

        for epoch in range(epochs):
            print("\nTeacher forcing ratio:", model.tf_ratio)
            train(model, DEVICE, train_loader, optimizer, epoch, print_interval, writer, log_interval)
            scheduler.step()                                    # Decrease learning rate
            torch.save(model.state_dict(), os.path.join(save_dir, f'las_model_{epoch}'))
            model.tf_ratio = max(model.tf_ratio - 0.01, 0.8)    # Decrease teacher force ratio                       
    
    else:
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
