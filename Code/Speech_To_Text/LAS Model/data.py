import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader


CHARS = ['<sos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', \
        'y', 'z', ' ', "'", '<eos>', '<pad>']

char_to_token = {c:i for i,c in enumerate(CHARS)} 
token_to_char = {i:c for c,i in char_to_token.items()}


sos_token = char_to_token['<sos>']
eos_token = char_to_token['<eos>']
pad_token = char_to_token['<pad>']

class SpeechDataset(Dataset):
    def __init__(self, df, data_dir):
        """
            df - dataframe from which clips have to be loaded
            data_dir - directory where clips are stored
        """
        self.df = df
        self.data_dir = data_dir
        self.specgram = MelSpectrogram()  # returns spectogram of raw audio
      
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # preparing audio data
        filename = os.path.join(self.data_dir, self.df['id'].iloc[idx])+'.flac'
        waveform, sample_rate = torchaudio.load(filename)
        x = self.specgram(waveform)

        X = x.log2().clamp(min=-50) # avoid log(0)=-inf
        # Normalize input
        X = (X - X.mean())
        X = X/X.abs().max()

        # preparing target
        sent = self.df['sent'].iloc[idx].lower().replace('\n', '')
        y = torch.tensor([sos_token] + [char_to_token[c] for c in sent] + [eos_token])
        return X, y