import os
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset

import torchaudio
from torchaudio.transforms import Spectrogram

class SpeechDataset(Dataset):
    def __init__(self, df, data_dir, max_t=-1, max_sent=-1):
        """
            df - dataframe from which clips have to be loaded
            data_dir - directory where clips are stored
            max_t - maximum time steps of audio in dataset
            max_sent - max target/sentence len 
        """
        self.df = df
        self.data_dir = data_dir
        self.specgram = Spectrogram()  # returns spectogram of raw audio
        self.max_t = max_t
        self.max_sent = max_sent
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx, pad=True):
        # preparing audio data
        filename = os.path.join(self.data_dir, self.df['clip'].iloc[idx])
        waveform, sample_rate = torchaudio.load(filename)
        x = self.specgram(waveform)

        # preparing target
        sent = self.df['sentence'].iloc[idx]
        sent = list(map(int, sent.split(',')))
        
        if pad:
            input_length = x.shape[2]
            x = F.pad(x, (0,self.max_t-x.shape[2]), "constant", 0)

            X = x.log2().clamp(min=-50) # avoid log(0)=-inf
            # Normalize input
            X = (X - X.mean())
            X = X/X.abs().max()

            # pad target
            target_length = len(sent)
            target = F.pad(torch.tensor(sent), (0, self.max_sent-len(sent)), 
                            'constant', 0)
            return (X, target, input_length, target_length)
        else:
            return (x, sent)
