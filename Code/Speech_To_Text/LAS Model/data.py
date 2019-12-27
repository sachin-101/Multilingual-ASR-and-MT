import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SpeechDataset(Dataset):
    def __init__(self, df, data_dir, char_to_token, n_fft=2048, hop_length=512):
        """
            df - dataframe from which clips have to be loaded
            data_dir - directory where clips are stored
        """
        self.df = df
        self.data_dir = data_dir
        self.char_to_token = char_to_token
        
        self.specgram = MelSpectrogram(n_fft=n_fft, hop_length=hop_length)  

        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # preparing audio data
        path = os.path.join(self.data_dir, self.df['path'].iloc[idx])
        waveform, sample_rate = torchaudio.load(path)
        X = self.specgram(waveform)

        # Normalize input
        X = (X - X.mean())
        X = X/X.abs().max()

        # preparing target
        sent = self.df['sent'].iloc[idx]
        tokens = []
        for c in sent:
            try:
                tokens.append(self.char_to_token[c])
            except:
                tokens.append(self.char_to_token['<unk>'])
        y = torch.tensor([self.char_to_token['<sos>']] + tokens + [self.char_to_token['<eos>']])
        return X, y

  
class AudioDataLoader(DataLoader):

    def __init__(self, pad_token, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs, collate_fn=self.my_collate_fn)
        self.pad_token = kwargs.get('pad_token', 0)

    
    def my_collate_fn(self, data):
        """
            overwriting collate_fn method, to pad the
            different length audio present in dataset

            #TODO : Implement BucketIterator, to load
                    samples, with same length together
        """
        specgrams = [x.squeeze(0).permute(1, 0) for (x, y) in data]
        targets = [y for (x, y) in data]
        X = pad_sequence(specgrams).permute(1, 0, 2)    # (N, T, H)
        Y = pad_sequence(targets, padding_value=self.pad_token).permute(1, 0)         # (N, L)
        return X, Y

        # If using ctc loss: sort, pack, pad_packed, returns -> packed, true_lengths
        # X, Y = zip(*sorted(zip(specgrams, targets),     # sort w.r.t to timesteps in spectrogram
        #                     key=lambda x:x[0].shape[2], 
        #                     reverse=True))
