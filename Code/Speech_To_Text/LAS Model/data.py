import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SpeechDataset(Dataset):
    def __init__(self, df, data_dir, sos_token, char_to_token, eos_token, device, file_extension='.wav'):
        """
            df - dataframe from which clips have to be loaded
            data_dir - directory where clips are stored
        """
        self.df = df
        self.data_dir = data_dir
        self.specgram = MelSpectrogram()  # returns spectogram of raw audio
        self.sos_token = sos_token
        self.char_to_token = char_to_token
        self.eos_token = eos_token
        self.file_extension = file_extension
        self.device = device
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # preparing audio data
        filename = os.path.join(self.data_dir, self.df['id'].iloc[idx])+self.file_extension
        waveform, sample_rate = torchaudio.load(filename)
        x = self.specgram(waveform).to(self.device)

        X = x.log2().clamp(min=-50) # avoid log(0)=-inf
        # Normalize input
        X = (X - X.mean())
        X = X/X.abs().max()

        # preparing target
        sent = self.df['sent'].iloc[idx].lower().replace('\n', '')
        tokens = []
        for c in sent:
            try:
                tokens.append(self.char_to_token[c])
            except:
                tokens.append(self.char_to_token['<unk>'])
        y = torch.tensor([self.sos_token] + tokens + [self.eos_token])
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
