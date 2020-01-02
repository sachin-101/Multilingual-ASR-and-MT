
import torch.nn as nn


class Listener(nn.Module):
    """
        Multi-layer LSTM with pyramidal structure 
        Performs time down sampling by factor 2 on every layer
    """
    def __init__(self, input_size, hidden_dim, num_layers, dropout=0.0,
                 bidirectional=True, layer_norm=False):
     
        super(Listener, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.b = bidirectional

        
        # Stacking normal lstm layers
        if self.num_layers[0] > 0:
            self.lstm_layers = nn.LSTM(input_size, hidden_dim, bidirectional=self.b, 
                                       num_layers=num_layers[0], 
                                       dropout=dropout if self.num_layers[0] > 1 else 0)
            out_dim = self.hidden_dim*2 if self.b else self.hidden_dim
            
        # Stacking piBLSTM layers
        self.pblstm_layers = nn.ModuleList()
        input_dim = out_dim if self.num_layers[0] > 0 else self.input_size
        for l in range(num_layers[1]):
            self.pblstm_layers.append(piBLSTM(input_dim, hidden_dim, bidirectional, dropout, layer_norm))
            input_dim = self.pblstm_layers[-1].out_dim
              
        self.output_size = self.pblstm_layers[-1].out_dim if self.num_layers[1] > 0 else out_dim
                                                                                    
            
    def forward(self, x):
        """
            x - padded sequence of input (batch_size, T, input_size)
        """
        
        if self.num_layers[0] > 0:
            x, _ = self.lstm_layers(x)
            
        for layer in self.pblstm_layers:
            x = layer(x)
        return x

class piBLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, bidir, dropout, layer_norm):
        super(piBLSTM, self).__init__()
        
        lstm_out_dim = 2*hidden_dim if bidir else hidden_dim
        self.out_dim = 2 * lstm_out_dim  # downsampling in T, upsampling in H
        self.dropout = dropout
        self.layer_norm = layer_norm

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidir, batch_first=True)
        
        # Layer norm : # https://arxiv.org/abs/1607.06450
        if self.layer_norm:
            self.ln = nn.LayerNorm(lstm_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        
        if not self.training:
            self.lstm.flatten_parameters()
        
        out, _ = self.lstm(x)

        if self.layer_norm:
            out = self.ln(out)
        if self.dropout > 0:
            out = self.dp(out)
    
        # ---- Time downsampling --------#

        b, t, h = out.shape
        if t % 2 == 0:
            out = out.reshape(b, int(t/2), int(h*2)).contiguous()
        else:
            out = out[:,:-1,:].reshape(b, int((t-1)/2), -1)  # drop last one from out

        return out