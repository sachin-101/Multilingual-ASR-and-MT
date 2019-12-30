import torch.nn as nn


class BasicASR(nn.Module):
    def __init__(self, input_len, hidden_size, num_layers, output_shape, bidirectional):
        super(BasicASR, self).__init__()
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_shape) 
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x):
        """
            input shape : (batch_size, 1, 201, 2500)
            lstm input shape: (2500, batch_size, 201) # (seq_len, bs, input_len) 
            lstm output shape: (2500, batch_size, hidden_size) # hidden state from each timestep
            time distributed linear layer output: (2500, batch_size, output_size)
        """
        x = self.lstm(x.flatten(start_dim=1, end_dim=2).permute(2, 0, 1))
        x = self.softmax(self.linear(x))
        return x