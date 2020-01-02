import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tf_ratio, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tf_ratio = tf_ratio
        self.device = device

    def forward(self, data, target):
        Ty = target.shape[1]

        # forward propagte through encoder
        encoder_output = self.encoder(data)
        
        # initialising loss for batch
        loss = 0.0

        N = data.shape[0]   # batch size
        y_in = torch.zeros((N)).to(self.device, dtype=torch.int64)  # <sos>
        hidden_prev = [ ( torch.zeros((N, self.decoder.hid_sz)).to(self.device), 
                          torch.zeros((N, self.decoder.hid_sz)).to(self.device)  ),
                       ( torch.zeros((N, self.decoder.hid_sz)).to(self.device), 
                          torch.zeros((N, self.decoder.hid_sz)).to(self.device)  )
            ]
        context = torch.zeros((N, self.decoder.n_h)).to(self.device)

        output = []        
        for t in range(0, Ty):
            y_true = target[:, t]   
            y_out, hidden_prev, context = self.decoder(y_in, hidden_prev, encoder_output, context)
            output.append(y_out)
            teacher_force = random.random() < self.tf_ratio
            y_in = y_true if teacher_force else y_out.max(dim=1)[1]
        
        out = torch.stack(output, dim=1)
        loss = F.cross_entropy(out.view(N*Ty, -1), target.view(-1))
        return  loss, out

