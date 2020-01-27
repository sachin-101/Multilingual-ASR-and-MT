import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.las_model.listener import Listener
from models.las_model.attend_and_spell import AttendAndSpell

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
        attention_vectors = []
        
        for t in range(0, Ty):
            y_true = target[:, t]   
            y_out, hidden_prev, context, alphas = self.decoder(y_in, hidden_prev, encoder_output, context)
            output.append(y_out)
            teacher_force = random.random() < self.tf_ratio
            y_in = y_true if teacher_force else y_out.max(dim=1)[1]
            attention_vectors.append(alphas)
        
        out = torch.stack(output, dim=1)
        return out, attention_vectors

    
if __name__ == "__main__":

    hid_sz = 10
    embed_size = 20
    vocab_size = 5
    ip_size = 40
    embed_dim = 10

    encoder = Listener(ip_size, 20, 3)
    decoder = AttendAndSpell(embed_dim, hid_sz, encoder.output_size, vocab_size)

    X = torch.rand((32, 64, ip_size))
    Y = torch.randint(0, vocab_size, (32, 15))

    s2s = Seq2Seq(encoder, decoder, tf_ratio=1, device='cpu')
    pred_out, _ = s2s(X, Y)
    
    print(pred_out.shape)
    