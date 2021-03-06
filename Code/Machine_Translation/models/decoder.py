import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    
    
    def __init__(self, embedding_matrix, hidden_size, encoder_out_size, vocab_size):
        """
            hidden_size: units in LSTM cell
            encoder_out_size: dim of encoder output
            vocab_size: dim of softmax output
        """

        super(Decoder, self).__init__()    
        
        self.n_h = encoder_out_size
        self.hid_sz = hidden_size
        self.vocab_size = vocab_size

        # Note: shape of c : (N, 1, n_h)
        #       shape of s : (N, 1, hid_sz)
        #       shape of y : (N, 1, embed_dim)

        self.embed_layer, num_embed, embed_dim = self.create_embedding_layer(embedding_matrix)
        self.embed_dim = embed_dim
        
        self.attention_layer = Attention(self.n_h, self.hid_sz)
        
        self.pre_lstm_cell = nn.LSTMCell(self.n_h + self.embed_dim, self.hid_sz)
        self.post_lstm_cell = nn.LSTMCell(self.hid_sz + self.n_h, self.hid_sz)

        self.mlp = nn.Sequential(
            nn.Linear(self.hid_sz, vocab_size),
            nn.ReLU(),
            nn.BatchNorm1d(vocab_size),
            nn.Softmax(dim=1)
        )
    
    
    @staticmethod
    def create_embedding_layer(embed_matrix, trainable=False):
        num_embeddings, embedding_dim = embed_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        emb_layer.load_state_dict({'weight':embed_matrix})
        if trainable:
            raise NotImplementedError
            # emb_layer.weight.requires_grad = True
        emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    
    
    
    def forward(self, yt_prev, hidden_prev, encoder_output, c_prev):
        """
            Decode a single time step
        """

        # s_i = RNN(y_i-1, c_i-1, s_i-1)
        yt_prev = self.embed_layer(yt_prev)
        rnn_input = torch.cat([yt_prev, c_prev], dim=1)
        h_0, c_0 = self.pre_lstm_cell(rnn_input, hidden_prev[0])
        s_i = h_0

        # context vector: c_i = AttentionContext(encoder_out, s_i)
        context = self.attention_layer(encoder_output, s_i)
        
        # concat s_i and c_i and feed to Spell
        spell_input = torch.cat([s_i, context], dim=1)
        h_1, c_1 = self.post_lstm_cell(spell_input, hidden_prev[1])
        out = self.mlp(h_1)
        return out, [(h_0, c_0), (h_1, c_1)], context  



class Attention(nn.Module):

    def __init__(self, n_h, n_s):
        super(Attention, self).__init__()
        ip_size = n_h + n_s
        self.linear1 = nn.Linear(ip_size, int(ip_size/2))
        self.linear2 = nn.Linear(int(ip_size/2), 1)
        
    def forward(self, h, s_prev):
        """
        Args:
            h -- endoder output (N, Tx, n_h)
            s_prev -- previous hidden state of the Decoder (N, n_s)
        """        
        Tx = h.shape[1] # batch size, N
        s_prev = s_prev.unsqueeze(dim=1).expand(-1, Tx, -1)   # (N, Tx, n_s)
        concat = torch.cat([h, s_prev], dim=2)  

        e = F.relu(self.linear1(concat))    # (N, Tx)
        alphas = F.softmax(self.linear2(e), dim=1)    # sum(alphas) = 1, over Tx axis

        context = torch.bmm(alphas.squeeze(dim=2).unsqueeze(dim=1), h)  # (N, 1, Tx)*(N, Tx, n_h)->(N, 1, n_h)
        return context.squeeze(dim=1)
