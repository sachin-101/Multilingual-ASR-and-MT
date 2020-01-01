import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout=0.0,
                 bidirectional=True):
     
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # TODO: Check if adding Layer Norm imporoves performance

        self.embed_layer, _, embed_dim = self.create_embedding_layer(embedding_matrix)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=bidirectional,
                             dropout=dropout,  num_layers=num_layers)
        
        self.output_size = hidden_dim*2 if bidirectional else hidden_dim


    def forward(self, x):
        """
            x - padded sequence of input (batch_size, T, input_size)
        """
        x = self.embed_layer(x)
        x, _ = self.lstm(x)
        return x
    
    
    @staticmethod
    def create_embedding_layer(embed_matrix, trainable=False):
        num_embeddings, embedding_dim = embed_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':embed_matrix})
        if trainable:
            raise NotImplementedError
            #emb_layer.weight.requires_grad = True
        emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    

