import torch 
import torch.nn as nn 

class NERModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embedding = nn.Embedding(kwargs['vocab_size'],kwargs['embedding_dim'])

        self.LSTM = nn.LSTM(
            kwargs['embedding_dim'],
            kwargs['hidden_dim'],
            bidirectional=True,
            batch_first=True
            )
        
        self.fc = nn.Linear(kwargs['hidden_dim']*2,kwargs['output_dim'])

    def forward(self,x):

        embedded = self.embedding(x)
        outputs, _ = self.LSTM(embedded)
        predictions = self.fc(outputs)

    