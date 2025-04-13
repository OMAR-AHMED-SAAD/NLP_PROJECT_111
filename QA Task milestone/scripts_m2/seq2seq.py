import torch
import torch.nn as nn
from typing import Tuple, Union


class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 num_layers: int=1, 
                 rnn_type: str="RNN", 
                 dropout: float=0.0):
        super(Encoder, self).__init__()
        assert rnn_type in ["RNN", "LSTM", "GRU"]
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"Encoder input shape: {x.shape}")  
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        # print(f"Encoder Embedded shape: {embedded.shape}")
        outputs, h_t = self.rnn(embedded) # outputs: (batch, seq_len, hidden_dim), hidden: (num_layers, batch, hidden_dim)
        # print(f"Encoder output shape: {outputs.shape}, hidden shape: {h_t[0].shape}")  
        return outputs, h_t  

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int=1, rnn_type: str="RNN", dropout: float=0.0):
        super(Decoder, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h_t: Union[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len) - input token at time step t
        # print(f"Decoder input shape: {x.shape}")
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        # print(f"Decoder Embedded shape: {x.shape}")
        output, h_t = self.rnn(x, h_t) # output: (batch_size, seq_len, hidden_dim), hidden: (num_layers, batch_size, hidden_dim)
        # print(f"Decoder output shape: {output.shape}, hidden shape: {h_t[0].shape}")
        prediction = self.fc(output)  # (batch_size, seq_len, vocab_size)
        # print(f"Decoder prediction shape: {prediction.shape}")
        return prediction, h_t 

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, vocab_size: int, sos_token: int=1, eos_token: int=4):
        super(Seq2Seq, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)

        encoder_outputs, encoder_hidden = self.encoder(src) # (batch_size, src_len, hidden_dim), (num_layers, batch_size, hidden_dim)
        decoder_outputs, decoder_hidden = self.decoder(trg, encoder_hidden) # (batch_size, trg_len, vocab_size), (num_layers, batch_size, hidden_dim)
        return decoder_outputs

    
    def predict(self, src: torch.Tensor, max_len: int=25) -> torch.Tensor:
        batch_size = src.size(0)
        trg = torch.zeros(batch_size, max_len+1).long().to(self.device)
        trg[:, 0] = self.sos_token 
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0].unsqueeze(1)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        for t in range(1, max_len+1):
            output, hidden = self.decoder(input, hidden)
            top1 = output[:,-1,:].argmax(axis=-1)
            # print(f"Decoder output shape: {output.shape}, top1 shape: {top1.shape}")
            trg[:, t] = top1
            finished |= (top1 == self.eos_token)
            if finished.all():
                break
            input = trg[:, :t+1]
            print("-" * 50)

        for i in range(trg.size(0)):
            if finished[i]:
                eos_idx = (trg[i] == self.eos_token).nonzero(as_tuple=True)[0]
                print(f"Finished: {finished[i]}, eos_idx: {eos_idx}")
                if eos_idx.numel() > 0:
                    eos_index = eos_idx[0].item()
                    trg[i, eos_index + 1:] = self.eos_token  # set all tokens after eos to eos token
        return trg
