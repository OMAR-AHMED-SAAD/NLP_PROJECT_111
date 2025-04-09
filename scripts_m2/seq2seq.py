import torch
import torch.nn as nn
from typing import Tuple, List, Dict


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int=1, rnn_type: str="RNN", dropout: float=0.0):
        super(Encoder, self).__init__()
        assert rnn_type in ["RNN", "LSTM", "GRU"]
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden  # hidden will be (num_layers, batch, hidden_size)

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int=1, rnn_type: str="RNN", dropout: float=0.0):
        super(Decoder, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, 1) - input token at time step t
        x = self.embedding(x)  # (batch_size, 1, embed_dim)
        output, hidden = self.rnn(x, hidden)
        prediction = self.fc(output.squeeze(1))  # (batch_size, vocab_size)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, vocab_size: int):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)

        batch_size = trg.size(0)
        trg_len = trg.size(1)
        

        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0].unsqueeze(1)  # <sos> token for each sequence

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)

            input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs
    
    def prediction(self, src: torch.Tensor, max_len: int=20) -> List[int]:
        batch_size = src.size(0)
        trg = torch.zeros(batch_size, max_len).long().to(self.device)
        trg[:, 0] = 1
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0].unsqueeze(1)
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            top1 = output.argmax(1).unsqueeze(1)
            trg[:, t] = top1.squeeze(1)
            input = top1
        return trg
