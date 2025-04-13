import torch
import torch.nn as nn


class RNN_QA_Model(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        bidirectional: bool = True,
        pad_idx: int = 1,
        output_dim: int = 1,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super(RNN_QA_Model, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        ) if pretrained_embeddings is None else nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=freeze_embeddings, sparse=True)
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        rnn_hidden = hidden_dim // 2 if bidirectional else hidden_dim

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim,
                               rnn_hidden,
                               num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim,
                              rnn_hidden,
                              num_layers,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(embedding_dim,
                              rnn_hidden,
                              num_layers,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidirectional)

        self.fc_start = nn.Linear(hidden_dim, output_dim)
        self.fc_end = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor) -> tuple:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)

        Returns:
            start_logits: (batch_size, seq_len)
            end_logits:   (batch_size, seq_len)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        _, h_t = self.rnn(embedded)  # (batch, seq_len, hidden_dim)
        if self.rnn_type == "LSTM":
            h_t, c_t = h_t

        # h_n has shape (num_layers * num_directions, batch_size, rnn_hidden)
        if self.bidirectional:
            forward_hidden = h_t[-2]  # forward direction from the last layer
            backward_hidden = h_t[-1]  # backward direction from the last layer
            # Concatenate to form a vector of shape (batch_size, 2 * rnn_hidden) == (batch_size, hidden_dim)
            h_last = torch.cat((forward_hidden, backward_hidden), dim=-1)
        else:
            h_last = h_last[
                -1]  # (batch_size, rnn_hidden) == (batch_size, hidden_dim) if bidirectional=False

        start_logits = self.fc_start(h_last)  # (batch, seq_len)
        end_logits = self.fc_end(h_last)  # (batch, seq_len)

        return start_logits, end_logits
