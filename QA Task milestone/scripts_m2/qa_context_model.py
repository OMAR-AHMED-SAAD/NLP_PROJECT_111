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

    def forward(self, context_question: torch.Tensor) -> tuple:
        """
        Args:
            context_question: Tensor of shape (batch_size, seq_len)

        Returns:
            start_logits: (batch_size, seq_len)
            end_logits:   (batch_size, seq_len)
        """
        embedded = self.embedding(context_question)  # (batch, seq_len, embedding_dim)
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





class RNN_QA_Model2(nn.Module):

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
        super(RNN_QA_Model2, self).__init__()

        self.context_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        ) if pretrained_embeddings is None else nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=freeze_embeddings, sparse=True)
        
        self.question_embedding = nn.Embedding(
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
            self.rnn2 = nn.LSTM(embedding_dim,
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
            self.rnn2 = nn.GRU(embedding_dim,
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
            self.rnn2 = nn.RNN(embedding_dim,
                               rnn_hidden,
                               num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)

        self.linear1 = nn.LazyLinear(1024)
        self.fc_start = nn.LazyLinear(output_dim)
        self.fc_end = nn.LazyLinear(output_dim)

    def forward(self, context: torch.Tensor, question: torch.Tensor) -> tuple:
        """
        Args:
            context_ids: Tensor of shape (batch_size, context_len)
            question_ids: Tensor of shape (batch_size, question_len)

        Returns:
            start_logits: (batch_size, seq_len)
            end_logits:   (batch_size, seq_len)
        """
        context_embedded = self.context_embedding(context)
        question_embedded = self.question_embedding(question)

        context_out, h_t_context = self.rnn(context_embedded)  # (batch, seq_len, hidden_dim)
        question_out, h_t_question = self.rnn2(question_embedded)  # (batch, seq_len, hidden_dim)
        if self.rnn_type == "LSTM":
            h_t, c_t = h_t_context
            h_t2, c_t2 = h_t_question
        

        # h_n has shape (num_layers * num_directions, batch_size, rnn_hidden)
        if self.bidirectional:
            forward_hidden = h_t[-2]  # forward direction from the last layer
            backward_hidden = h_t[-1] 
            # Concatenate to form a vector of shape (batch_size, 2 * rnn_hidden) == (batch_size, hidden_dim)
            h_last_context = torch.cat((forward_hidden, backward_hidden), dim=-1)

            forward_hidden2 = h_t2[-2]  # forward direction from the last layer
            backward_hidden2 = h_t2[-1]
            # Concatenate to form a vector of shape (batch_size, 2 * rnn_hidden) == (batch_size, hidden_dim)
            h_last_question = torch.cat((forward_hidden2, backward_hidden2), dim=-1)

        else:
            h_last_context = h_last_context[-1] # (batch_size, rnn_hidden) == (batch_size, hidden_dim) if bidirectional=False
            h_last_question = h_last_question[-1]
        # Concatenate context and question hidden states
        h_last = torch.cat((h_last_context, h_last_question), dim=-1)
        # Apply dropout to the concatenated hidden state
        h_last = torch.dropout(h_last, p=0.1, train=self.training)

        combined_out = self.linear1(h_last)
        combined_out = torch.relu(combined_out) 
        combined_out = torch.dropout(combined_out, p=0.1, train=self.training)
        start_logits = self.fc_start(combined_out)
        end_logits = self.fc_end(combined_out)

        return start_logits, end_logits