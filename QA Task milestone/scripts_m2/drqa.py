import torch
import torch.nn as nn
import torch.nn.functional as F

class DrQAReader(nn.Module):
    """
    Simplified DrQA Document Reader for SQuAD-like QA.
    Modules:
      - Embedding layer
      - Separate BiLSTM encoders for context and question
      - Dot-product attention to fuse question into context
      - Pointer network (MLP) to predict start and end positions
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 300,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx  

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.ctx_rnn = nn.LSTM(embed_dim,
                               hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=True,
                               batch_first=True)
        self.qst_rnn = nn.LSTM(embed_dim,
                               hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=True,
                               batch_first=True)
        
        enc_dim = 2 * hidden_size  
        self.start_mlp = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, 1)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, 1)
        )

    def forward(self,
                context: torch.LongTensor,
                question: torch.LongTensor,
                attention_mask_question: torch.Tensor = None):
        # context: (batch, c_len)
        # question: (batch, q_len)
        # masks: True for PAD (optional)

        # 1) Embedding
        ctx_emb = self.dropout(self.embed(context))  # (B, c_len, D)
        qst_emb = self.dropout(self.embed(question))  # (B, q_len, D)

        # 2) Encode with BiLSTM
        ctx_enc, _ = self.ctx_rnn(ctx_emb)  # (B, c_len, 2H)
        qst_enc, _ = self.qst_rnn(qst_emb)  # (B, q_len, 2H)

        # 3) Dot-product attention
        # Compute similarity matrix S: (B, c_len, q_len)
        S = torch.bmm(ctx_enc, qst_enc.transpose(1, 2))
        if attention_mask_question is not None:
            # cast attention mask to boolean for masking
            pad_mask = (attention_mask_question == 0).unsqueeze(1)
            S = S.masked_fill(pad_mask, -1e9)
        alpha = F.softmax(S, dim=-1)  # (B, c_len, q_len)

        # 4) Attend question: weighted sum
        q_att = torch.bmm(alpha, qst_enc)  # (B, c_len, 2H)

        # 5) Fusion: concatenate context encoding and attended question
        fusion = torch.cat([ctx_enc, q_att], dim=-1)  # (B, c_len, 4H)

        # 6) Predict start logits, end logits
        start_logits = self.start_mlp(fusion).squeeze(-1)  # (B, c_len)
        end_logits = self.end_mlp(fusion).squeeze(-1)     # (B, c_len)

        return start_logits, end_logits

