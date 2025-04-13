import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerQAModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int = 512,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 512, d_model))
        self.segment_embeddings = nn.Embedding(2, d_model)  # 0 = question, 1 = context

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # QA head
        self.qa_outputs = nn.Linear(d_model, 2)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.qa_outputs.weight, std=0.02)
        nn.init.zeros_(self.qa_outputs.bias)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len]
        token_type_ids: [batch_size, seq_len] - 0 for question, 1 for context
        attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        """

        seq_len = input_ids.size(1)

        tok_emb = self.token_embeddings(input_ids)                    # [B, L, D]
        seg_emb = self.segment_embeddings(token_type_ids)             # [B, L, D]
        pos_emb = self.position_embeddings[:, :seq_len, :]            # [1, L, D]

        # print("Token Embeddings Shape:", tok_emb.shape)
        # print("Segment Embeddings Shape:", seg_emb.shape)
        # print("Position Embeddings Shape:", pos_emb.shape)
        x = tok_emb + seg_emb + pos_emb
        x = self.dropout(x)

        # Transformer expects: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)

        # Create transformer attention mask: True for tokens to ignore
        attn_mask = attention_mask == 0  # [B, L] → 0 is padding, so mask=True

        x = self.encoder(x, src_key_padding_mask=attn_mask)
        x = x.transpose(0, 1)  # Back to [B, L, D]

        logits = self.qa_outputs(x)  # [B, L, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
