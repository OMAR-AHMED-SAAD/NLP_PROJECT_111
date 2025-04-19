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





class TransformerQAModel2(nn.Module):
    """
    Transformer-based QA model with cross-attention between context and question.
    Architecture:
      - Token + positional embeddings
      - Question encoding via TransformerEncoder
      - Cross-attention layers: context queries attend to question keys/values
      - Feed-forward network per layer
      - Start & end logits prediction via linear heads
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 ff_dim: int = 512,
                 num_q_layers: int = 2,
                 num_cross_layers: int = 2,
                 dropout: float = 0.1,
                 max_len: int = 160,
                 pad_idx: int = 1):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed_dim = embed_dim
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Positional embeddings
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Question encoder (self-attention stack)
        q_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu'
        )
        self.question_encoder = nn.TransformerEncoder(q_encoder_layer, num_layers=num_q_layers)
        
        # Cross-attention & feed-forward layers for context
        self.cross_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            layer = nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
                'attn_layernorm': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, ff_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, embed_dim)
                ),
                'ffn_layernorm': nn.LayerNorm(embed_dim)
            })
            self.cross_layers.append(layer)
        
        # Linear heads for start & end logits
        self.start_head = nn.Linear(embed_dim, 1)
        self.end_head   = nn.Linear(embed_dim, 1)

    def forward(self,
                context: torch.LongTensor,
                question: torch.LongTensor,
                attention_mask_context: torch.Tensor = None,
                attention_mask_question: torch.Tensor = None):
        """
        Args:
          context: (batch, c_len)
          question: (batch, q_len)
          attention_mask_context: (batch, c_len) with 1 for real tokens, 0 for pad
          attention_mask_question: (batch, q_len) with 1 for real tokens, 0 for pad
        Returns:
          start_logits, end_logits: (batch, c_len)
        """
        B, c_len = context.size()
        _, q_len = question.size()

        # 1) Embedding + positional
        ctx_pos = torch.arange(c_len, device=context.device).unsqueeze(0).expand(B, -1)
        qst_pos = torch.arange(q_len, device=question.device).unsqueeze(0).expand(B, -1)
        ctx_emb = self.token_emb(context) + self.pos_emb(ctx_pos)
        qst_emb = self.token_emb(question) + self.pos_emb(qst_pos)
        ctx_emb = self.dropout(ctx_emb)
        qst_emb = self.dropout(qst_emb)

        # 2) Encode question with TransformerEncoder
        # Transformer expects (seq_len, batch, embed_dim)
        q_enc = qst_emb.transpose(0, 1)
        q_key_pad = attention_mask_question == 0 if attention_mask_question is not None else None
        q_enc = self.question_encoder(q_enc, src_key_padding_mask=q_key_pad)
        # Back to (batch, q_len, embed_dim)
        q_enc = q_enc.transpose(0, 1)

        # 3) Cross-attention layers on context
        # Prepare context for attention: (seq_len, batch, embed_dim)
        ctx = ctx_emb.transpose(0, 1)
        k = v = q_enc.transpose(0, 1)

        
        for layer in self.cross_layers:
            attn_out, _ = layer['cross_attn'](
                query=ctx,
                key=k,
                value=v,
                key_padding_mask=q_key_pad
            )
            ctx = layer['attn_layernorm'](ctx + attn_out)
            ffn_out = layer['ffn'](ctx.transpose(0,1)).transpose(0,1)
            ctx = layer['ffn_layernorm'](ctx + ffn_out)
        # back to (batch, c_len, embed_dim)
        ctx_out = ctx.transpose(0, 1)

        # 4) Compute start & end logits
        start_logits = self.start_head(ctx_out).squeeze(-1)
        end_logits   = self.end_head(ctx_out).squeeze(-1)

        # 5) Mask out context pads so they can't be selected
        if attention_mask_context is not None:
            ctx_pad = attention_mask_context == 0
            start_logits = start_logits.masked_fill(ctx_pad, -1e9)
            end_logits   = end_logits.masked_fill(ctx_pad, -1e9)
        
        return start_logits, end_logits
