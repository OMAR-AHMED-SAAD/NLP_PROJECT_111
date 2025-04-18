import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAFAttention(nn.Module):
    """
    Implements the Bi-Directional Attention Flow (BiDAF) layer.
    Given encoded context and query (question) representations, it computes:
      - Context-to-query (c2q) attention
      - Query-to-context (q2c) attention
    It then forms an augmented representation G by combining the original
    context with the attended vectors.
    
    Args:
        hidden_dim (int): Dimension of the context and query encodings,
                          which should be 2*hidden_size from the BiLSTM encoder.
    """
    def __init__(self, hidden_dim):
        super(BiDAFAttention, self).__init__()
        # The similarity function takes as input [c, q, c ∘ q] for each (c,q) pair.
        # Its input dimension is 3 * hidden_dim.
        self.similarity_linear = nn.Linear(3 * hidden_dim, 1, bias=False)

    def forward(self, context, query):
        """
        Args:
            context: Tensor of shape (batch, c_len, hidden_dim)
            query:   Tensor of shape (batch, q_len, hidden_dim)
        Returns:
            G: Tensor of shape (batch, c_len, 4 * hidden_dim) — the combined representation.
        """
        batch_size, c_len, _ = context.size()
        q_len = query.size(1)
        
        # Expand dimensions for pairwise similarity.
        # context_expanded: (batch, c_len, q_len, hidden_dim)
        # query_expanded:   (batch, c_len, q_len, hidden_dim)
        context_expanded = context.unsqueeze(2).expand(-1, -1, q_len, -1)
        query_expanded = query.unsqueeze(1).expand(-1, c_len, -1, -1)
        
        # Elementwise multiplication: (batch, c_len, q_len, hidden_dim)
        elementwise_product = context_expanded * query_expanded
        
        # Concatenate features along the last dimension: dimension = 3 * hidden_dim.
        similarity_input = torch.cat([context_expanded, query_expanded, elementwise_product], dim=3)
        
        # Compute similarity matrix S: (batch, c_len, q_len)
        S = self.similarity_linear(similarity_input).squeeze(3)
        
        # --- Context-to-Query Attention ---
        # For each context word, attend over query words.
        a = F.softmax(S, dim=2)  # (batch, c_len, q_len)
        attended_query = torch.bmm(a, query)  # (batch, c_len, hidden_dim)
        
        # --- Query-to-Context Attention ---
        # For each query word, compute its max similarity over all context words,
        # then use a softmax over these values.
        b = F.softmax(torch.max(S, dim=2)[0], dim=1)  # (batch, c_len)
        b = b.unsqueeze(1)  # (batch, 1, c_len)
        attended_context = torch.bmm(b, context)  # (batch, 1, hidden_dim)
        attended_context = attended_context.repeat(1, c_len, 1)  # (batch, c_len, hidden_dim)
        
        # --- Combine ---
        # For each context word, combine:
        #   - the context encoding,
        #   - the attended query vector,
        #   - elementwise product of context and attended query,
        #   - elementwise product of context and attended context.
        G = torch.cat([context, attended_query, context * attended_query, context * attended_context], dim=2)
        return G

class BiDAFOutput(nn.Module):
    """
    Computes the start and end logits from the concatenated representations.
    It first concatenates the attention output (G) with the output of the modeling
    layer (M) to compute start logits. Then it passes the modeling output through
    another LSTM (producing M2), concatenates G and M2, and computes end logits.
    
    Args:
        hidden_size (int): The LSTM base hidden size.
                           (The contextual encoding from the BiLSTM is 2*hidden_size.)
    """
    def __init__(self, hidden_size, dropout=0.2):
        super(BiDAFOutput, self).__init__()
        # Note: G has shape (batch, c_len, 8*hidden_size)
        #       M has shape (batch, c_len, 2*hidden_size)
        # Therefore, the concatenated representation has dimension 10*hidden_size.
        self.p1_weight = nn.Linear(10 * hidden_size, 1)
        self.p2_weight = nn.Linear(10 * hidden_size, 1)
        # An additional LSTM for processing M further for predicting the end index.
        self.modelling_LSTM_end = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, G, M):
        """
        Args:
            G: Tensor of shape (batch, c_len, 8*hidden_size)
            M: Tensor of shape (batch, c_len, 2*hidden_size)
        Returns:
            start_logits: Tensor of shape (batch, c_len)
            end_logits:   Tensor of shape (batch, c_len)
        """
        # Start index prediction:
        GM = torch.cat([G, M], dim=2)  # (batch, c_len, 10*hidden_size)
        start_logits = self.p1_weight(self.dropout(GM)).squeeze(2)  # (batch, c_len)
        
        # End index prediction:
        # First, further process M with an LSTM to obtain M2.
        M2, _ = self.modelling_LSTM_end(M)  # (batch, c_len, 2*hidden_size)
        GM2 = torch.cat([G, M2], dim=2)      # (batch, c_len, 10*hidden_size)
        end_logits = self.p2_weight(self.dropout(GM2)).squeeze(2)  # (batch, c_len)
        return start_logits, end_logits

class BiDAF(nn.Module):
    """
    The full BiDAF model for machine comprehension.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the word embeddings.
        hidden_size (int): Base hidden size for the LSTM encoders.
                           The BiLSTM outputs have dimension 2*hidden_size.
        pretrained_embeddings (Tensor, optional): Pretrained embedding matrix.
        dropout (float): Dropout rate.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, pretrained_embeddings=None, dropout=0.2):
        super(BiDAF, self).__init__()
        # Embedding layer: converts word indices into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.dropout = nn.Dropout(dropout)
        # Contextual encoder: a BiLSTM that yields outputs of dimension 2*hidden_size.
        self.context_LSTM = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        # Attention flow layer: receives context and query encoded representations.
        # The input dimension for BiDAFAttention is 2*hidden_size.
        self.att_flow = BiDAFAttention(hidden_size * 2)
        # Modeling layer: a 2-layer BiLSTM over the attention output.
        # The attention output G has dimension 4 * (2*hidden_size) = 8*hidden_size.
        self.modeling_LSTM = nn.LSTM(8 * hidden_size, hidden_size, bidirectional=True, batch_first=True,
                                     num_layers=2, dropout=dropout)
        # Output layer: predicts the start and end positions.
        self.output_layer = BiDAFOutput(hidden_size, dropout)

    def forward(self, context: torch.Tensor, question: torch.Tensor) -> tuple:
        """
        Args:
            context: Tensor of shape (batch, c_len)
                          Contains word indices for the context passage.
            question: Tensor of shape (batch, q_len)
                           Contains word indices for the question.
        Returns:
            start_logits: Tensor of shape (batch, c_len)
            end_logits:   Tensor of shape (batch, c_len)
        """
        # (Optional) Create masks for context and question where indices are nonzero
        context_mask = (context != 0)
        question_mask = (question != 0)
        
        # --- Embedding Layer ---
        context_emb = self.dropout(self.embedding(context))   # (batch, c_len, embed_dim)
        question_emb = self.dropout(self.embedding(question))   # (batch, q_len, embed_dim)
        
        # --- Contextual Encoder (BiLSTM) ---
        context_encoded, _ = self.context_LSTM(context_emb)   # (batch, c_len, 2*hidden_size)
        question_encoded, _ = self.context_LSTM(question_emb)  # (batch, q_len, 2*hidden_size)
        
        # --- Attention Flow Layer ---
        # Computes an augmented representation G of shape (batch, c_len, 8*hidden_size)
        G = self.att_flow(context_encoded, question_encoded)
        
        # --- Modeling Layer ---
        # Further process G with a 2-layer BiLSTM, output M: (batch, c_len, 2*hidden_size)
        M, _ = self.modeling_LSTM(G)
        
        # --- Output Layer ---
        # Obtain start and end logits for each position in the context.
        start_logits, end_logits = self.output_layer(G, M)
        return start_logits, end_logits

