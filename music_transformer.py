import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Token embedding (maps input tokens to vectors)
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional embedding (adds information about sequence order)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layer (multi-head attention + feed-forward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout
        )
        # Stack of encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output layer (projects back to vocabulary space)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Add token and positional embeddings, then apply dropout
        x = self.dropout(self.embedding(x) + self.pos_encoder(positions))

        # Create a triangular mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # Pass through transformer (transpose required by PyTorch)
        x = self.transformer(x.transpose(0, 1), mask=mask).transpose(0, 1)

        # Final linear layer for next-token prediction
        x = self.fc(x)
        return x