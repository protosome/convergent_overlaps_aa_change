import torch
import torch.nn as nn


# Identify available GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
torch.cuda.get_device_name(0)

# Tokenization and text vectorization
max_length = 315  # max len of the overlap
vocab_size = 27

# Define model parameters
embedding_dim = 156
num_heads = 13
ffn_dim = 192
num_blocks = 5
dropout_rate = 0.1
max_length = 315
num_classes = vocab_size  # for language modeling

# Transformer model
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embedding_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=max_length):
        super(SinusoidalPositionalEncoding, self).__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_blocks, num_heads, ffn_dim, max_length, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_length)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embedding_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embedding_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, vocab_size, dtype=torch.float32)

        # Weight Initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        x = self.embedding(x).to(device, dtype=torch.float32)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x