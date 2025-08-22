import torch
import torch.nn as nn
from attention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_in=configs["embedding_size"], 
            d_out=configs["embedding_size"], 
            num_heads=configs["num_heads"], 
            context_length=configs["context_length"], 
            dropout=configs["dropout_rate"], 
            qkv_bias=configs["qkv_bias"]
        )
        self.ff = FeedForward(configs)
        self.norm1 = LayerNorm(configs)
        self.norm2 = LayerNorm(configs) 
        self.dropout_shortcut = nn.Dropout(configs["dropout_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1.forward(x)
        x = self.mha.forward(x)
        x = self.dropout_shortcut(x) + shortcut

        shortcut = x
        x = self.norm2.forward(x)
        x = self.ff.forward(x)
        x = self.dropout_shortcut(x) + shortcut

        return x


class LayerNorm(nn.Module):
    def __init__(self, configs) :
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(configs["embedding_size"]))
        self.shift = nn.Parameter(torch.zeros(configs["embedding_size"]))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(configs["embedding_size"], 4 * configs["embedding_size"]),
            GELU(),
            nn.Linear(4 * configs["embedding_size"], configs["embedding_size"])
        )

    def forward(self, x):
        return self.layers(x)
    

class GELU(nn.Module):
    def __init__(self) :
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))