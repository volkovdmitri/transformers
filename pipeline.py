import torch
import torch.nn as nn
import tiktoken

from text_utils import Tokenizer
from data_preparation import create_dataloader
from attention import MultiHeadAttention
    

class GPTModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(configs["vocab_size"], configs["embedding_size"])
        self.pos_embedding_layer = torch.nn.Embedding(configs["context_length"], configs["embedding_size"])
        self.dropout_embedding = nn.Dropout(configs["dropout_rate"])
        self.transformer_blocks = nn.Sequential(*[Transformer(configs) for _ in range(configs["n_layers"])])
        self.final_norm = LayerNorm(configs)
        self.out_head = nn.Linear(configs["embedding_size"], configs["vocab_size"], bias=False)

    def forward(self, x):
        b, num_tokens = x.shape
        x = self.embedding_layer(x) + self.pos_embedding_layer(torch.arange(num_tokens))
        x = self.dropout_embedding(x)
        x = self.transformer_blocks.forward(x)
        x = self.final_norm.forward(x)
        logits = self.out_head.forward(x)
        return logits


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


def generate_text_simple(model, tokens, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -context_length:]
        with torch.no_grad():
            logits = model.forward(tokens_cond)
    
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        token_next = torch.argmax(probas, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, token_next), dim=1)
    return tokens


if __name__ == "__main__":

    configs = {
        "vocab_size": 50257,
        "context_length": 4,
        "embedding_size": 768,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "n_layers": 12,
        "qkv_bias": False
    }
    
    txt = "Hello, I'm a "

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(txt)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    model = GPTModel(configs)
    
    
    res = generate_text_simple(
        model=model, 
        tokens=tokens_tensor, 
        max_new_tokens=6, 
        context_length=4
    )
    res = tokenizer.decode(res.squeeze(0).tolist())
    print(res)