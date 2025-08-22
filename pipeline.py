import torch
import torch.nn as nn
import tiktoken

from data_preparation import create_dataloader
from transformer import Transformer, LayerNorm


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
    

def generate_text(model, tokens, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -context_length:]
        with torch.no_grad():
            logits = model.forward(tokens_cond)
    
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        token_next = torch.argmax(probas, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, token_next), dim=1)
    return tokens


def text_to_tokens(text, tokenizer):
    tokens = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    return tokens_tensor

def tokens_to_text(tokens, tokenizer):
    return tokenizer.decode(tokens.squeeze(0).tolist())

def calc_loss(input, target, model):
    out = model.forward(input)
    loss = torch.nn.functional.cross_entropy(out.flatten(0, 1), target.flatten())
    return loss

def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input, target) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss(input, target, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(input, target, model)
            loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): ",
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
        generate_print_sample(model, tokenizer, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()    
    train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_embedding_layer.weight.shape[0]
    tokens = text_to_tokens(start_context, tokenizer)
    with torch.no_grad():
        out = generate_text(model, tokens, max_new_tokens=50, context_length=context_size)
        out = tokens_to_text(out, tokenizer)
        print(out.replace("\n", " "))
    model.train()


if __name__ == "__main__":

    configs = {
        "vocab_size": 100256,
        "context_length": 128,
        "embedding_size": 768,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "n_layers": 12,
        "qkv_bias": False
    }
    
    with open("tolstoy.txt", 'r', encoding="utf-8") as f:
        txt = f.read()
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(txt)
    tokens = tokens[: int(len(tokens) / 100)]

    train_ratio = 0.9
    train_tokens = tokens[: int(train_ratio * len(tokens))]
    val_tokens = tokens[int(train_ratio * len(tokens)): ]

    train_loader = create_dataloader(
        tokens=train_tokens, 
        n_tokens=None, 
        batch_size=2, 
        max_length=configs["context_length"], 
        stride=configs["context_length"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )
    val_loader = create_dataloader(
        tokens=val_tokens, 
        n_tokens=None, 
        batch_size=2, 
        max_length=configs["context_length"], 
        stride=configs["context_length"], 
        shuffle=False, 
        drop_last=False, 
        num_workers=0
    )

model = GPTModel(configs)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

print("Number of steps: ", len(train_tokens)/configs["context_length"]/2)

train_losses, val_losses, tokens_seen = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=1,
    start_context="Привет, меня зовут Дима, и я очень хочу знать",
    tokenizer=tokenizer
)
