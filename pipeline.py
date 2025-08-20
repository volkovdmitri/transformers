from text_utils import Tokenizer
from data_preparation import create_dataloader
import torch


if __name__ == "__main__":
    
    configs = {
        "file_name": "tolstoy.txt",
        "n_tokens": None,
        "batch_size": 8,
        "context_length": 4,
        "stride": 4,
        "embedding_size": 16
    }

    tokenizer = Tokenizer(configs["file_name"])
    tokens = tokenizer.encode(tokenizer.raw_text)

    dataloader = create_dataloader(
        tokens=tokens, 
        n_tokens=configs["n_tokens"], 
        batch_size=configs["batch_size"], 
        max_length=configs["context_length"], 
        stride=configs["stride"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )
    dataloader = iter(dataloader)

    embedding_layer = torch.nn.Embedding(tokenizer.vocab_size, configs["embedding_size"])
    pos_embedding_layer = torch.nn.Embedding(configs["context_length"], configs["embedding_size"])

    inputs, targets = next(dataloader)
    emb_inputs = embedding_layer(inputs) + pos_embedding_layer(torch.arange(configs["context_length"]))

    print(emb_inputs.shape)