import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, tokens, max_length, stride, n_tokens=None):
        self.input_tokens = []
        self.target_tokens = []

        if n_tokens:
            tokens = tokens[: n_tokens + max_length]

        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i: i + max_length]
            target_chunk = tokens[i + 1: i + max_length + 1]
            self.input_tokens.append(torch.tensor(input_chunk))
            self.target_tokens.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_tokens)
    
    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]


def create_dataloader(tokens, n_tokens=None, batch_size=8, max_length=4, stride=4, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset(tokens, max_length, stride, n_tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader