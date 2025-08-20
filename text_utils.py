import tiktoken
import re


def tokenize(file_name, model="cl100k_base"):
    with open(file_name, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding(model)
    tokens = tokenizer.encode(raw_text)
    return tokens

if __name__ == "__main__":
    smpl = tokenize("tolstoy.txt")
    print(smpl[:50])
    print(len(set(smpl)))
