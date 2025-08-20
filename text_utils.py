import tiktoken
import re

class Tokenizer:
    def __init__(self, file_name, model_name="cl100k_base"):
        self.file_name = file_name
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.vocab_size = len(set(self.tokenizer._mergeable_ranks))
        with open(file_name, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

    def encode(self, txt):
        tokens = self.tokenizer.encode(txt)
        return tokens

    def decode(self, tokens):
        text = self.tokenizer.decode(tokens)
        return text
    

if __name__ == "__main__":
    smpl = Tokenizer("tolstoy.txt")
    print(smpl.vocab_size)
    tokens = smpl.encode(smpl.raw_text)
    print(tokens[:200])
    print(smpl.decode(tokens[:200]))
