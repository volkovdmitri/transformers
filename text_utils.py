import tiktoken
import re

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, my name is Dima"
tokens = tokenizer.encode(text)
print(tokens)