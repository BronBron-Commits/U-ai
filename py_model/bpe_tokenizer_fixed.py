from tokenizers import Tokenizer

class BPETokenizer:
    def __init__(self, tokenizer_json_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
