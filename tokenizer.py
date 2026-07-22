import re
from collections import Counter

class Tokenizer:
    def __init__(self, min_freq = 5):
        self.min_freq = min_freq

        self.word_to_idx = {}
        self.idx_to_word = {}

        self.special_tokens = {
            "<pad>" : 0, 
            "<unk>" : 1, 
            "<bos>" : 2, 
            "<eos>" : 3, 
        }

    def __len__(self):
        return len(self.word_to_idx)

    def tokenize(self, text):
        
        # raw text -> list of words

        text = text.lower()

        # split words and punctuations

        tokens = re.findall(r"\w+|[^\w\s]", text)

        return tokens
    
    def build_vocab(self, texts):

        # create word -> id mapping

        counter = Counter()

        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        self.word_to_idx = self.special_tokens.copy()

        for word, freq in sorted(counter.items()):
            if freq >= self.min_freq:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {
            idx: word for word, idx in self.word_to_idx.items()
        }

    def encode(self, text):

        # text -> ids 
        text = text.lower()

        tokens = self.tokenize(text)

        ids = [
            self.word_to_idx.get(
                token,
                self.word_to_idx["<unk>"]
            )
            for token in tokens
        ]

        ids = (
            [self.word_to_idx["<bos>"]]
            + ids
            +[self.word_to_idx["<eos>"]]
        )
        
        return ids
    
    def decode(self, ids):
        words = []

        for idx in ids:

            word = self.idx_to_word.get(
                idx,
                "<unk>"
            )

            if word not in [
                "<pad>",
                "<bos>",
                "<eos>",
                "<unk>"
            ]:
                words.append(word)

        return " ".join(words)