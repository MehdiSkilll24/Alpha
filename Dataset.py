from datasets import load_dataset
from torch.utils.data import Dataset
from tokenizers import Tokenizer as HFTokenizer
from bpe import build_bpe_tokenizer

# Load raw WikiText-103
dataset = load_dataset("wikitext", "wikitext-103-v1")

combined_texts = dataset["train"]["text"]  # iterator of texts
tokenizer = build_bpe_tokenizer(combined_texts, vocab_size=40000)

class WikiTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024):
        super().__init__()
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample["text"]
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        return token_ids

# Usage
train_dataset = WikiTextDataset(dataset["train"], tokenizer, max_length=1024)
val_dataset = WikiTextDataset(dataset["validation"], tokenizer, max_length=1024)
test_dataset = WikiTextDataset(dataset["test"], tokenizer, max_length=1024)