from datasets import load_dataset
from torch.utils.data import Dataset

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
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids