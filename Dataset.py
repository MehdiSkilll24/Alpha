from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer):
        super().__init__()

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        sample = self.dataset[idx]

        src_txt = sample["translation"]["en"]
        tgt_txt = sample["translation"]["fr"]

        src_ids = self.src_tokenizer.encode(src_txt)
        tgt_ids = self.tgt_tokenizer.encode(tgt_txt)

        return src_ids, tgt_ids