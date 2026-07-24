from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        src_txt = sample["translation"]["en"]
        tgt_txt = sample["translation"]["fr"]

        src_ids = self.tokenizer.encode(src_txt)
        tgt_ids = self.tokenizer.encode(tgt_txt)

        return src_ids, tgt_ids