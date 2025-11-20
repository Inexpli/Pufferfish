import lmdb
import pickle
import torch
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            stat = txn.stat()
            self.length = stat["entries"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f"sample_{idx}".encode()

        with self.env.begin(write=False) as txn:
            data = txn.get(key)

        if data is None:
            raise IndexError(f"Brak rekordu: {idx}")

        sample = pickle.loads(data)
        
        x = torch.tensor(sample["x"], dtype=torch.float16).permute(2, 0, 1)
        y = torch.tensor(sample["y"], dtype=torch.long)

        return x, y
