import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, train=True):
        self.train = train

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        rng = torch.Generator()
        rng.manual_seed(idx)

        input = torch.randn(3, 224, 224, generator=rng)

        meta = {}
        meta['target'] = (torch.rand([], generator=rng) > 0.5).float()

        return input, meta
        

    

