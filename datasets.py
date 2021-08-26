import torch

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        all_len = torch.Tensor([len(d) for d in self.datasets])
        # Check datasets are of equal length
        assert (all_len == all_len[0]).all()

        return min(len(d) for d in self.datasets)