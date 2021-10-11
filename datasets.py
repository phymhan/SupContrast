import torch
import numpy as np

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.class_size = len(np.unique(datasets[0].targets))

    def __getitem__(self, i):
        j = np.random.randint(0, self.class_size, size=(len(self.datasets)-1))

        return tuple(self.datasets[0][i], self.datasets[1][j])
        # return tuple(d[i] for d in self.datasets)

    def __len__(self):
        all_len = torch.Tensor([len(d) for d in self.datasets])
        # Check datasets are of equal length
        assert (all_len == all_len[0]).all()

        return min(len(d) for d in self.datasets)