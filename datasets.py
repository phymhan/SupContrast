import torch
import numpy as np

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        j = np.random.randint(0, len(self.datasets[1]), size=1)
        j = j[0]

        return tuple([self.datasets[0][i], self.datasets[1][j]])
        # return tuple(d[i] for d in self.datasets)

    def __len__(self):
        all_len = torch.Tensor([len(d) for d in self.datasets])
        # Check datasets are of equal length
        assert (all_len == all_len[0]).all()

        return min(len(d) for d in self.datasets)
    

if __name__ == '__main__':
    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    d1 = torch.utils.data.TensorDataset(torch.from_numpy(np.arange(0,10)).float())
    d2 = torch.utils.data.TensorDataset(torch.from_numpy(np.arange(0,10)).float())

    # ds = ConcatDataset(d1, d2)
    # dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)#, worker_init_fn=worker_init_fn)
    dl = torch.utils.data.DataLoader(d1, batch_size=2, shuffle=True)#, worker_init_fn=worker_init_fn)
    dl_other = torch.utils.data.DataLoader(d2, batch_size=2, shuffle=True)#, worker_init_fn=worker_init_fn)
    from itertools import cycle
    dl_other_iter = iter(dl_other)
    for e in range(2):
        # np.random.seed(e)
        for i, x in enumerate(dl):
            print(x)
            try:
                y = next(dl_other_iter)
                print(y)
            except StopIteration:
                dl_other_iter = iter(dl_other)
                y = next(dl_other_iter)
                print(y)