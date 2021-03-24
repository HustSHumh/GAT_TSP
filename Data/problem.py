from torch.utils.data import Dataset
import torch
import pickle
from Data.state import State

class TSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return State.initialize(*args, **kwargs)

class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    data = TSPDataset()
    print(1)

