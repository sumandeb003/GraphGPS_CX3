import os
import os.path as osp
import torch

from torch_geometric.data import Data, InMemoryDataset


class MyTrojans(InMemoryDataset):

    def __init__(self, split='train'):
        assert split in ['train', 'val', 'test']
        super().__init__(root)
        path = osp.join('./datasets/Trojans', f'{split}.pt')#path w.r.t main.py 
        self.data, self.slices = torch.load(path)
