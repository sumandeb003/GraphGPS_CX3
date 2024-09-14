import os
import os.path as osp
from typing import Callable, List, Optional

import torch

#TrustHubDFG: https://drive.google.com/file/d/1hGmKgRmw0_IlE6y-Z0cPI5PDz9AzDcLu/view?usp=sharing
#TrustHubAST: https://drive.google.com/file/d/1kav8vMtQO4ekdfy976swnurw_-52R89w/view?usp=sharing
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_google_url,
    extract_zip,
)
from torch_geometric.io import fs


class TrustHubGraphDataset(InMemoryDataset):
    id = '1mgBILYWXRyY9jAXeslmpKgwilP1etvSs'
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'

    def __init__(
        self,
        root: str, #supplied as `dataset_dir`: path where to store the cached dataset
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['TrustHubGraphDataset.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.url, self.root) #download zip file into the 'self.root' directory
        extract_zip(path, self.root) #the dataset is extracted into the 'self.root' directory
        os.rename(osp.join(self.root, 'trusthub_graphs_raw'), self.raw_dir)#'root' has 'raw','processed' directories. 
        os.unlink(path) #remove the downloaded zip file after extraction
        
    def process(self) -> None: #creates 3 splits - train.pt, val.pt, test.pt - and can be used with the `join_dataset_splits()` method in master_loader.py
        #convert to undirected graphs
        #remove previous split files if they are made using different circuits for val and test
        #create splits - train.pt, val.pt, test.pt
        #balancing TjIn and TjFree classes
        #node-level trojan detection
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graphs = torch.load(f)

            data_list: List[Data] = []
            for graph in graphs:
                x, edge_attr, edge_index, y = graph

                x = torch.from_numpy(x)
                edge_attr = torch.from_numpy(edge_attr)
                edge_index = torch.from_numpy(edge_index)
                y = torch.tensor([y]).float()

                #if edge_index.numel() == 0:
                #    continue  # Skipping for graphs with no bonds/edges.

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, path)
