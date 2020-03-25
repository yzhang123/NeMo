from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ....core.neural_types import *
from ...pytorch.nm import DataLayerNM

__all__ = ['MultiDataLayer']


class MultiDataLayer(DataLayerNM):
    def __init__(
        self,
        data_layers: List[DataLayerNM],
        batch_size: int,
        shuffle: bool = False,
        comb_mode: str = "cross_product",
        port_names: List[str] = None,
    ):
        """
        data_layers: list of DataLayerNM objects
        batch_size: (int) batchsize when the underlying dataset is loaded
        Comb_mode: str, defines how to combine the datasets, Options are [“cross_product”, “zip”]. “cross_product” to full requirement1.b, “zip” to fulfill requirement 1.a
        shuffle: (bool) whether underlying multi dataset should be shuffled in each epoch
        port_names: list(str). User can override all port names if specified 
        """
        super().__init__()
        self._data_layers = data_layers
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._comb_mode = comb_mode
        self._port_names = port_names
        self._dataset = MultiDataset([dl.dataset for dl in self._data_layers])

    @property
    def output_ports(self):
        """Return: dict
        Returns union of all individual data_layer output ports
        In case of name collision, resolve by renaming 
        """
        total_num_port = sum([len(dl.output_ports) for dl in self._data_layers])
        ports = dict()
        if self._port_names:
            assert (len(self._port_names) == total_num_port, "Number of ports is does not match.")
            i = 0
            for dl in self._data_layers:
                for _, port_type in dl.output_ports.items():
                    ports[self.port_names[i]] = port_type
                    i += 1
        else:
            for dl_idx, dl in enumerate(self._data_layers):
                for port_name, port_type in dl.output_ports.items():
                    if port_name in ports:
                        raise ValueError(f"name collision {port_name}, will rename")
                    ports[f"{port_name}_{dl_idx}"] = port_type
        return ports

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset], comb_mode: str = "cross_product"):
        """
        Datasets: list of torch.utils.data.Dataset objects.
        Comb_mode: str, defines how to combine the datasets, Options are [“cross_product”, “zip”]. 
        """
        self.datasets = datasets
        self.comb_mode = comb_mode

    def __getitem__(self, i):
        """
        Returns tuple (x1, x2, ...xn) where x1 \in D1, x2 \in D2, ...xn\ Dn
        """

        return [x for d in self.datasets for x in d[i % len(d)] ]

    def __len__(self):
        """
        Returns length of this dataset (int).
        In case of  comb_mode=”cross_product” this would be prod(len(d) for d in self.datasets). 
        In case of  comb_mode=”zip” this would be min(len(d) for d in self.datasets) given that all datasets have same length. 
        """

        # return max(len(d) for d in self.datasets)
        return np.prod([len(d) for d in self.datasets])
