# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import torch
import random
from torch.utils import data as pt_data
import numpy as np
import nemo
from nemo.backends.pytorch import DataLayerNM
from nemo.collections.nlp.data.datasets.sgd_dataset.sgd_dataset import SGDDataset
from nemo.core.neural_types import ChannelType, LabelsType, LengthsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDDataLayer']


class SGDDataLayer(DataLayerNM):
    """
    Data layer for Schema Guided Dialogue State Tracking Dataset.
    Args:
        dataset_split (str): train/ dev/ test,
        dialogues_processor (obj):  containt dialogue data,
        dataset_type (Dataset): Dataset Type,
        shuffle (bool): enables shuffling, default=False
        num_workers (int): number of workers
        batch_size (int): batch size
        pin_memory (bool): enables copying Tensors into CUDA pinned memory before returning them
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        example_id_num (int): example ids
        service_id  (int): service ids
        is_real_example (bool): flag to determine is the example is valid
        utterance_ids (int): utterance ids
        utterance_segment (int): Denotes the identity of the sequence. Takes values 0 (system utterance) and 1 (user utterance)
        utterance_mask (int): Mask which takes the value 0 for padded tokens and 1 otherwise
        ...

        """

        return {
            "example_id_num": NeuralType(('B'), ChannelType()),
            "service_id": NeuralType(('B'), ChannelType()),
            "is_real_example": NeuralType(('B'), ChannelType()),
            "utterance_ids": NeuralType(('B', 'T'), ChannelType()),
            "utterance_segment": NeuralType(('B', 'T'), ChannelType()),
            "utterance_mask": NeuralType(('B', 'T'), ChannelType()),
            "intent_id": NeuralType(('B'), LabelsType()),
            "intent_status": NeuralType(('B'), LabelsType()),
            "requested_slot_id": NeuralType(('B'), LabelsType()),
            "requested_slot_status": NeuralType(('B'), LabelsType()),
            "categorical_slot_id": NeuralType(('B'), LabelsType()),
            "categorical_slot_value_id": NeuralType(('B'), LabelsType()),
            "categorical_slot_status": NeuralType(('B'), LabelsType()),
            "categorical_slot_value_status": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_status": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_id": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_value_start": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_value_end": NeuralType(('B'), LabelsType()),
            "start_char_idx": NeuralType(('B', 'T'), LabelsType()),
            "end_char_idx": NeuralType(('B', 'T'), LabelsType()),
            "task_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        dataset_split,
        dialogues_processor,
        tokenizer=None,
        mask_prob=0.0,
        dataset_type=SGDDataset,
        shuffle=False,
        batch_size=1,
        num_workers=-1,
        pin_memory=False,
    ):
        super().__init__()
        dataset_params = {
            'dataset_split': dataset_split,
            'dialogues_processor': dialogues_processor,
        }
        self._dataset = dataset_type(**dataset_params)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self.tokenizer = tokenizer
        if num_workers >= 0:
            self._num_workers = num_workers

        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset, batch_size=batch_size, collate_fn=self._collate_fn, shuffle=sampler is None, sampler=sampler
        )
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader

    def _collate_fn(self, x):

        num_components = len(x[0])
        components = [[] for _ in range(num_components)]
        batch_size = len(x)
        for i in range(batch_size):
            for j in range(num_components):
                if j == 5 and self.mask_prob > 0:
                    for k in range(sum(x[i][5])):
                        if random.random() < self.mask_prob:
                            x[i][j][k] = 0
                components[j].append(x[i][j])
        
        res = [torch.from_numpy(np.stack(x, axis=0)).to(self._device) for x in components]
        return res
