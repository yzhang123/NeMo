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

'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

import torch

from nemo import logging
from nemo.backends.pytorch import LossNM, BCEWithLogitsLossNM, CrossEntropyLossNM
from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import STATUS_ACTIVE
from nemo.core import ChannelType, LabelsType, LogitsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDDialogueStateLossNM']


class SGDDialogueStateLossNM(LossNM):
    """
    Neural module which implements loss for SGD model.
    """

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports.
            logit (float): Output of SGD model
            label (int): The status of each categorical slot in the service
            task_mask
        """

        return {
            "logit": NeuralType(('B', 'D'), LogitsType()),
            "label": NeuralType(('B'), LabelsType()),
            "task_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def output_ports(self):
        """
        Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): specifies the reduction to apply to the final loss, choose 'mean' or 'sum'
        """
        super().__init__()

        if reduction not in ['mean', 'sum']:
            logging.warning(f'{reduction} reduction is not supported. Setting reduction to "mean"')
            reduction = 'mean'

        self.reduction = reduction
        self._cross_entropy_bin = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)

    def _helper(self, logits, labels, loss_mask):
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        return logits_flatten, labels_flatten

    def _loss_function(
        self,
        logit,
        label,
        task_mask
    ):


        old_logit = logit
        logit, label = self._helper(logit, label, task_mask[:, 3])
        if len(label) == 0:
            loss = torch.clamp(torch.max(old_logit.view(-1)), 0, 0)
        else:
            loss = self._cross_entropy_bin(
                logit.squeeze(dim=-1),
                label,
            )
     
        total_loss = loss #sum(losses.values())
        if self.reduction == 'mean':
            total_loss = total_loss 
        else:
            batch_size = logit.shape[0]
            total_loss = total_loss / batch_size
        return total_loss
