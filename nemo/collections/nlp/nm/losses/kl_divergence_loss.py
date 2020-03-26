# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
from torch import nn

from nemo.backends.pytorch import LossNM
from nemo.core import LogitsType, LossType, NeuralType, LabelsType, MaskType
from nemo.utils.decorators import add_port_docs
__all__ = ['KLDivergenceLossNM']


class KLDivergenceLossNM(LossNM):

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(['B'] + ['ANY' for _ in range(self._logits_ndim - 1)], LogitsType()),
            "labels": NeuralType(['B'] + ['ANY' for _ in range(self._logits_ndim - 1)], LogitsType()),
            "loss_mask": NeuralType(['B'] + ['ANY' for _ in range(self._logits_ndim - 2)], MaskType(), optional=True),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, logits_ndim, reduction='batchmean'):
        super().__init__()
        self._kldiv = nn.KLDivLoss(reduction=reduction)
        self._logsoftmax = nn.LogSoftmax(dim=-1)
        self._softmax = nn.Softmax(dim=-1)
        self._logits_ndim = logits_ndim

    def _loss_function(self, logits, labels, loss_mask=None):
        """
        Args:
            logits (float): output of the classifier
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        # """
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-2)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits = logits_flatten[loss_mask_flatten]
            labels = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return 0
        labels_probs = self._softmax(labels).detach()
        log_logits = self._logsoftmax(logits)
        loss = self._kldiv(input=log_logits, target=labels_probs)
        return loss
