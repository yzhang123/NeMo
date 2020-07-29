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
            logit_cat_slot_status (float): Output of SGD model
            categorical_slot_status (int): The status of each categorical slot in the service
            task_mask
        """

        return {
            "logit_cat_slot_status": NeuralType(('B', 'D'), LogitsType()),
            "categorical_slot_status": NeuralType(('B'), LabelsType()),
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
        self._cross_entropy = torch.nn.CrossEntropyLoss(reduction=self.reduction)

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
        logit_cat_slot_status,
        categorical_slot_status,
        task_mask
    ):
        # # Intent loss
        # old_logit_intent_status = logit_intent_status
        # logit_intent_status, intent_status = self._helper(logit_intent_status, intent_status, task_mask[:, 0])
        # if len(intent_status) == 0:
        #     intent_loss = torch.clamp(torch.max(old_logit_intent_status.view(-1)), 0, 0)
        # else:
        #     intent_loss = self._cross_entropy_bin(logit_intent_status.squeeze(dim=-1), intent_status)

        # old_logit_req_slot_status = logit_req_slot_status
        # logit_req_slot_status, requested_slot_status = self._helper(logit_req_slot_status, requested_slot_status, task_mask[:, 1])
        # if len(requested_slot_status) == 0:
        #     requested_slot_loss = torch.clamp(torch.max(old_logit_req_slot_status.view(-1)), 0, 0)
        # else:            
        #     requested_slot_loss = self._cross_entropy_bin(
        #         logit_req_slot_status.squeeze(dim=-1), requested_slot_status
        #     )

        old_logit_cat_slot_status = logit_cat_slot_status
        logit_cat_slot_status, categorical_slot_status = self._helper(logit_cat_slot_status, categorical_slot_status, task_mask[:, 2])
        if len(categorical_slot_status) == 0:
            cat_slot_status_loss = torch.clamp(torch.max(old_logit_cat_slot_status.view(-1)), 0, 0)
        else:
            cat_slot_status_loss = self._cross_entropy(
                logit_cat_slot_status,
                categorical_slot_status,
            )
        # old_logit_cat_slot_value_status = logit_cat_slot_value_status
        # logit_cat_slot_value_status, categorical_slot_value_status = self._helper(logit_cat_slot_value_status, categorical_slot_value_status, task_mask[:, 3])
        # if len(categorical_slot_value_status) == 0:
        #     cat_slot_value_status_loss = torch.clamp(torch.max(old_logit_cat_slot_value_status.view(-1)), 0, 0)
        # else:
        #     cat_slot_value_status_loss = self._cross_entropy_bin(logit_cat_slot_value_status.squeeze(dim=-1), categorical_slot_value_status)

        # old_logit_noncat_slot_status = logit_noncat_slot_status
        # logit_noncat_slot_status, noncategorical_slot_status = self._helper(logit_noncat_slot_status, noncategorical_slot_status, task_mask[:, 4])
        # if len(noncategorical_slot_status) == 0:
        #     noncat_slot_status_loss = torch.clamp(torch.max(old_logit_noncat_slot_status.view(-1)), 0, 0)
        # else:
        #     noncat_slot_status_loss = self._cross_entropy(
        #         logit_noncat_slot_status,
        #         noncategorical_slot_status,
        #     )

        # _, max_num_tokens = logit_noncat_slot_start.size()
        # old_logit_noncat_slot_start = logit_noncat_slot_start
        # logit_noncat_slot_start, noncategorical_slot_value_start = self._helper(logit_noncat_slot_start, noncategorical_slot_value_start, task_mask[:, 5])
        # if len(noncategorical_slot_value_start) == 0:
        #     span_start_loss = torch.clamp(torch.max(old_logit_noncat_slot_start.view(-1)), 0, 0)
        # else:
        #     span_start_loss = self._cross_entropy(logit_noncat_slot_start, noncategorical_slot_value_start)

        
        # old_logit_noncat_slot_end = logit_noncat_slot_end
        # logit_noncat_slot_end, noncategorical_slot_value_end = self._helper(logit_noncat_slot_end, noncategorical_slot_value_end, task_mask[:, 5])
        # if len(noncategorical_slot_value_end) == 0:
        #     span_end_loss = torch.clamp(torch.max(old_logit_noncat_slot_end.view(-1)), 0, 0)
        # else:
        #     span_end_loss = self._cross_entropy(logit_noncat_slot_end, noncategorical_slot_value_end)

        # losses = {
        #     "intent_loss": intent_loss,
        #     "requested_slot_loss": requested_slot_loss,
        #     "cat_slot_status_loss": cat_slot_status_loss,
        #     "cat_slot_value_status_loss": cat_slot_value_status_loss,
        #     "noncat_slot_status_loss": noncat_slot_status_loss,
        #     "span_start_loss": span_start_loss,
        #     "span_end_loss": span_end_loss,
        # }

        total_loss = cat_slot_status_loss #sum(losses.values())
        if self.reduction == 'mean':
            total_loss = total_loss 
        else:
            batch_size = logit_intent_status.shape[0]
            total_loss = total_loss / batch_size
        return total_loss
