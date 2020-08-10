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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.backends.pytorch import MultiLayerPerceptron, TrainableNM
from nemo.collections.nlp.utils.transformer_utils import transformer_weights_init

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core import ChannelType, EmbeddedTextType, LogitsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDDecoderNM']



class LogitsQA(nn.Module):
    def __init__(self, num_classes, embedding_dim,device,
        num_layers=2,
        activation='relu',
        log_softmax=False,
        dropout=0.1,
        use_transformer_pretrained=True,
        ):
        """Get logits for elements by conditioning on utterance embedding.
        Args:
          num_classes (int): An int containing the number of classes for which logits are to be generated.
          embedding_dim (int): hidden size of the BERT
    
        Returns:
          A tensor of shape (batch_size, num_elements, num_classes) containing the logits.
        """
        super().__init__()
    
        self.mlp = MultiLayerPerceptron(embedding_dim, num_classes, device, num_layers, activation, log_softmax)
        self.dropout = nn.Dropout(dropout)
        # self.to(self._device) # sometimes this is necessary
    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        return logits

class SGDDecoderNM(TrainableNM):
    """
    Baseline model for schema guided dialogue state tracking with option to make schema embeddings learnable
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module output ports.
        encoded_utterance (float): [CLS] token hidden state from BERT encoding of the utterance
        token_embeddings (float): BERT encoding of utterance (all tokens)
        utterance_mask (bool): Mask which takes the value 0 for padded tokens and 1 otherwise
        cat_slot_values_mask (int): Masks out categorical slots values for slots not used in the service, takes values 0 and 1
        intent_status_mask (int): Masks out padded intents in the service, takes values 0 and 1
        service_ids (int): service ids
        """
        return {
            "token_embeddings": NeuralType(('B', 'T', 'C'), ChannelType()),
            "utterance_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
            logit_intent_status (float): output for intent status
            logit_req_slot_status (float): output for requested slots status
            logit_cat_slot_status (float): output for categorical slots status
            logit_cat_slot_value (float): output for categorical slots values
            logit_noncat_slot_status (float): Output of SGD model
            logit_noncat_slot_start (float): output for non categorical slots values start
            logit_noncat_slot_end (float): output for non categorical slots values end
        """
        return {
            "logit_intent_status": NeuralType(('B'), LogitsType()),
            "logit_req_slot_status": NeuralType(('B'), LogitsType()),
            "logit_cat_slot_status": NeuralType(('B'), LogitsType()),
            "logit_cat_slot_value_status": NeuralType(('B'), LogitsType()),
            "logit_noncat_slot_status": NeuralType(('B'), LogitsType()),
            "logit_noncat_slot_start": NeuralType(('B', 'T'), LogitsType()),
            "logit_noncat_slot_end": NeuralType(('B', 'T'), LogitsType()),
        }

    def __init__(self, embedding_dim, dropout):
        """Get logits for elements by conditioning on utterance embedding.

        Args:
            embedding_dim (int): hidden size of the BERT
            schema_emb_processor (obj): contains schema embeddings for services and config file
            head_transform (str): transformation to use for computing head
        """
        super().__init__()

        projection_module = LogitsQA

        self.intent_layer = projection_module(1, embedding_dim,self._device, dropout=dropout).to(self._device)
        self.requested_slots_layer = projection_module(1, embedding_dim,self._device, dropout=dropout).to(self._device)

        self.cat_slot_value_layer = projection_module(1, embedding_dim,self._device, dropout=dropout).to(self._device)

        # Slot status values: none, dontcare, active.
        self.cat_slot_status_layer = projection_module(3, embedding_dim,self._device, dropout=dropout).to(self._device)
        self.noncat_slot_layer = projection_module(3, embedding_dim,self._device, dropout=dropout).to(self._device)

        # dim 2 for non_categorical slot - to represent start and end position
        self.noncat_layer = MultiLayerPerceptron(embedding_dim, 2, self._device, 1, log_softmax=False).to(self._device)
        self.non_cat_dropout = nn.Dropout(dropout)

        self.apply(lambda module: transformer_weights_init(module, xavier=False))


    def forward(
        self,
        token_embeddings,
        utterance_mask
    ):
        batch_size, _, emb_dim = token_embeddings.size()
        logit_intent_status = self._get_intents(
            token_embeddings
        )

        logit_req_slot_status = self._get_requested_slots(
            token_embeddings
        )

        logit_cat_slot_status, logit_cat_slot_value_status = self._get_categorical_slot_goals(
            token_embeddings
        )

        (
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        ) = self._get_noncategorical_slot_goals(utterance_mask=utterance_mask, token_embeddings=token_embeddings)

        return (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        )

    def _get_intents(self, token_embeddings):
        """
        Args:
            encoded_utterance - representation of untterance
        """
        logits = self.intent_layer(
            token_embeddings
        )
        return logits

    def _get_requested_slots(self, token_embeddings):
        """Obtain logits for requested slots."""

        logits = self.requested_slots_layer(
            token_embeddings
        )
        return logits

    def _get_categorical_slot_goals(
        self,
        token_embeddings
    ):
        """
        Obtain logits for status and values for categorical slots
        Slot status values: none, dontcare, active
        """

        # Predict the status of all categorical slots.
        status_logits = self.cat_slot_status_layer(
            token_embeddings
        )


        value_status_logits = self.cat_slot_value_layer(
            token_embeddings
        )
        return status_logits, value_status_logits

    def _get_noncategorical_slot_goals(self, utterance_mask, token_embeddings):
        """
        Obtain logits for status and slot spans for non-categorical slots.
        Slot status values: none, dontcare, active
        """
        status_logits = self.noncat_slot_layer(token_embeddings)

        # Project the combined embeddings to obtain logits, Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.non_cat_dropout(token_embeddings)
        span_logits = self.noncat_layer(span_logits)

        # Mask out invalid logits for padded tokens.
        utterance_mask = utterance_mask.to(bool)  # Shape: (batch_size, max_num_tokens).
        repeated_utterance_mask = utterance_mask.unsqueeze(-1)
        negative_logits = (torch.finfo(span_logits.dtype).max * -0.7) * torch.ones(
            span_logits.size(), device=self._device, dtype=span_logits.dtype
        )

        span_logits = torch.where(repeated_utterance_mask, span_logits, negative_logits)

        # Shape of both tensors: (batch_size, max_num_slots, max_num_tokens).
        span_start_logits, span_end_logits = torch.unbind(span_logits, dim=-1)
        return status_logits, span_start_logits, span_end_logits

    def _get_negative_logits(self, logits):
        # returns tensor with negative logits that will be used to mask out unused values
        # for a particular service
        negative_logits = (torch.finfo(logits.dtype).max * -0.7) * torch.ones(
            logits.size(), device=self._device, dtype=logits.dtype
        )
        return negative_logits
