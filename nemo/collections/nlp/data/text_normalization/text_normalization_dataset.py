# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import pickle
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from collections import namedtuple
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
# from nemo.collections.nlp.data.data_utils.data_preprocessing import (
#     fill_class_weights,
#     get_freq_weights,
#     get_label_stats,
#     get_stats,
# )
from nemo.collections.nlp.parts.utils_funcs import list2str
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['TextNormalizationDataset']


EOS_TYPE="EOS"
PUNCT_TYPE="PUNCT"
PLAIN_TYPE="PLAIN"
Instance = namedtuple('Instance', 'token_type un_normalized normalized')
tag_labels = {'TS': 0, 'TC': 1, 'NS': 2, 'NC': 3}

def load_files(file_paths: List[str]) -> List[Instance]:
    res = []
    for file_path in file_paths:
        res.extend(load_file(file_path=file_path))
    return res


def load_file(file_path: str) -> List[Instance]:
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts[0] == "<eos>":
                res.append(Instance(token_type=EOS_TYPE, un_normalized="", normalized=""))
            else:
                l_type, l_token, l_normalized = parts
                if l_type in [PUNCT_TYPE, PLAIN_TYPE]:
                    res.append(Instance(token_type=l_type, un_normalized=l_token, normalized=l_token))
                else:
                    res.append(Instance(token_type=l_type, un_normalized=l_token, normalized=l_normalized))
    return res

class TextNormalizationDataset(Dataset):

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """

        sent_ids, tag_ids, unnormalized_ids, normalized_ids, l_context_id, r_context_id
        return {
            'sent_ids': NeuralType(('B', 'T'), ChannelType()),
            'tag_ids': NeuralType(('B', 'T'), ChannelType()),
            'unnormalized_ids': NeuralType(('B', 'T'), MaskType()),
            'normalized_ids': NeuralType(('B', 'T'), MaskType()),
            'l_context_id': NeuralType(('B'), MaskType()),
            'r_context_id': NeuralType(('B'), LabelsType()),
        }

    def __init__(
        self,
        input_file: str,
        tokenizer_sent: TokenizerSpec,
        tokenizer_char: TokenizerSpec,
        num_samples: int = -1,
        use_cache: bool = True,
    ):

        data_dir = os.path.dirname(input_file)

        sent_vocab_size = getattr(tokenizer_sent, "vocab_size", 0)
        char_vocab_size = getattr(tokenizer_char, "vocab_size", 0)
        filename = os.path.basename(input_file)
        features_pkl = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                filename, tokenizer_sent.name, str(sent_vocab_size), tokenizer_char.name, str(char_vocab_size), str(num_samples)
            ),
        )

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        features = None
        if master_device and (not use_cache or not os.path.exists(features_pkl)):
            if num_samples == 0:
                raise ValueError("num_samples has to be positive", num_samples)

            instances = load_file(input_file)

            features = get_features(
                instances=instances,
                tokenizer_sent=tokenizer_sent,
                tokenizer_char=tokenizer_char,
            )

            pickle.dump(features, open(features_pkl, "wb"))
            logging.info(f'features saved to {features_pkl}')

        # wait until the master process writes to the processed data files
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if features is None:
            features = pickle.load(open(features_pkl, 'rb'))
            logging.info(f'features restored from {features_pkl}')

        self.features = features
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            np.array(self.features[idx][0]),
            np.array(self.features[idx][1]),
            np.array(self.features[idx][2]),
            np.array(self.features[idx][3]),
            np.array(self.features[idx][4]),
            np.array(self.features[idx][5]),
        )


def get_features(
    instances: List[Instance],
    tokenizer_sent: TokenizerSpec,
    tokenizer_char: TokenizerSpec
):

    def process_sentence(sentence: List[Instance]):
        # unnormalized_ids: char_ids <EOS>
        # normalized_ids <BOS>, word ids .., <EOS>
        #return list of unnormalized_ids, normalized_ids, l_context_id, r_context_id, sent_ids, tag_ids
        
        sent_ids = []
        tag_ids = []
        l_unnormalized_ids = []
        l_normalized_ids = []
        left_context_ids = []
        right_context_ids = []

        for instance in sentence:
            if instance.token_type in [PLAIN_TYPE, PUNCT_TYPE]:
                tokens = tokenizer_sent.text_to_ids(instance.un_normalized)
                sent_ids.extend(tokens)
                tag_ids.append(tag_labels['TS'])
                if len(tokens) > 1: 
                    tag_ids.extend(tag_labels['TC'] * (len(tokens) - 1))
            else:
                # semiotic class
                tokens = tokenizer_sent.text_to_ids(instance.un_normalized)
                left_context_ids.append( len(sent_ids) )
                sent_ids.extend(tokens)
                right_context_ids.append(len(sent_ids) - 1)
                tag_ids.append(tag_labels['NS'])
                if len(tokens) > 1: 
                    tag_ids.extend([tag_labels['NC']] * (len(tokens) - 1))
                l_unnormalized_ids.append(tokenizer_char.text_to_ids(instance.un_normalized) + [tokenizer_char.eos_id])
                l_normalized_ids.append([tokenizer_char.bos_id] + tokenizer_char.text_to_ids(instance.normalized) + [tokenizer_char.eos_id])

        features  =  [(sent_ids, tag_ids, l_unnormalized_ids[i], l_normalized_ids[i], left_context_ids[i], right_context_ids[i]) for i in range(len(l_unnormalized_ids))]
    
        return features

    features = []
    sentence = []
    for instance in instances:
        if instance.token_type == EOS_TYPE:
            features.extend(process_sentence(sentence))
            sentence = []
        else:
            sentence.append(instance)
    return features
