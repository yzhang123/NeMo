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
from torch.nn.utils.rnn import pad_sequence

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
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType, LengthsType
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
        return {
            'sent_ids': NeuralType(('B', 'T'), ChannelType()),
            'tag_ids': NeuralType(('B', 'T'), LabelsType()),
            'sent_lens': NeuralType(('B'), LengthsType()),
            'unnormalized_ids': NeuralType(('B', 'T'), ChannelType()),
            'char_lens_input': NeuralType(('B'), LengthsType()),
            'normalized_ids': NeuralType(('B', 'T'), LabelsType()),
            'char_lens_output': NeuralType(('B'), LengthsType()),
            'l_context_ids': NeuralType(('B'), ChannelType()),
            'r_context_ids': NeuralType(('B'), ChannelType()),
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

        self.tokenizer_sent = tokenizer_sent
        self.tokenizer_char = tokenizer_char
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

    def _collate_fn(self, batch):
        """collate batch of sent_ids, tag_ids, unnormalized_ids, normalized_ids l_context_id r_context_id
        """

        bs = len(batch)
        sent_lens = [0 for _ in range(bs)]
        char_lens_input = [0 for _ in range(bs)]
        char_lens_output = [0 for _ in range(bs)]
        max_length_sent = 0
        max_length_chars_input = 0
        max_length_chars_output = 0
        for i in range(bs):
            sent_ids, tag_ids, unnormalized_ids, normalized_ids, _, _ = batch[i]
            sent_lens[i] = len(sent_ids)
            char_lens_input[i] = len(unnormalized_ids)
            char_lens_output[i] = len(normalized_ids)
            if len(sent_ids) > max_length_sent:
                max_length_sent = len(sent_ids)
            if len(unnormalized_ids) > max_length_chars_input:
                max_length_chars_input = len(unnormalized_ids)
            if len(normalized_ids) > max_length_chars_output:
                max_length_chars_output = len(normalized_ids)

        sent_ids_padded = []
        tag_ids_padded = []
        unnormalized_ids_padded = []
        normalized_ids_padded = []
        l_context_ids = [0 for _ in range(bs)]
        r_context_ids = [0 for _ in range(bs)]
        for i in range(bs):
            sent_ids, tag_ids, unnormalized_ids, normalized_ids, l_context_id, r_context_id = batch[i]
            l_context_ids[i] = l_context_id
            r_context_ids[i] = r_context_id
            
            assert(len(sent_ids) == len(tag_ids))
            if len(sent_ids) < max_length_sent:
                pad_width = max_length_sent - len(sent_ids)
                sent_ids_padded.append(np.pad(sent_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_sent.pad_id))
                tag_ids_padded.append(np.pad(tag_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_sent.pad_id))
            else:
                sent_ids_padded.append(sent_ids)
                tag_ids_padded.append(tag_ids)

            if len(unnormalized_ids) < max_length_chars_input:
                pad_width = max_length_chars_input - len(unnormalized_ids)
                unnormalized_ids_padded.append(np.pad(unnormalized_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_char.pad_id))
            else:
                unnormalized_ids_padded.append(unnormalized_ids)
            
            if len(normalized_ids) < max_length_chars_output:
                pad_width = max_length_chars_output - len(normalized_ids)
                normalized_ids_padded.append(np.pad(normalized_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_char.pad_id))
            else:
                normalized_ids_padded.append(normalized_ids)
        
        # sent_ids_padded = pad_sequence(sent_ids, batch_first=False, padding_value=self.tokenizer_sent.pad_id)
        # tag_ids_padded = pad_sequence(tag_ids, batch_first=False, padding_value=-1)
        # unnormalized_ids_padded = pad_sequence(unnormalized_ids, batch_first=False, padding_value=self.tokenizer_char.pad_id)
        # normalized_ids_padded = pad_sequence(normalized_ids, batch_first=False, padding_value=self.tokenizer_char.pad_id)

        return (
            torch.LongTensor(sent_ids_padded),
            torch.LongTensor(tag_ids_padded),
            torch.LongTensor(sent_lens),
            torch.LongTensor(unnormalized_ids_padded),
            torch.LongTensor(char_lens_input),
            torch.LongTensor(normalized_ids_padded),
            torch.LongTensor(char_lens_output),
            torch.LongTensor(np.asarray(l_context_ids)),
            torch.LongTensor(np.asarray(r_context_ids)),
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
