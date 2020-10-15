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

from typing import Dict, Optional
import numpy as np
from torch import nn as nn
from nemo.core.classes import typecheck
from nemo.core.neural_types import LogitsType, NeuralType

__all__ = ['EncoderRNN', 'DynamicEncoder', 'AttnDecoderRNN', 'DecoderRNN', 'AttnDecoderMultiContextRNN']



import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, padding_idx=0, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size,embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

class DynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, padding_idx=0, n_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        sort_idx = torch.argsort(input_lens, descending=True)   
        unsort_idx = torch.argsort(input_lens, descending=False)   
        input_lens = input_lens[sort_idx]
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, encoder_hidden_size, decoder_hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attn = nn.Linear(self.encoder_hidden_size + self.decoder_hidden_size, encoder_hidden_size)
        self.v = nn.Parameter(torch.rand(encoder_hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score        
        if src_len is not None:
            mask_bool = torch.arange(max_len, device=attn_energies.device)[None, :] >= src_len[:, None]
            mask = torch.zeros_like(attn_energies, dtype=attn_energies.dtype)
            mask.masked_fill_(mask=mask_bool, value=-np.inf)

        attn_energies = attn_energies + mask
        
        return F.softmax(attn_energies, dim=-1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

        # import ipdb; ipdb.set_trace()
        # attn_weights = self.attn(torch.cat([hidden, encoder_outputs], 2)).squeeze(1)
        # return attn_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_embed_size, output_size, padding_idx=0, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        # Define parameters
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_embed_size = decoder_embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, decoder_embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', encoder_hidden_size, decoder_hidden_size)
        self.gru = nn.GRU(encoder_hidden_size, decoder_hidden_size, n_layers, dropout=dropout_p)
        self.attn_combine = nn.Linear(encoder_hidden_size + decoder_embed_size, encoder_hidden_size)
        self.out = nn.Linear(decoder_hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs, src_len=None):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden, encoder_outputs, src_len=src_len)
    
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        rnn_input = F.relu(rnn_input)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output), dim=-1)
        # Return final output, hidden state
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNNEmb(DecoderRNN):
    def __init__(self, hidden_size, output_size, padding_idx=0):
        super(DecoderRNNEmb, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden  = super.forward(input=output, hidden=hidden)
        return output, hidden




class AttnDecoderMultiContextRNN(AttnDecoderRNN):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_embed_size, output_size, padding_idx=0, n_layers=1, dropout_p=0.1):
        super(AttnDecoderMultiContextRNN, self).__init__(encoder_hidden_size, decoder_hidden_size, decoder_embed_size, output_size, padding_idx, n_layers, dropout_p)
        self.attn_combine = nn.Linear(encoder_hidden_size * 3 + decoder_embed_size, encoder_hidden_size)

    def forward(self, word_input, last_hidden, encoder_outputs, src_len, l_context, r_context):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden, encoder_outputs, src_len=src_len)# (B,1,V)
    
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        l_context = l_context.unsqueeze(0)
        r_context = r_context.unsqueeze(0)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, l_context, context, r_context), 2)
        
        rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        rnn_input = F.relu(rnn_input)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output), dim=-1)
        # Return final output, hidden state
        return output, hidden
