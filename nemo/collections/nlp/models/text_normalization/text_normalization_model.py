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
from typing import Dict, List, Optional

import onnx
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_normalization import TextNormalizationDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
# from nemo.collections.nlp.models.nlp_model import NLPModel

from nemo.core.classes.modelPT import ModelPT
from nemo.collections.nlp.modules.common import AttnDecoderRNN, EncoderRNN, DynamicEncoder, DecoderRNN, AttnDecoderMultiContextRNN
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.export_utils import attach_onnx_to_onnx
from torch import nn

__all__ = ['TextNormalizationModel']


class TextNormalizationModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the BERTTextClassifier model."""

        # tokenizer needs to get initialized before the super.__init__()
        # as dataloaders and datasets need it to process the data
        self._tokenizer_sent = self._setup_tokenizer(cfg.tokenizer_tagger)
        self._tokenizer_char = self._setup_tokenizer(cfg.tokenizer_normalizer)
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher_forcing = True

        self.encoder_sent = DynamicEncoder(
            input_size=self._tokenizer_sent.vocab_size, 
            embed_size=cfg.context_encoder.embed_size, 
            hidden_size=cfg.context_encoder.hidden_size, 
            n_layers=cfg.context_encoder.n_layers, 
            padding_idx=self._tokenizer_sent.pad_id,
            dropout=cfg.context_encoder.dropout)


        self.tagger_output_emb = nn.Linear(cfg.context_encoder.hidden_size, cfg.tagger.hidden_size)
        self.tagger = DecoderRNN(hidden_size=cfg.tagger.hidden_size, output_size=cfg.tagger.hidden_size)
        self.tagger_output_layer = nn.Linear(cfg.tagger.hidden_size, cfg.tagger.num_classes)
        self.tagger_loss = CrossEntropyLoss(logits_ndim=3)

        self.seq2seq_encoder = DynamicEncoder(
            input_size=self._tokenizer_char.vocab_size, 
            embed_size=cfg.seqseq_encoder.embed_size, 
            hidden_size=cfg.seqseq_encoder.hidden_size, 
            n_layers=cfg.seqseq_encoder.n_layers, 
            padding_idx=self._tokenizer_char.pad_id,
            dropout=cfg.seqseq_encoder.dropout)

        self.seq2seq_decoder = AttnDecoderMultiContextRNN(
                encoder_hidden_size=cfg.seqseq_encoder.hidden_size, 
                decoder_hidden_size=cfg.seqseq_decoder.hidden_size, 
                decoder_embed_size=cfg.seqseq_decoder.embed_size, 
                output_size=self._tokenizer_sent.vocab_size, 
                n_layers=cfg.seqseq_decoder.n_layers, 
                padding_idx=self._tokenizer_sent.pad_id,
                dropout_p=cfg.seqseq_decoder.dropout,
)

        self.seq2seq_loss = CrossEntropyLoss(logits_ndim=3)


    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            vocab_file=self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file),
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        )
        return tokenizer

    # @typecheck()
    def forward(self, sent_ids, tag_ids, sent_lens, unnormalized_ids, char_lens_input, normalized_ids, char_lens_output, l_context_ids, r_context_ids):

    
        batch_size = len(sent_ids)

        context_outputs, context_hidden = self.encoder_sent(input_seqs=sent_ids, input_lens=sent_lens, hidden=None)
        max_seq_length = sent_ids.shape[1]
        tagger_hidden = self.tagger_output_emb(context_hidden[0] + context_hidden[1]).unsqueeze(0)
        all_tagger_outputs = torch.zeros((batch_size, max_seq_length, self._cfg.tagger.num_classes), device=self._device)
        for i in range(max_seq_length):
            context_in = self.tagger_output_emb(context_outputs[i]).unsqueeze(0)
            tagger_outputs, tagger_hidden = self.tagger(input=context_in, hidden=tagger_hidden)  # tagger outputs [B, H]
            logits = self.tagger_output_layer(tagger_outputs)
            all_tagger_outputs[:, i, :] = logits

        seq_enc_outputs, seq_enc_hidden = self.seq2seq_encoder(input_seqs=unnormalized_ids, input_lens=char_lens_input, hidden=None)
        max_target_length = normalized_ids.shape[1]


        all_decoder_outputs = torch.zeros((batch_size, max_target_length, self._tokenizer_sent.vocab_size), device=self._device)
        decoder_hidden = context_hidden.view(1, batch_size, -1)
        decoder_input = normalized_ids[:, 0]
        for i in range(max_target_length):
            # word_input torch.Size([5])    decoder_hidden=torch.Size([1, 5, 25])
            if self.teacher_forcing:
                decoder_input = normalized_ids[:, i]
            l_context = context_outputs[l_context_ids, torch.arange(end=batch_size, dtype=torch.long)]
            r_context = context_outputs[r_context_ids, torch.arange(end=batch_size, dtype=torch.long)]
            decoder_output, decoder_hidden = self.seq2seq_decoder(word_input=decoder_input, last_hidden=decoder_hidden, encoder_outputs=context_outputs, l_context=l_context, r_context=r_context, src_len=char_lens_input)
            all_decoder_outputs[:, 0, :] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
        return all_tagger_outputs, all_decoder_outputs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        sent_ids, tag_ids, sent_lens, unnormalized_ids, char_lens_input, normalized_ids, char_lens_output, l_context_ids, r_context_ids = batch
        bs, max_seq_length = sent_ids.shape
        _, max_target_length = normalized_ids.shape
        tagger_logits, decoder_logits = self.forward(sent_ids, tag_ids, sent_lens, unnormalized_ids, char_lens_input, normalized_ids, char_lens_output, l_context_ids, r_context_ids)
        tagger_loss_mask = torch.arange(max_seq_length).to(self._device).expand(bs, max_seq_length) < sent_lens.unsqueeze(1)  
        decoder_loss_mask = torch.arange(max_target_length).to(self._device).expand(bs, max_target_length) < sent_lens.unsqueeze(1)  
        tagger_loss = self.tagger_loss(logits=tagger_logits, labels=tag_ids, loss_mask=tagger_loss_mask)
        decoder_loss = self.seq2seq_loss(logits=decoder_logits, labels=normalized_ids, loss_mask=decoder_loss_mask)
        train_loss = tagger_loss + decoder_loss

        tensorboard_logs = {} #{'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        sent_ids, tag_ids, sent_lens, unnormalized_ids, char_lens_input, normalized_ids, char_lens_output, l_context_ids, r_context_ids = batch
        bs, max_seq_length = sent_ids.shape
        _, max_target_length = normalized_ids.shape
        tagger_logits, decoder_logits = self.forward(sent_ids, tag_ids, sent_lens, unnormalized_ids, char_lens_input, normalized_ids, char_lens_output, l_context_ids, r_context_ids)
        tagger_loss_mask = torch.arange(max_seq_length).to(self._device).expand(bs, max_seq_length) < sent_lens.unsqueeze(1)  
        decoder_loss_mask = torch.arange(max_target_length).to(self._device).expand(bs, max_target_length) < sent_lens.unsqueeze(1)  
        tagger_loss = self.tagger_loss(logits=tagger_logits, labels=tag_ids, loss_mask=tagger_loss_mask)
        decoder_loss = self.seq2seq_loss(logits=decoder_logits, labels=normalized_ids, loss_mask=decoder_loss_mask)
        train_loss = tagger_loss + decoder_loss

        tensorboard_logs = {} #{'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'val_loss': train_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return {}

        avg_loss = torch.stack([x[f'val_loss'] for x in outputs]).mean()
        tensorboard_logs = { }
        return {f'val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_idx):
    #     """
    #     Lightning calls this inside the test loop with the data from the test dataloader
    #     passed in as `batch`.
    #     """
    #     pass
    #     # return self.validation_step(batch, batch_idx)

    # def test_epoch_end(self, outputs):
    #     """
    #     Called at the end of test to aggregate outputs.
    #     :param outputs: list of individual outputs of each test step.
    #     """
    #     return {}
    #     # return self.validation_epoch_end(outputs)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.text_file:
            logging.info(
                f"Dataloader config or text_file for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.text_file:
            logging.info(
                f"Dataloader config or text_file for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    # def setup_test_data(self, test_data_config: Optional[DictConfig]):
    #     if not test_data_config or not test_data_config.text_file:
    #         logging.info(
    #             f"Dataloader config or text_file for the test is missing, so no data loader for test is created!"
    #         )
    #         self._test_dl = None
    #         return
    #     self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: Dict) -> 'torch.utils.data.DataLoader':
        input_file = cfg.text_file
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.text_file" and "train_ds.text_file" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        dataset = TextNormalizationDataset(
            input_file=input_file,
            tokenizer_sent=self._tokenizer_sent,
            tokenizer_char=self._tokenizer_char,
            num_samples=cfg.get("num_samples", -1),
            use_cache=self._cfg.dataset.use_cache,
        )

        dl =  torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def _prepare_for_export(self):
        return self.bert_model._prepare_for_export()

    # def export(
    #     self,
    #     output: str,
    #     input_example=None,
    #     output_example=None,
    #     verbose=False,
    #     export_params=True,
    #     do_constant_folding=True,
    #     keep_initializers_as_inputs=False,
    #     onnx_opset_version: int = 12,
    #     try_script: bool = False,
    #     set_eval: bool = True,
    #     check_trace: bool = True,
    #     use_dynamic_axes: bool = True,
    # ):
    #     if input_example is not None or output_example is not None:
    #         logging.warning(
    #             "Passed input and output examples will be ignored and recomputed since"
    #             " TextClassificationModel consists of two separate models with different"
    #             " inputs and outputs."
    #         )

    #     bert_model_onnx = self.bert_model.export(
    #         'bert_' + output,
    #         None,  # computed by input_example()
    #         None,
    #         verbose,
    #         export_params,
    #         do_constant_folding,
    #         keep_initializers_as_inputs,
    #         onnx_opset_version,
    #         try_script,
    #         set_eval,
    #         check_trace,
    #         use_dynamic_axes,
    #     )

    #     classifier_onnx = self.classifier.export(
    #         'classifier_' + output,
    #         None,  # computed by input_example()
    #         None,
    #         verbose,
    #         export_params,
    #         do_constant_folding,
    #         keep_initializers_as_inputs,
    #         onnx_opset_version,
    #         try_script,
    #         set_eval,
    #         check_trace,
    #         use_dynamic_axes,
    #     )

    #     output_model = attach_onnx_to_onnx(bert_model_onnx, classifier_onnx, "CL")
    #     onnx.save(output_model, output)
