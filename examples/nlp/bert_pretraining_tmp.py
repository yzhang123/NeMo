#!/usr/bin/env python3
# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os

import nemo
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
from nemo_nlp.utils.callbacks.bert_pretraining import \
    eval_iter_callback, eval_epochs_done_callback
from pytorch_transformers import BertConfig


parser = argparse.ArgumentParser(description='BERT pretraining')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--config_file", default=None, type=str, required=True, help="The BERT model config")

parser.add_argument("--max_predictions_per_seq", default=20, type=int, required=True, help="maximum number of masked tokens to predict")
parser.add_argument("--batch_size", default=64, required=True, type=int)
parser.add_argument("--lr", default=0.0000875, required=True, type=float)
parser.add_argument("--num_epochs", default=40, type=int)
parser.add_argument("--gradient_predivide", action="store_true", default=False, help="use gradient predivide")
parser.add_argument("--total_iterations_per_gpu", default=-1, type=int)
parser.add_argument("--num_gpus", default=16, required=True, type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)
parser.add_argument("--batches_per_step", default=1, type=int)
parser.add_argument("--batches_per_step_eval", default=1, type=int)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--lr_warmup_proportion", default=0.01, type=float)
parser.add_argument("--optimizer", default="adam_w", type=str)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--amp_opt_level",
                    default="O0",
                    type=str,
                    choices=["O0", "O1", "O2"])
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--data_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument("--data_dir_eval", default=None, type=str)
parser.add_argument("--load_dir", default=None, type=str)
parser.add_argument("--work_dir", default="outputs/bert_lm", type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=1000, type=int)
parser.add_argument("--eval_step_freq", default=100, type=int)
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased",
                    type=str, help="Name of the pre-trained model")
args = parser.parse_args()

# work_dir = f'{args.work_dir}/{args.upper()}'

if args.data_dir_eval is None:
    args.data_dir_eval = args.data_dir
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

config = BertConfig.from_json_file(args.config_file)
# Padding for divisibility by 8
# if config.vocab_size % 8 != 0:
#     config.vocab_size += 8 - (config.vocab_size % 8)


""" Use this if you're using a standard BERT model.
To see the list of pretrained models, call:
nemo_nlp.huggingface.BERT.list_pretrained_models()
"""
tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
# model = nemo_nlp.huggingface.BERT(
#     pretrained_model_name=args.pretrained_bert_model)
bert_model = nemo_nlp.huggingface.BERT(
**config.to_dict(),
factory=nf)



""" create necessary modules for the whole translation pipeline, namely
data layers, BERT encoder, and MLM and NSP loss functions
"""
mlm_classifier = nemo_nlp.TokenClassifier(config.hidden_size,
                                          num_classes=tokenizer.vocab_size,
                                          num_layers=1,
                                          log_softmax=True)
mlm_loss_fn = nemo_nlp.MaskedLanguageModelingLossNM()

nsp_classifier = nemo_nlp.SequenceClassifier(config.hidden_size,
                                             num_classes=2,
                                             num_layers=2,
                                             log_softmax=True)
nsp_loss_fn = nemo.backends.pytorch.common.CrossEntropyLoss()

bert_loss = nemo_nlp.LossAggregatorNM(num_inputs=2)

# tie weights of MLM softmax layer and embedding layer of the encoder
mlm_classifier.mlp.last_linear_layer.weight = \
    bert_model.bert.embeddings.word_embeddings.weight


def create_pipeline(data_file, max_predictions_per_seq, batch_size, mode, batches_per_step=1):
    data_layer = nemo_nlp.BertJoCPretrainingDataLayer(data_file,
                                                   max_predictions_per_seq,
                                                   batch_size=batch_size, mode=mode)
    steps_per_epoch = len(data_layer) // (batch_size * args.num_gpus * batches_per_step)

    input_ids, input_type_ids, input_mask, \
        output_ids, output_mask, nsp_labels = data_layer()
    hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)
    mlm_logits = mlm_classifier(hidden_states=hidden_states)
    mlm_loss = mlm_loss_fn(logits=mlm_logits,
                           output_ids=output_ids,
                           output_mask=output_mask)
    nsp_logits = nsp_classifier(hidden_states=hidden_states)
    nsp_loss = nsp_loss_fn(logits=nsp_logits, labels=nsp_labels)

    loss = bert_loss(loss_1=mlm_loss, loss_2=nsp_loss)
    return loss, [mlm_loss, nsp_loss], steps_per_epoch



train_loss, _, steps_per_epoch = create_pipeline(args.data_dir,
                                                 args.max_predictions_per_seq,
                                                 args.batch_size, mode="training", batches_per_step=args.batches_per_step)
eval_loss, eval_tensors, _ = create_pipeline(args.data_dir_eval,
                                            args.max_predictions_per_seq,
                                             args.eval_batch_size, mode="test", batches_per_step=args.batches_per_step_eval)

# callback which prints training loss and perplexity once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=args.eval_step_freq,
    tb_writer=nf.tb_writer)

ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                                             epoch_freq=args.save_epoch_freq,
                                             load_from_folder=args.load_dir,
                                             step_freq=args.save_step_freq)

# define learning rate decay policy
if args.total_iterations_per_gpu < 0:
    print("using num_epochs for lr schedule", args.num_epochs)
    lr_policy_fn = get_lr_policy(args.lr_policy,
                                total_steps=args.num_epochs * steps_per_epoch,
                                warmup_ratio=args.lr_warmup_proportion)
else:
    print("using total_iterations_per_gpu for lr schedule", args.total_iterations_per_gpu)
    lr_policy_fn = get_lr_policy(args.lr_policy,
                                total_steps=args.total_iterations_per_gpu,
                                warmup_ratio=args.lr_warmup_proportion)

# config_path = f'{nf.checkpoint_dir}/bert-config.json'
# if not os.path.exists(config_path):
#     bert_model.config.to_json_file(config_path)

# define and launch training algorithm (optimizer)
nf.train(tensors_to_optimize=[train_loss],
         lr_policy=lr_policy_fn,
         callbacks=[train_callback, eval_callback, ckpt_callback],
         optimizer=args.optimizer,
         batches_per_step=args.batches_per_step,
         gradient_predivide=args.gradient_predivide,
         optimization_params={"batch_size": args.batch_size,
                              "num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "betas": (args.beta1, args.beta2),
                              "weight_decay": args.weight_decay})
