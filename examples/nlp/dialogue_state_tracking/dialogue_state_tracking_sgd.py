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

import argparse
import math
import os
import json
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.datasets.sgd_dataset.data_processor as data_processor
from nemo.collections.nlp.callbacks.sgd_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.sgd_dataset import schema
from nemo.collections.nlp.nm.trainables import SGDDecoderNM, SGDEncoderNM
from nemo.core import Backend, CheckpointCallback, EvaluatorCallback, NeuralModuleFactory, SimpleLossLoggerCallback, WandbCallback
from nemo.utils import logging
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
parser = argparse.ArgumentParser(description='Schema_guided_dst')

# BERT based utterance encoder related arguments
parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate for BERT representations.")
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-cased",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to model checkpoint")
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument("--vocab_file", default=None, help="Path to the vocab file.")
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)

# Hyperparameters and optimization related flags.
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    type=str,
    help="The folder containing the checkpoints for the model to continue training",
)
parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
parser.add_argument("--num_epochs", default=80, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--batches_per_step", default=1, type=int, help="Number of iterations per step.")
parser.add_argument("--optimizer_kind", default="adam_w", type=str)
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_policy", default="PolynomialDecayAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument(
    "--lr_warmup_proportion",
    default=0.1,
    type=float,
    help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10% of training.",
)
parser.add_argument("--grad_norm_clip", type=float, default=1, help="Gradient clipping")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--num_gpus", default=1, type=int)

# Input and output paths and other flags.
parser.add_argument(
    "--task_name",
    default="sgd_single_domain",
    type=str,
    choices=data_processor.FILE_RANGES.keys(),
    help="The name of the task to train.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory for the downloaded SGD data, which contains the dialogue files"
    " and schema files of all datasets (eg train, dev)",
)
parser.add_argument(
    "--work_dir",
    type=str,
    default="output/SGD",
    help="The output directory where the model checkpoints will be written.",
)
parser.add_argument(
    "--no_overwrite_schema_emb_files",
    action="store_false",
    help="Whether to generate a new file saving the dialogue examples.",
    dest="overwrite_schema_emb_files",
)
parser.add_argument(
    "--joint_acc_across_turn",
    action="store_true",
    help="Whether to compute joint accuracy across turn instead of across service. Should be set to True when conducting multiwoz style evaluation.",
)
parser.add_argument(
    "--no_fuzzy_match",
    action="store_true",
    help="Whether to use fuzzy string matching when comparing non-categorical slot values. Fuzz match should not be used when conducting multiwoz style evaluation.",
)
parser.add_argument(
    "--dialogues_example_dir",
    type=str,
    default="dialogues_example_dir",
    help="Directory where preprocessed SGD dialogues are stored.",
)
parser.add_argument(
    "--no_overwrite_dial_files",
    action="store_false",
    help="Whether to generate a new file saving the dialogue examples.",
    dest="overwrite_dial_files",
)
parser.add_argument("--no_shuffle", action="store_true", help="Whether to shuffle training data")
parser.add_argument("--no_time_to_log_dir", action="store_true", help="whether to add time to work_dir or not")
parser.add_argument(
    "--eval_dataset",
    type=str,
    default="dev_test",
    choices=["dev", "test", "dev_test"],
    help="Dataset splits for evaluation.",
)
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=-1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)

parser.add_argument(
    "--loss_log_freq", default=-1, type=int, help="Frequency of logging loss values, '-1' - at the end of the epoch",
)

parser.add_argument(
    "--loss_reduction",
    default='mean',
    type=str,
    help="specifies the reduction to apply to the final loss, choose 'mean' or 'sum'",
)

parser.add_argument(
    "--eval_epoch_freq", default=1, type=int, help="Frequency of evaluation",
)

parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
    help="Number of workers for data loading, -1 means set it automatically to the number of CPU cores",
)

parser.add_argument(
    "--enable_pin_memory", action="store_true", help="Enables the pin_memory feature of Pytroch's DataLoader",
)

parser.add_argument("--cat_value_thresh", default=0.0, type=float)
parser.add_argument("--non_cat_value_thresh", default=0.0, type=float)
parser.add_argument(
    "--state_tracker",
    type=str,
    default='baseline',
    choices=['baseline', 'nemotracker'],
    help="Specifies the state tracker model",
)
parser.add_argument(
    "--schema_emb_init",
    type=str,
    default='baseline',
    choices=['baseline', 'random', 'last_layer_average'],
    help="Specifies how schema embeddings are generated. Baseline uses ['CLS'] token",
)
parser.add_argument(
    "--train_schema_emb", action="store_true", help="Specifies whether schema embeddings are trainables.",
)
parser.add_argument(
    "--add_attention_head",
    action="store_true",
    help="Whether to use attention when computing projections. When False, uses linear projection.",
)
parser.add_argument(
    "--debug_mode", action="store_true", help="Enables debug mode with more info on data preprocessing and evaluation",
)

parser.add_argument(
    "--checkpoints_to_keep", default=1, type=int, help="The number of last checkpoints to keep",
)
parser.add_argument(
    "--num2str", action="store_true", help="make categorical values that are numbers in text to string, e.g. 2-> 2 two",
)
parser.add_argument(
    "--subsample", action="store_true", help="subsample negative slot statuses to be same as active/dont care ones",
)
parser.add_argument("--num_output_layers", default=1, type=int, help="Number of layers in the Classifier")
parser.add_argument("--exp_name", default="SGD_Baseline", type=str)
parser.add_argument("--project", default="SGD", type=str)

args = parser.parse_args()
logging.info(args)

if args.debug_mode:
    logging.setLevel("DEBUG")


schema_config = {
    "NUM_TASKS": 5,
}

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

nf = NeuralModuleFactory(
    backend=Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    checkpoint_dir=args.checkpoint_dir,
    files_to_copy=[__file__],
    add_time_to_log_dir=not args.no_time_to_log_dir,
)
if args.bert_config:
    bert_config_json = json.load(open(args.bert_config))
    bert_config_json['attention_probs_dropout_prob'] = args.dropout
    bert_config_json['hidden_dropout_prob'] = args.dropout
    args.bert_config = 'tmp_bert_config.json'
    with open(args.bert_config, 'w') as f:
        json.dump(bert_config_json, f)


pretrained_bert_model = nemo_nlp.nm.trainables.get_pretrained_lm_model(
    pretrained_model_name=args.pretrained_model_name,
    config=args.bert_config,
    vocab=args.vocab_file,
    checkpoint=args.bert_checkpoint,
)

schema_config["EMBEDDING_DIMENSION"] = pretrained_bert_model.hidden_size
schema_config["MAX_SEQ_LENGTH"] = args.max_seq_length

tokenizer = nemo_nlp.data.tokenizers.get_tokenizer(
    tokenizer_name=args.tokenizer,
    pretrained_model_name=args.pretrained_model_name,
    tokenizer_model=args.tokenizer_model,
    vocab_file=args.vocab_file,
    do_lower_case=args.do_lower_case,
)

hidden_size = pretrained_bert_model.hidden_size

all_schema_json_paths = []
for dataset_split in ['train', 'test', 'dev']:
    all_schema_json_paths.append(os.path.join(args.data_dir, dataset_split, "schema.json"))
schemas = schema.Schema(all_schema_json_paths)

dialogues_processor = data_processor.SGDDataProcessor(
    task_name=args.task_name,
    data_dir=args.data_dir,
    dialogues_example_dir=args.dialogues_example_dir,
    tokenizer=tokenizer,
    schemas=schemas,
    schema_config=schema_config,
    num2str=args.num2str,
    subsample=args.subsample,
    overwrite_dial_files=args.overwrite_dial_files,
)
# define model pipeline\
sgd_decoder = nemo_nlp.nm.trainables.SequenceClassifier(
    hidden_size=hidden_size,
    num_classes=3,
    dropout=args.dropout,
    num_layers=args.num_output_layers,
    log_softmax=False,
)
dst_loss = nemo_nlp.nm.losses.SGDDialogueStateLossNM(reduction=args.loss_reduction)


def create_pipeline(dataset_split='train'):
    datalayer = nemo_nlp.nm.data_layers.SGDDataLayer(
        dataset_split=dataset_split,
        dialogues_processor=dialogues_processor,
        batch_size=args.train_batch_size,
        shuffle=not args.no_shuffle if dataset_split == 'train' else False,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        pin_memory=args.enable_pin_memory,
    )
    data = datalayer()

    # Encode the utterances using BERT.
    hidden_states = pretrained_bert_model(
        input_ids=data.utterance_ids, attention_mask=data.utterance_mask, token_type_ids=data.utterance_segment,
    )
    logits = sgd_decoder(hidden_states=hidden_states)

    loss = dst_loss(
        logit_cat_slot_status=logits,
        categorical_slot_status=data.categorical_slot_status,
        task_mask=data.task_mask
    )
    if dataset_split == 'train':
        tensors = [loss]
    else:
        tensors = [
            data.example_id_num,
            data.service_id,
            data.is_real_example,
            data.start_char_idx,
            data.end_char_idx,
            logits,
            data.categorical_slot_status,
            loss
        ]

    steps_per_epoch = math.ceil(len(datalayer) / (args.train_batch_size * args.num_gpus * args.batches_per_step))
    return steps_per_epoch, tensors

steps_per_epoch, train_tensors = create_pipeline()
logging.info(f'Steps per epoch: {steps_per_epoch}')

# Create trainer and execute training action
train_callback = SimpleLossLoggerCallback(
    tensors=train_tensors,
    print_func=lambda x: logging.info("Loss: {:.8f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
    step_freq=args.loss_log_freq if args.loss_log_freq > 0 else steps_per_epoch,
)


def get_eval_callback(eval_dataset):
    _, eval_tensors = create_pipeline(dataset_split=eval_dataset)
    eval_callback = EvaluatorCallback(
        eval_tensors=eval_tensors,
        user_iter_callback=lambda x, y: eval_iter_callback(x, y, schemas, eval_dataset),
        user_epochs_done_callback=lambda x: eval_epochs_done_callback(
            x,
            args.task_name,
            eval_dataset,
            args.data_dir,
            nf.work_dir,
            args.state_tracker,
            args.debug_mode,
            dialogues_processor,
            schemas,
        ),
        tb_writer=nf.tb_writer,
        eval_step=args.eval_epoch_freq * steps_per_epoch,
        wandb_name=args.exp_name,
        wandb_project=args.project,
    )
    return eval_callback


if args.eval_dataset == 'dev_test':
    eval_callbacks = [get_eval_callback('dev'), get_eval_callback('test')]
else:
    eval_callbacks = [get_eval_callback(args.eval_dataset)]

ckpt_callback = CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq, checkpoints_to_keep=1
)

wand_callback = WandbCallback(
    train_tensors=train_tensors,
    wandb_name=args.exp_name,
    wandb_project=args.project,
    update_freq=args.loss_log_freq if args.loss_log_freq > 0 else steps_per_epoch,
    args=args,
)

lr_policy_fn = get_lr_policy(
    args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
)

nf.train(
    tensors_to_optimize=train_tensors,
    callbacks=[train_callback, wand_callback, ckpt_callback] + eval_callbacks,
    lr_policy=lr_policy_fn,
    optimizer=args.optimizer_kind,
    batches_per_step=args.batches_per_step,
    optimization_params={
        "num_epochs": args.num_epochs,
        "lr": args.learning_rate,
        "eps": 1e-6,
        "weight_decay": args.weight_decay,
        "grad_norm_clip": args.grad_norm_clip,
    },
)
