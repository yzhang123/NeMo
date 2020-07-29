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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst
"""

import json
import os

import torch

import nemo.collections.nlp.data.datasets.sgd_dataset.prediction_utils as pred_utils
from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.data_processor import SGDDataProcessor
from nemo.collections.nlp.data.datasets.sgd_dataset.evaluate import (
    ALL_SERVICES,
    PER_FRAME_OUTPUT_FILENAME,
    SEEN_SERVICES,
    UNSEEN_SERVICES,
    get_dataset_as_dict,
    get_in_domain_services,
    get_metrics,
)

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def get_str_example_id(eval_dataset, ids_to_service_names_dict, example_id_num):
    def format_turn_id(ex_id_num):
        dialog_id_1, dialog_id_2, turn_id, service_id, model_task_id, slot_intent_id, value_id = ex_id_num
        return "{}-{}_{:05d}-{:02d}-{}-{}-{}-{}".format(
            eval_dataset, dialog_id_1, dialog_id_2, turn_id, ids_to_service_names_dict[service_id], model_task_id, slot_intent_id, value_id
        )

    return list(map(format_turn_id, tensor2list(example_id_num)))


def eval_iter_callback(tensors, global_vars, schemas, eval_dataset):
    if 'predictions' not in global_vars:
        global_vars['predictions'] = []

    output = {}
    # keys of eval tensors
    # dict_keys(['example_id_num', 'service_id', 'is_real_example', 'start_char_idx', 'end_char_idx', 'logit_intent_status', 'logit_req_slot_status', 'logit_cat_slot_status', 'logit_cat_slot_value', 'logit_noncat_slot_status', 'logit_noncat_slot_start', 'logit_noncat_slot_end', 'intent_status_labels', 'requested_slot_status', 'categorical_slot_status', 'categorical_slot_values', 'noncategorical_slot_status'])
    for k, v in tensors.items():
        ind = k.find('~~~')
        if ind != -1:
            output[k[:ind]] = torch.cat(v)

    predictions = {}
    ids_to_service_names_dict = schemas._services_id_to_vocab
    predictions['example_id'] = get_str_example_id(eval_dataset, ids_to_service_names_dict, output['example_id_num'])

    predictions['service_id'] = output['service_id']
    predictions['is_real_example'] = output['is_real_example']

    # For categorical slots, the status of each slot and the predicted value are output.
    cat_slot_status_dist = torch.nn.Softmax(dim=-1)(output['logits'])

    predictions['cat_slot_status'] = torch.argmax(output['logits'], axis=-1)
    predictions['cat_slot_status_p'] = cat_slot_status_dist

    # added for debugging
    predictions['cat_slot_status_GT'] = output['categorical_slot_status']
    batch_size = cat_slot_status_dist.size()[0]
    global_vars['predictions'].extend(combine_predictions_in_example(predictions, batch_size))


def combine_predictions_in_example(predictions, batch_size):
    '''
    Combines predicted values to a single example. 
    Dict: sample idx-> keys-> values
    '''
    examples_preds = [{} for _ in range(batch_size)]
    for k, v in predictions.items():
        if k != 'example_id':
            v = torch.chunk(v, batch_size)

        for i in range(batch_size):
            if k == 'example_id':
                examples_preds[i][k] = v[i]
            else:
                examples_preds[i][k] = v[i].view(-1)
    return examples_preds


def eval_epochs_done_callback(
    global_vars,
    task_name,
    eval_dataset,
    data_dir,
    prediction_dir,
    state_tracker,
    eval_debug,
    dialogues_processor,
    schemas,
):
    # added for debugging
    in_domain_services = get_in_domain_services(
        os.path.join(data_dir, eval_dataset, "schema.json"), dialogues_processor.get_seen_services("train")
    )
    ##############
    # we'll write predictions to file in Dstc8/SGD format during evaluation callback
    prediction_dir = os.path.join(prediction_dir, 'predictions', 'pred_res_{}_{}'.format(eval_dataset, task_name))
    os.makedirs(prediction_dir, exist_ok=True)

    input_json_files = SGDDataProcessor.get_dialogue_files(data_dir, eval_dataset, task_name)
    pred_utils.write_predictions_to_file(
        global_vars['predictions'],
        input_json_files,
        prediction_dir,
        schemas=schemas,
        state_tracker=state_tracker,
        eval_debug=eval_debug,
        in_domain_services=in_domain_services,
    )
    metrics = evaluate(
        prediction_dir, data_dir, eval_dataset, in_domain_services
    )
    return metrics


def evaluate(prediction_dir, data_dir, eval_dataset, in_domain_services):

    with open(os.path.join(data_dir, eval_dataset, "schema.json")) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service
        f.close()

    dataset_ref = get_dataset_as_dict(os.path.join(data_dir, eval_dataset, "dialogues_*.json"))
    dataset_hyp = get_dataset_as_dict(os.path.join(prediction_dir, "*.json"))

    # has ALLSERVICE, SEEN_SERVICES, UNSEEN_SERVICES, SERVICE, DOMAIN
    all_metric_aggregate, _ = get_metrics(
        dataset_ref, dataset_hyp, eval_services, in_domain_services
    )
    if SEEN_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {SEEN_SERVICES}  : {sorted(all_metric_aggregate[SEEN_SERVICES].items())}')
    if UNSEEN_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {UNSEEN_SERVICES}: {sorted(all_metric_aggregate[UNSEEN_SERVICES].items())}')
    if ALL_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {ALL_SERVICES}   : {sorted(all_metric_aggregate[ALL_SERVICES].items())}')

    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(os.path.join(prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
        f.close()
    return all_metric_aggregate[ALL_SERVICES]
