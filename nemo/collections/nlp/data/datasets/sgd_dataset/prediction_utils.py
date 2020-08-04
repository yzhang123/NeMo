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
Prediction and evaluation-related utility functions.
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/pred_utils.py
"""

import json
import os
from collections import OrderedDict, defaultdict

from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import (
    STATUS_ACTIVE,
    STATUS_DONTCARE,
    STATUS_OFF,
    STR_DONTCARE,
)
from nemo.collections.nlp.utils.functional_utils import _compute_softmax

REQ_SLOT_THRESHOLD = 0.5


# MIN_SLOT_RELATION specifes the minimum number of relations between two slots in the training dialogues to get considered for carry-over
MIN_SLOT_RELATION = 25

__all__ = ['get_predicted_dialog', 'write_predictions_to_file']


def set_cat_slot(predictions_status, cat_slots, cat_slot_values):
    """
    write predicted slot and values into out_dict 
    """
    out_dict = {}
    debug_cat_slots_dict = None
    for slot_idx, slot in enumerate(cat_slots):
        tmp = predictions_status[slot_idx]
        pred_value_idx = max(tmp, key=lambda k: tmp[k]['cat_slot_value_status'][0].item()) # predicted value
        # value_probs = _compute_softmax([v['cat_slot_value_status'][0].item() for k, v in tmp.items()])
        value_probs = [v['cat_slot_value_status'][0].item() for k, v in tmp.items()]
        value_prob = value_probs[pred_value_idx] 

        # if status is wrong, or wrong value when gt status is NOT == 0
        gt_status_id = predictions_status[slot_idx][0]["cat_slot_status_GT"].item()
        if (gt_status_id == STATUS_ACTIVE and tmp[pred_value_idx]['cat_slot_status_value_GT'] == 0):
            if debug_cat_slots_dict is None:
                debug_cat_slots_dict = defaultdict(tuple)
            
            value_idx_GT = max(tmp, key=lambda k: tmp[k]['cat_slot_status_value_GT'][0].item())
            value_prob_GT = value_probs[value_idx_GT]
            debug_cat_slots_dict[slot] = (
                gt_status_id,
                cat_slot_values[slot][value_idx_GT],
                value_prob_GT,
                cat_slot_values[slot][pred_value_idx],
                value_prob,
            )

        # if gt_status_id == STATUS_DONTCARE:
        #     out_dict[slot] = STR_DONTCARE
        if gt_status_id == STATUS_ACTIVE:
            out_dict[slot] = cat_slot_values[slot][pred_value_idx]
    return out_dict, debug_cat_slots_dict





def get_predicted_dialog(dialog, all_predictions, schemas, state_tracker):
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.

    # for debug
    total_mistakes = 0
    false_negative = 0
    false_positive = 0

    dialog_id = dialog["dialogue_id"]
    all_slot_values = defaultdict(dict)
    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else ""
            system_user_utterance = system_utterance + ' ' + user_utterance
            turn_id = "{:02d}".format(turn_idx)
            for frame in turn["frames"]:

                predictions = all_predictions[(dialog_id, turn_id, frame["service"])]
                slot_values = all_slot_values[frame["service"]]
                service_schema = schemas.get_service_schema(frame["service"])
                # Remove the slot spans and state if present.
                frame.pop("slots", None)
                frame.pop("state", None)

                # The baseline model doesn't predict slot spans. Only state predictions
                # are added.
                state = {}
                
                cat_out_dict, debug_cat_slots_dict = set_cat_slot(predictions_status=predictions[3], cat_slots=service_schema.categorical_slots, cat_slot_values=service_schema.categorical_slot_values)
                if debug_cat_slots_dict is not None:
                    print(debug_cat_slots_dict)
                #     for k, v in debug_cat_slots_dict.items():
                #         total_mistakes += 1
                #         if v[0] == 1 and v[1] == 0:
                #             false_negative += 1
                #         elif v[0] == 0 and v[1] == 1:
                #             false_positive += 1
                for k, v in cat_out_dict.items():
                    slot_values[k] = v

                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                frame["state"] = state
    # print(false_negative, total_mistakes)
    return dialog, total_mistakes, false_negative, false_positive






def write_predictions_to_file(
    predictions, input_json_files, output_dir, schemas, state_tracker, eval_debug, in_domain_services
):
    """Write the predicted dialogues as json files.

  Args:
    predictions: An iterator containing model predictions. This is the output of
      the predict method in the estimator.
    input_json_files: A list of json paths containing the dialogues to run
      inference on.
    schemas: Schemas to all services in the dst dataset (train, dev and test splits).
    output_dir: The directory where output json files will be created.
  """
    logging.info(f"Writing predictions to {output_dir} started.")

    # Index all predictions.
    all_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for idx, prediction in enumerate(predictions):
        if not prediction["is_real_example"]:
            continue
        eval_dataset, dialog_id, turn_id, service_name, model_task, slot_intent_id, value_id = prediction['example_id'].split('-')
        all_predictions[(dialog_id, turn_id, service_name)][int(model_task)][int(slot_intent_id)][int(value_id)] = prediction
    logging.info(f'Predictions for {idx} examples in {eval_dataset} dataset are getting processed.')

    # Read each input file and write its predictions.
    total_mistakes = 0
    false_negative = 0
    false_positive = 0
    for input_file_path in input_json_files:
        with open(input_file_path) as f:
            dialogs = json.load(f)
            logging.debug(f'{input_file_path} file is loaded')
            pred_dialogs = []
            for d in dialogs:
                pred_dialog, a, b, c = get_predicted_dialog(d, all_predictions, schemas, state_tracker)
                total_mistakes += a 
                false_negative += b
                false_positive += c
                pred_dialogs.append(pred_dialog)
            f.close()
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
            f.close()

    print("final false negative", false_negative, total_mistakes)
    print("final false positive", false_positive, total_mistakes)
