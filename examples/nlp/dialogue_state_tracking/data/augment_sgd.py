import os
import argparse
import sys
import json
from num2words import num2words 
from pprint import pprint
import re
from collections import defaultdict
import copy
import numpy as np
in_file_path = '/home/yzhang/data/nlp/sgd/train/dialogues_001.json' #sys.argv[1]
schema_path = '/home/yzhang/data/nlp/sgd/train/schema.json' #sys.argv[2]

orig_dialog = json.load(open(in_file_path, 'r'))
orig_schema = json.load(open(schema_path, 'r'))

is_categorical=dict()
for schema in orig_schema:
    service_name = schema['service_name']
    for slot in schema['slots']:
        slot_name = slot['name']
        is_categorical[(service_name, slot_name)] = slot['is_categorical']
ontology = {}
#4.
def replace_utterance_and_span(turn, st_ch_idx, exclusive_end_ch_idx, new_word):
    """
    sentence (str):
    st_ch_idx (int): char start index in sentence
    exclusive_end_ch_idx (int): char end index in sentence
    status_updates (List[dict]) list of updated frames 
    """
    word_to_replace = turn["utterance"][st_ch_idx: exclusive_end_ch_idx]
    offset = len(new_word) - len(word_to_replace)
    # update spans
    # check if st_ch_idx <  start idx of any status update non-categorial, if so start idx+= offset len(new_word) - len(word_to_replace)
    # check if st_ch_idx < end idx of  any status update non-categorial, if so end idx += offset len(new_word) - len(word_to_replace)
    for frame in turn["frames"]:
        for slot in frame.get('slots', []):
            if st_ch_idx < slot['start']:
                slot['start'] += offset
            if st_ch_idx < slot['exclusive_end']:
                slot['exclusive_end'] += offset

#5.
def replace_user_state(old_value, new_value, frame, key):
    for k, v in frame["state"]["slot_values"].items():
        if k == key and v[0] == old_value:
            v[0] = new_value


def get_affected_future_frames(dialogue, from_turn_id, slot_name, slot_value, service):
    res = []
    for turn_id, turn in enumerate(dialogue["turns"][from_turn_id:], start=from_turn_id):
        if turn["speaker"] == "SYSTEM":
            continue
        for frame in turn["frames"]:
            if frame["service"] == service and frame["state"]["slot_values"].get(slot_name, [None]) == slot_value:
                res.append(frame)
                continue
    return res

def pick_replacement_word(ontology, service, slot, old_value):
    return "hiiii"

#1.
def augment_dialogue_by_new_slots(dialogue):
    prev_service = ""
    prev_state_slots = {} # key, value
    for turn_id, turn in enumerate(dialogue["turns"]):
        if turn["speaker"] == "SYSTEM":
            continue
        for frame in turn["frames"]:
            new_slots = defaultdict(list) # map from slot_value -> List[frames] in future
            for k, v in frame["state"]["slot_values"].items():
                if k not in prev_state_slots or prev_state_slots[k] != v:
                    new_slots[k] = get_affected_future_frames(dialogue, turn_id + 1, slot_name=k, slot_value=v, service=frame["service"])
            frame["state_update"] = new_slots
        
            


# 2. 
def get_sentence_components(turn):
    """
    return list of start and end indices of all terms(words can be multiple words )
    """
    sentence = turn["utterance"]
    word_indices = np.asarray([False for _ in range(len(sentence) + 1)])
    for frame in turn["frames"]:
        for k, v in frame["state"]["slot_values"].items():
            v = v[0]
            m = re.search(v, sentence)
            if m:
                word_indices[m.start():m.end()]= True
    for i in range(len(sentence)):
        if sentence[i].isalnum():
            word_indices[i] = True
    res= []
    idx = 0
    while (idx < len(word_indices)):
        if word_indices[idx]:
            start_idx = idx
            while word_indices[idx]:
                idx += 1
            end_idx = idx
            res.append((start_idx, end_idx))
        idx += 1
    return res
            


def find_word_in_new_values(word, frames):
    for frame in frames:
        for k, v in frame["state"]["slot_values"].items():
            if k in frame["state_update"] and word == v[0]:
                return frame, k
    return None


# 3. 
def replace(turn, start_idx, end_idx, ontology):
    """
    frame in turn has auxiliary new slot key
    ontology: (service, slot) = [list of all seen values/ possible values]
    """
    # does old_value segment start_idx : end_idx correspond to a new slot_value v of key k some frame f?
    # if no, do nothing
    # if yes 
        # determine new value
        # depends whether slot is categorical or not
        # replace_utterance_and_span(turn, start_idx, end_idx, new_value)
        # replace_user_state(old_value, new_value, f, k)
        # for each affected frame f' that is linked in frame f
            # replace_user_state(old_value, new_value, f', k)
    sentence = turn["utterance"]
    old_word = sentence[start_idx: end_idx]
    found = find_word_in_new_values(old_word, turn["frames"])
    if found:
        frame, key = found
        new_value = pick_replacement_word(ontology, frame["service"], key, old_word)
        if not is_categorical[(frame["service"], key)]:
            replace_utterance_and_span(turn, start_idx, end_idx, new_value)
        replace_user_state(old_word, new_value, frame, key)
        for affected_frame in frame["state_update"][key]:
            replace_user_state(old_word, new_value, affected_frame, key)
        turn["utterance"] =  turn["utterance"][:start_idx] + new_value + turn["utterance"][end_idx:]
            






def digit2str(x):
    x = x.split()
    for i in range(len(x)):
        if x[i].is_digit():
            x[i] = num2words(x[i])
    return " ".join(x)





dialogue = copy.deepcopy(orig_dialog[0])
augment_dialogue_by_new_slots(dialogue)
spans = get_sentence_components(dialogue["turns"][2])
replace(turn=dialogue['turns'][2], start_idx=spans[-1][0], end_idx=spans[-1][1], ontology=ontology)

for turn in dialogue["turns"]:
    for frame in turn["frames"]:
        if "state_update" in frame:
            frame.pop("state_update")
pprint(dialogue)
d_str_new = json.dumps(dialogue, sort_keys=True, indent=2)
d_str_old = json.dumps(orig_dialog[0], sort_keys=True, indent=2)
print(d_str_new == d_str_old)

# print(spans)
# for span in spans:
#     print(f"#{dialogue['turns'][2]['utterance'][span[0]: span[1]]}#")


