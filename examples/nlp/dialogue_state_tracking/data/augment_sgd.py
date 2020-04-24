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
def replace_frame_values(turn, service, key, old_value, new_value):
    assert(isinstance(old_value, str))
    assert(isinstance(new_value, str))
    """
    sentence (str):
    st_ch_idx (int): char start index in sentence
    exclusive_end_ch_idx (int): char end index in sentence
    status_updates (List[dict]) list of updated frames 
    """
    offset = len(new_value) - len(old_value)
    # update spans
    # check if st_ch_idx <  start idx of any status update non-categorial, if so start idx+= offset len(new_value) - len(word_to_replace)
    # check if st_ch_idx < end idx of  any status update non-categorial, if so end idx += offset len(new_value) - len(word_to_replace)
    for frame in turn["frames"]:
        if frame["service"] == service:
            st_ch_idx, exclusive_end_ch_idx = frame['slot_to_span'].get(key, {old_value: (-1, -1)}).get(old_value, (-1, -1))
            print(turn["utterance"])
            if st_ch_idx != -1:
                for slot in frame.get('slots', []):
                    if st_ch_idx < slot['start']:
                        slot['start'] += offset
                    if st_ch_idx < slot['exclusive_end']:
                        slot['exclusive_end'] += offset
        
            is_sys_utt = "actions" in frame
            if is_sys_utt:
                for action in frame["actions"]:
                    if key == action["slot"] and old_value in action["values"]:
                        action["values"].remove(old_value)
                        action["values"].append(new_value)
            else:
                for k, v in frame["state"]["slot_values"].items():
                    if k == key and v[0] == old_value:
                        v[0] = new_value

            # update slot_to_span if old_val in utterance
            if st_ch_idx != -1:
                for slot_name, val_spans in frame["slot_to_span"].items():
                    for slot_val, spans in val_spans.items():
                        if spans[0] == -1:
                            continue
                        if slot_name == key and slot_val == old_value:
                            frame["slot_to_span"][slot_name].pop(slot_val)
                            frame["slot_to_span"][slot_name][new_value] = spans
                        if st_ch_idx < spans[0]:
                            spans[0] += offset
                        if st_ch_idx < spans[1]:
                            spans[1] += offset 
                # update utterance
                turn["utterance"] = turn["utterance"][:st_ch_idx] + new_value +  turn["utterance"][exclusive_end_ch_idx:]


def get_affected_future_frames(dialogue, from_turn_id, slot_name, slot_value, service):
    assert(isinstance(from_turn_id, int))
    assert(isinstance(slot_name, str))
    assert(isinstance(slot_value, str))
    assert(isinstance(service, str))
    res = []
    for turn_id, turn in enumerate(dialogue["turns"][from_turn_id:], start=from_turn_id):
        for frame_id, frame in enumerate(turn["frames"]):
            if turn["speaker"] == "SYSTEM":
                # import ipdb; ipdb.set_trace()
                if frame["service"] == service:
                    for action in frame["actions"]:
                        if action["slot"] == slot_name and slot_value in action["values"]:
                            res.append((turn_id, frame_id, service, slot_name))
                            continue
            else:
                if frame["service"] == service and frame["state"]["slot_values"].get(slot_name, [None])[0] == slot_value:
                    res.append((turn_id, frame_id, service, slot_name))
                    continue
    return res

def pick_replacement_word(ontology, service, slot, old_value):
    assert(isinstance(slot, str))
    assert(isinstance(old_value, str))
    assert(isinstance(service, str))
    return "MIAMIIIIIIIIII"

#1.
def augment_dialog_by_auxiliary_entries(dialogue):
    prev_service_user = ""
    prev_state_slots_user = {} # key, value
    for turn_id, turn in enumerate(dialogue["turns"]):
        if turn["speaker"] == "SYSTEM":
            for frame in turn["frames"]:
                new_slots = defaultdict(list)
                slot_to_spans = defaultdict(dict)
                for action in frame["actions"]:
                    slot = action["slot"]
                    slot_values = action["values"]
                    for v in slot_values:
                        new_slots[slot] = get_affected_future_frames(dialogue, turn_id + 1, slot_name=slot, slot_value=v, service=frame["service"])
                        m = re.search(v, turn["utterance"])
                        if m:
                            slot_to_spans[slot][v]=[m.start(), m.end()]
                frame["state_update"] = new_slots
                frame["slot_to_span"] = slot_to_spans
        else:
            for frame in turn["frames"]:
                new_slots = defaultdict(list) # map from slot_value -> List[frames] in future
                slot_to_spans = defaultdict(dict)
                for k, v in frame["state"]["slot_values"].items():
                    m = re.search(v[0], turn["utterance"])
                    if m:
                        slot_to_spans[slot][v[0]]=[m.start(), m.end()]
                    if k not in prev_state_slots_user or prev_state_slots_user[k] != v:
                        new_slots[k] = get_affected_future_frames(dialogue, turn_id + 1, slot_name=k, slot_value=v[0], service=frame["service"])
                frame["state_update"] = new_slots
                frame["slot_to_span"] = slot_to_spans

            if len(turn["frames"]) == 1:
                use_frame = turn["frames"][0]
            else:
                use_frame = [frame for frame in turn["frames"] if frame["service"] != prev_service_user][0]
            prev_service_user = use_frame["service"]
            prev_state_slots_user = use_frame["state"]["slot_values"]

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
                word_indices[m.start():m.end()]=True
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
            


def find_new_word_in_turn(word, turn):
    assert(isinstance(word, str))
    frames = turn["frames"]
    for frame_id, frame in enumerate(frames):
        for k, v in frame["state"]["slot_values"].items():
            if k in frame["state_update"] and word == v[0]:
                return frame_id, frame["service"], k, v[0]
    return None


# 3. 
def replace(dialogue, turn_id, start_idx, end_idx, ontology):
    assert(isinstance(turn_id, int))
    assert(isinstance(start_idx, int))
    assert(isinstance(end_idx, int))
    """
    frame in turn has auxiliary new slot key
    ontology: (service, slot) = [list of all seen values/ possible values]
    TODO: add for each frame mapping from ALL slot value -> span in utterance

    """
    # does old_value segment start_idx : end_idx correspond to a new slot_value v of key k some frame f?
    # if no, do nothing
    # if yes 
        # determine new value
        # depends whether slot is categorical or not
        # replace_utterance_and_span(turn, start_idx, end_idx, new_value)
        # replace_frame_state_values(old_value, new_value, f, k)
        # for each affected frame f' that is linked in frame f
            # replace_frame_state_values(old_value, new_value, f', k)
    turn = dialogue["turns"][turn_id]
    sentence = turn["utterance"]
    old_value = sentence[start_idx: end_idx]
    found = find_new_word_in_turn(old_value, turn)
    if found:
        frame_id, service, key, _ = found
        frame = turn["frames"][frame_id]
        new_value = pick_replacement_word(ontology, frame["service"], key, old_value)

        affected_frames = [(turn_id, frame_id, service, key)] + frame["state_update"][key]
        for affected in affected_frames:
            affected_turn_id, affected_frame_id, affected_service, affected_slot = affected
            
            replace_frame_values(dialogue["turns"][affected_turn_id], affected_service, affected_slot,  old_value, new_value)
            






def digit2str(x):
    x = x.split()
    for i in range(len(x)):
        if x[i].is_digit():
            x[i] = num2words(x[i])
    return " ".join(x)





dialogue = copy.deepcopy(orig_dialog[0])
augment_dialog_by_auxiliary_entries(dialogue)
spans = get_sentence_components(dialogue["turns"][2])
replace(dialogue, 2, start_idx=spans[-1][0], end_idx=spans[-1][1], ontology=ontology)

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


