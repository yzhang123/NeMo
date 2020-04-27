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
in_file_path = '/home/yzhang/data/nlp/sgd/train/' # dialogues_001.json' #sys.argv[1]
schema_path = '/home/yzhang/data/nlp/sgd/train/schema.json' #sys.argv[2]

dialogue_files = [os.path.join(in_file_path, f) for f in os.listdir(in_file_path) if os.path.isfile(os.path.join(in_file_path, f)) if "dialogue" in f]
dialogue_files.sort()
orig_dialog = []
for d_file in dialogue_files:
    orig_dialog.extend(json.load(open(d_file, 'r')))
print(f"len(orig_dialog) = {len(orig_dialog)}")
orig_schema = json.load(open(schema_path, 'r'))

ontology=defaultdict(defaultdict)
for schema in orig_schema:
    service_name = schema['service_name']
    for slot in schema['slots']:
        slot_name = slot['name']
        ontology[(service_name, slot_name)]["is_categorical"] = slot['is_categorical']
        ontology[(service_name, slot_name)]["possible_values"] = set(slot['possible_values'])

for dialogue in orig_dialog:
    for turn in dialogue["turns"]:
        for frame in turn["frames"]:
            service_name = frame["service"]
            if "state" in frame:
                for k, vs in frame["state"]["slot_values"].items():
                    for v in vs:
                        ontology[(service_name, k)]["possible_values"].add(v)
            if "actions" in frame:
                for action in frame["actions"]:
                    k = action["slot"]
                    for v in action["values"]:
                        try:
                            ontology[(service_name, k)]["possible_values"].add(v)
                        except:
                            continue


def update_spans(dialogue, turn_id, frame_id, start_idx, end_idx, old_value, new_value):
    """
    update slot spans and slot_to_span
    """
    frame = dialogue["turns"][turn_id]["frames"][frame_id]
    offset = len(new_value) - len(old_value)

    for slot in frame['slots']:
        if start_idx < slot['start']:
            slot['start'] += offset
        if start_idx < slot['exclusive_end']:
            slot['exclusive_end'] += offset
    
    for k, vs in frame['slot_to_span'].items():
        for v, spans in vs.items():
            if start_idx < spans[0]:
                spans[0] += offset
            if start_idx < spans[1]:
                spans[1] += offset
        

def update_values(dialogue, turn_id, frame_id, key, old_value, new_value):
    """
    only update values: actions, state, slot_to_span
    """
    frame = dialogue["turns"][turn_id]["frames"][frame_id]
    if "actions" in frame:
        for action in frame["actions"]:
            if key == action["slot"] and old_value in action["values"]:
                action["values"].remove(old_value)
                action["values"].append(new_value)
    if "state" in frame:
        for k, vs in frame["state"]["slot_values"].items():
            for v_id, v in enumerate(vs):
                if k == key and v == old_value:
                    vs[v_id] = new_value

    for k, vs in frame["slot_to_span"].items():
        for v, spans in vs.items():
            if k == key and v == old_value:
                frame["slot_to_span"][k].pop(v)
                frame["slot_to_span"][k][new_value] = spans


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
                            res.append((turn_id, frame_id, slot_name))
                            continue
            else:
                if frame["service"] == service and slot_value in frame["state"]["slot_values"].get(slot_name, []) :
                    res.append((turn_id, frame_id, slot_name))
                    continue
    return res

def pick_replacement_word(ontology, service, slot, old_value):
    assert(isinstance(slot, str))
    assert(isinstance(old_value, str))
    assert(isinstance(service, str))
    return "moderate"

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
                for k, vs in frame["state"]["slot_values"].items():
                    for v_id, v in enumerate(vs):
                        m = re.search(v, turn["utterance"])
                        if m:
                            slot_to_spans[slot][v]=[m.start(), m.end()]
                        if k not in prev_state_slots_user or v not in prev_state_slots_user[k]:
                            new_slots[k] = get_affected_future_frames(dialogue, turn_id + 1, slot_name=k, slot_value=v, service=frame["service"])
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
    # find known keywords
    for frame in turn["frames"]:
        if "state" in frame:
            for k, v in frame["state"]["slot_values"].items():
                v = v[0]
                m = re.search(v, sentence)
                if m:
                    word_indices[m.start():m.end()]=True
        if "actions" in frame:
            for action in frame["actions"]:
                k = action["slot"]
                for v in action["values"]:
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
            
def find_word_in_turn(dialogue, turn_id, value, start_idx, end_idx):
    """
    find value only in new state_update
    dialogue
    turn_id
    value
    start_idx
    end_idx

    return  List[(turn_id, frame_id, key)]
    """
    assert(isinstance(value, str))
    frames = dialogue["turns"][turn_id]["frames"]
    res = []
    for frame_id, frame in enumerate(frames):
        if "state" in frame: 
            for k, vs in frame["state"]["slot_values"].items():
                if k in frame["state_update"] and value in vs:
                    res.append((turn_id, frame_id, k))
        if "actions" in frame:
            # system doesnt need state_update
            for action in frame["actions"]:
                k = action["slot"]
                for v in action["values"]:
                    if v == value:
                        res.append((turn_id, frame_id, k))
    return res

# 3. 
def replace(dialogue, turn_id, start_idx, end_idx, new_value):
    assert(isinstance(turn_id, int))
    assert(isinstance(start_idx, int))
    assert(isinstance(end_idx, int))
    """
    does old_value segment start_idx : end_idx correspond to a new slot_value v of key k some frame f?
    if no, do nothing
    if yes 
        determine new value
        depends whether slot is categorical or not
        replace_utterance_and_span(turn, start_idx, end_idx, new_value)
        replace_frame_state_values(old_value, new_value, f, k)
        for each affected frame f' that is linked in frame f
            replace_frame_state_values(old_value, new_value, f', k)
    """
    turn = dialogue["turns"][turn_id]
    sentence = turn["utterance"]
    old_value = sentence[start_idx: end_idx]
    affected_values = find_word_in_turn(dialogue=dialogue, turn_id=turn_id, value=old_value, start_idx=start_idx, end_idx=end_idx)
    affected_spans = [(turn_id, start_idx, end_idx)]
    for _, frame_id, key in affected_values.copy():
        frame = dialogue["turns"][turn_id]["frames"][frame_id]
        new_affected_values = frame["state_update"][key]
        affected_values += new_affected_values
        for a_turn_id, a_frame_id, a_key in new_affected_values:
            assert(key==a_key)
            spans=dialogue["turns"][a_turn_id]["frames"][a_frame_id]["slot_to_span"].get(a_key, {}).get(old_value, None)
            if spans:
                affected_spans += [(a_turn_id, spans[0], spans[1])]
    
    for a_turn_id, a_frame_id, a_key in affected_values:
        assert(a_key==key)
        update_values(dialogue, a_turn_id, a_frame_id, a_key, old_value, new_value)
    for a_turn_id, start_idx, end_idx in affected_spans:
        for a_frame_id in range(len(dialogue["turns"][a_turn_id]["frames"])):
            update_spans(dialogue, a_turn_id, a_frame_id, start_idx, end_idx, old_value, new_value)
        # update utterance
        dialogue["turns"][a_turn_id]["utterance"] = dialogue["turns"][a_turn_id]["utterance"][:start_idx] + new_value + dialogue["turns"][a_turn_id]["utterance"][end_idx:]
            
def digit2str(x):
    x = x.split()
    for i in range(len(x)):
        if x[i].is_digit():
            x[i] = num2words(x[i])
    return " ".join(x)

def validate(dialogue):
    # check slot spans match utterance and state value
    # check slot_values format
    # check slot spans appear in actions/state
    for turn in dialogue["turns"]:
        for frame in turn["frames"]:
            for slot in frame["slots"]:
                st_idx, end_idx, key = slot["start"], slot["exclusive_end"], slot["slot"]
                if turn["speaker"] == "SYSTEM":
                    found_key = False
                    for action in frame["actions"]:
                        if action["slot"] == key:
                            found_key = True
                            try:
                                assert(turn["utterance"][st_idx: end_idx] in action["values"])
                            except:
                                raise ValueError
                    assert(found_key)
                else:
                    assert(key in frame["state"]["slot_values"])
                    try:
                        assert(turn["utterance"][st_idx: end_idx] in frame["state"]["slot_values"][key])
                    except:
                        raise ValueError




def test_helper(dialogue, dialogue_id, turn_id, start_idx, end_idx, new_value):
    replace(dialogue, turn_id=turn_id, start_idx=start_idx, end_idx=end_idx, new_value=new_value)

    for turn in dialogue["turns"]:
        for frame in turn["frames"]:
            if "state_update" in frame:
                frame.pop("state_update")

# test_helper(orig_dialog, dialogue_id=1, turn_id=5, span_id=3, new_value="EXPENSIVEEE") # system cat, moderate
# test_helper(orig_dialog, dialogue_id=0, turn_id=2, span_id=-1, new_value="MIAMIIIIIIIIIIIII") # user non-cat, San Jose
# test_helper(orig_dialog, dialogue_id=0, turn_id=12, span_id=14, new_value="MODERATE") # replace value that does not match slotx

def test(dialogues, dialogue_id, turn_id, old_value, new_value):
    
    dialogue = copy.deepcopy(dialogues[dialogue_id])
    augment_dialog_by_auxiliary_entries(dialogue)
    # spans = get_sentence_components(dialogue["turns"][turn_id])
    m = re.search(old_value, dialogue["turns"][turn_id]["utterance"])
    
    test_helper(dialogue, dialogue_id, turn_id, start_idx=m.start(),end_idx=m.end(), new_value=new_value)
    
    pprint(dialogue)
    validate(dialogue)
    d_str_new = json.dumps(dialogue, sort_keys=True, indent=2)
    d_str_old = json.dumps(dialogues[dialogue_id], sort_keys=True, indent=2)
    print(d_str_new == d_str_old)




# test(dialogues=orig_dialog, dialogue_id=1, turn_id=5, old_value="moderate", new_value="EXPENSIVEEE") # system cat, moderate
# test(dialogues=orig_dialog, dialogue_id=0, turn_id=2, old_value="San Jose", new_value="MIAMIIIIIIIIIIIII") # user non-cat, San Jose
# test(dialogues=orig_dialog, dialogue_id=0, turn_id=12, old_value="economical", new_value="MODERATE") # replace value that does not match slotx
# test(dialogues=orig_dialog, dialogue_id=0, turn_id=3, old_value="Mexican", new_value="MEXICANNN") # sys non-cat 
# test(dialogues=orig_dialog, dialogue_id=0, turn_id=5, old_value="San Jose", new_value="MIAMIIIIIIIIIIIII") # user non-cat, San Jose

