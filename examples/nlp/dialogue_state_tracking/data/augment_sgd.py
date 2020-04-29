import os
import argparse
import sys
import json
from num2words import num2words 
from pprint import pprint
import re
import argparse
from collections import defaultdict
import copy
import random
from tqdm import tqdm
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--concat_orig_dialogue", action="store_true")
parser.add_argument("--data_dir", type=str, default="/home/yzhang/data/nlp/sgd/train/")
parser.add_argument("--dataset", choices=["sgd", "multiwoz"], type=str, required=True)
parser.add_argument("--num2string", action="store_true")
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--replace_turn_prob", type=float, default=1.0)
parser.add_argument("--replace_word_prob", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

in_file_path = args.data_dir
schema_path = os.path.join(in_file_path, 'schema.json')

dialogue_files = [os.path.join(in_file_path, f) for f in os.listdir(in_file_path) if os.path.isfile(os.path.join(in_file_path, f)) if "dialogue" in f]
dialogue_files.sort()
orig_dialog = []
for d_file in dialogue_files:
    orig_dialog.extend(json.load(open(d_file, 'r')))
print(f"len(orig_dialog) = {len(orig_dialog)}")
orig_schema = json.load(open(schema_path, 'r'))



def get_ontology(orig_dialog, orig_schema):
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
    return ontology



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
        for v, spans in list(vs.items()):
            if k == key and v == old_value:
                vs.pop(v)
                vs[new_value] = spans


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
        for frame in turn["frames"]:
            slot_to_spans = defaultdict(dict)
            for slot in frame["slots"]:
                k = slot["slot"]
                start_idx, end_idx = slot["start"], slot["exclusive_end"]
                slot_to_spans[k][turn["utterance"][start_idx: end_idx]]=[start_idx, end_idx]
            frame["slot_to_span"] = slot_to_spans

        if turn["speaker"] == "SYSTEM":
            for frame in turn["frames"]:
                new_slots = defaultdict(list)
                for action in frame["actions"]:
                    slot = action["slot"]
                    slot_values = action["values"]
                    for v in slot_values:
                        new_slots[slot] = get_affected_future_frames(dialogue, turn_id + 1, slot_name=slot, slot_value=v, service=frame["service"])
                        if v in turn["utterance"]:
                            if slot not in frame["slot_to_span"] or v not in frame["slot_to_span"][slot]:
                                if len(turn["utterance"].split(v)) == 2: 
                                    start_idx = turn["utterance"].index(v)
                                    end_idx = start_idx + len(v)
                                    frame["slot_to_span"][slot][v]=[start_idx, end_idx]
                frame["state_update"] = new_slots
        else:
            for frame in turn["frames"]:
                new_slots = defaultdict(list) # map from slot_value -> List[frames] in future
                for k, vs in frame["state"]["slot_values"].items():
                    for v_id, v in enumerate(vs):
                        if v in turn["utterance"]:
                            if k not in frame["slot_to_span"] or v not in frame["slot_to_span"][k]:
                                if len(turn["utterance"].split(v)) == 2: 
                                    start_idx = turn["utterance"].index(v)
                                    end_idx = start_idx + len(v)
                                    frame["slot_to_span"][k][v]=[start_idx, end_idx]
                        if k not in prev_state_slots_user or v not in prev_state_slots_user[k]:
                            new_slots[k] = get_affected_future_frames(dialogue, turn_id + 1, slot_name=k, slot_value=v, service=frame["service"])
                frame["state_update"] = new_slots

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
                if v in sentence:
                    start_idx = sentence.index(v)
                    end_idx = start_idx + len(v)
                    word_indices[start_idx:end_idx]=True
        if "actions" in frame:
            for action in frame["actions"]:
                k = action["slot"]
                for v in action["values"]:
                    if v in sentence:
                        start_idx = sentence.index(v)
                        end_idx = start_idx + len(v)
                        word_indices[start_idx:end_idx]=True

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
        for slot in frame["slots"]:
            if start_idx == slot["start"] and end_idx == slot["exclusive_end"]:
                res.append((turn_id, frame_id, slot["slot"]))
        # if "state" in frame: 
        #     for k, vs in frame["state"]["slot_values"].items():
        #         if k in frame["state_update"] and value in vs:
        #             res.append((turn_id, frame_id, k))
        # if "actions" in frame:
        #     # system doesnt need state_update
        #     for action in frame["actions"]:
        #         k = action["slot"]
        #         for v in action["values"]:
        #             if v == value:
        #                 res.append((turn_id, frame_id, k))
    return res


def get_new_value(dialogue, turn_id, value, start_idx, end_idx):
    """
    find value only in new state_update
    dialogue
    turn_id
    value
    start_idx
    end_idx

    return  List[(turn_id, frame_id, key)]
    """
    candidates = find_word_in_turn(dialogue, turn_id, value, start_idx, end_idx)
    possible_values = set()
    for _, frame_id, k in candidates:
        frame = dialogue["turns"][turn_id]["frames"][frame_id]
        service = frame["service"]
        if "possible_values" in ontology[(service, k)]:
            possible_values.update(ontology[(service, k)]["possible_values"])
    return random.choice(list(possible_values)) if possible_values else None

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
    old_value = sentence[start_idx:end_idx]
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
        update_values(dialogue, a_turn_id, a_frame_id, a_key, old_value, new_value)
    for a_turn_id, start_idx, end_idx in affected_spans:
        turn = dialogue["turns"][a_turn_id]
        assert(old_value==turn["utterance"][start_idx:end_idx])
        for a_frame_id in range(len(turn["frames"])):
            update_spans(dialogue, a_turn_id, a_frame_id, start_idx, end_idx, old_value, new_value)
        # update utterance
        turn["utterance"] = turn["utterance"][:start_idx] + new_value + turn["utterance"][end_idx:]
        # if 'Please confirm: I am booking a table ' in turn["utterance"]:
        #     import ipdb; ipdb.set_trace()


def digit2str(dialogue, turn_id, old_value, start_idx, end_idx):
    if old_value.isdigit():
        return num2words(old_value)
    return None

def validate(dialogue):
    # check slot spans match utterance and state value
    # check slot_values format
    # check slot spans appear in actions/state
    for turn_id, turn in enumerate(dialogue["turns"]):
        for frame_id, frame in enumerate(turn["frames"]):
            for slot in frame["slots"]:
                try:
                    st_idx, end_idx, key = slot["start"], slot["exclusive_end"], slot["slot"]
                    word = turn["utterance"][st_idx:end_idx]
                    assert(0 <= st_idx < end_idx <= len(turn["utterance"]))
                    if turn["speaker"] == "SYSTEM":
                        found_key = False
                        for action in frame["actions"]:
                            if action["slot"] == key:
                                if word in action["values"]:
                                    found_key=True
                        
                        assert(found_key)
                    else:
                        if key in frame["state"]["slot_values"]:
                            assert(word in frame["state"]["slot_values"][key])
                except:
                    raise ValueError(f"Turn {turn_id}, frame {frame_id}")



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


def process_dialogues(final_dialogues, dialogue_count, dialogues, replace_turn_prob, replace_word_prob, new_val_func):
    replace_success = 0
    replace_failed = 0
    for dialogue_id, dialogue in tqdm(enumerate(dialogues)):
        d_id, d_count = dialogue["dialogue_id"].split("_")
        d_id = int(d_id)
        dialogue["dialogue_id"]=f"{d_id}_{dialogue_count[d_id]:05d}"
        dialogue_count[d_id]+=1 
        for turn_id, turn in enumerate(dialogue["turns"]):
            if random.random() < replace_turn_prob:
                spans = get_sentence_components(turn=turn)
                for span in reversed(spans):
                    if random.random() < replace_word_prob:
                        old_value = dialogue["turns"][turn_id]["utterance"][span[0]:span[1]]
                        
                        new_value = new_val_func(dialogue, turn_id, old_value, span[0], span[1])
                        if new_value:
                            # print(old_value)
                            tmp_dialogue = copy.deepcopy(dialogue)
                            try:
                                replace(tmp_dialogue, turn_id, span[0], span[1], new_value)
                                validate(tmp_dialogue)
                                for k, v  in tmp_dialogue.items():
                                    dialogue[k] = v
                                replace_success += 1
                            except:
                                replace_failed += 1
                                pass
        for turn in dialogue["turns"]:
            for frame in turn["frames"]:
                if 'state_update' in frame:
                    frame.pop("state_update")
                if 'slot_to_span' in frame:
                    frame.pop("slot_to_span")
        final_dialogues[d_id].append(dialogue)
    print(f"Replacement success {replace_success}, failed {replace_failed}\n")

def change_numval_to_string(orig_dialog, orig_schema):
    # change schema categorical values
    iscategorical = defaultdict(bool)

    for schema in orig_schema:
        service_name = schema['service_name']
        for slot in schema['slots']:
            slot_name = slot['name']
            iscategorical[(service_name, slot_name)] = slot['is_categorical']
            for i, slot in enumerate(slot['possible_values']):
                if slot.isdigit():
                    slot['possible_values'][i] = num2words(slot)
    
    for dialogue in orig_dialog:
        for turn in dialogue["turns"]:
            for frame in turn["frames"]:
                service = frame["service"]
                if "state" in frame:
                    for k, vs in frame["state"]["slot_values"].items():
                        if iscategorical[(service, k)]:
                            for v_id, v in enumerate(vs):
                                if v.isdigit():
                                    vs[v_id] = num2words(v)
                if "actions" in frame:
                    for action in frame["actions"]:
                        k = action["slot"]
                        if iscategorical[(service, k)]:
                            for v_id, v in enumerate(action["values"]):
                                if v.isdigit():
                                    action["values"][v_id] = num2words(v)





if __name__=="__main__":


    dialogue_count = defaultdict(int)
    final_dialogues = defaultdict(list)
    if args.num2string:
        # change_numval_to_string(orig_dialog, orig_schema)
        if args.concat_orig_dialogue:
            process_dialogues(final_dialogues=final_dialogues, dialogue_count=dialogue_count, dialogues=orig_dialog, replace_turn_prob=1.0, replace_word_prob=1.0, new_val_func=digit2str)
        else:
            process_dialogues(final_dialogues=defaultdict(list), dialogue_count=defaultdict(int), dialogues=orig_dialog, replace_turn_prob=1.0, replace_word_prob=1.0, new_val_func=digit2str)
        

    ontology = get_ontology(orig_dialog=orig_dialog, orig_schema=orig_schema)

    for dialogue_id, dialogue in tqdm(enumerate(orig_dialog)):
        try:
            validate(dialogue)
        except:
            import ipdb; ipdb.set_trace()
        augment_dialog_by_auxiliary_entries(dialogue)
        validate(dialogue)


    
    for _ in range(args.repeat):
        dialogues = copy.deepcopy(orig_dialog)
        process_dialogues(final_dialogues=final_dialogues, dialogue_count=dialogue_count, dialogues=dialogues, replace_turn_prob=args.replace_turn_prob, replace_word_prob=args.replace_word_prob, new_val_func=get_new_value)
    
    if args.concat_orig_dialogue and not args.num2string:
        for dialogue_id, dialogue in tqdm(enumerate(orig_dialog)):
            d_id, d_count = dialogue["dialogue_id"].split("_")
            d_id = int(d_id)
            dialogue["dialogue_id"]=f"{d_id}_{dialogue_count[d_id]:05d}"
            dialogue_count[d_id]+=1 
            final_dialogues[d_id].append(dialogue)

    output_dir = f"{args.dataset}_augmented_repeat{args.repeat}_replace_turn_prob{args.replace_turn_prob}_replace_word_prob{args.replace_word_prob}_concatorig{args.concat_orig_dialogue}_num2string{args.num2string}"
    os.makedirs(output_dir, exist_ok=True)
    for dir_id, dialogues in final_dialogues.items():
        with open(os.path.join(output_dir, f"dialogues_{dir_id:03d}.json"), 'w') as outfile:
            json.dump(dialogues, outfile, indent=2)




