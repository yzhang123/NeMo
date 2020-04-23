import os
import argparse
import sys
import json
from num2words import num2words 
from pprint import pprint
in_file_path = '/home/yzhang/data/nlp/sgd/train/dialogues_001.json' #sys.argv[1]
schema_path = '/home/yzhang/data/nlp/sgd/train/schema.json' #sys.argv[2]

orig_dialog = json.load(open(in_file_path, 'r'))
orig_schema = json.load(open(schema_path, 'r'))


#4.
def replace_utterance_and_span(turn, st_ch_idx, exclusive_end_ch_idx, new_word):
    """
    sentence (str):
    st_ch_idx (int): char start index in sentence
    exclusive_end_ch_idx (int): char end index in sentence
    status_updates (List[dict]) list of updated frames 
    """
    word_to_replace = sentence[st_ch_idx: exclusive_end_ch_idx]
    offset = len(new_word) - len(word_to_replace)
    # update spans
    # check if st_ch_idx <  start idx of any status update non-categorial, if so start idx+= offset len(new_word) - len(word_to_replace)
    # check if st_ch_idx < end idx of  any status update non-categorial, if so end idx += offset len(new_word) - len(word_to_replace)
    for frame_update in status_updates:
        for slot in frame_update.get('slots', []):
            if st_ch_idx < slot['start']:
                slot['start'] += offset
            if st_ch_idx < slot['exclusive_end']:
                slot['exclusive_end'] += offset


        # replace slot values in actions if possible
        for action in frame_update.get('actions', []):
            # if action in ['INFORM', 'OFFER', 'REQUEST']
            if action['values'] and action['values'][0] == word_to_replace:
                action['values'][0] = new_word

            
        # replace slot values in state if possible
        for slot_name in frame_update.get('state', {}).get('slot_values', {}):
            if frame_update['state']['slot_values'][slot_name][0] ==  word_to_replace:
                frame_update['state']['slot_values'][slot_name][0] = new_word
    # replace sentence
    new_sentence = sentence[:st_ch_idx] + new_word + sentence[exclusive_end_ch_idx:]
    return new_sentence

#5.
def replace_user_state(old_value, new_value, frame, key):

# 1.
def augment_dialogue_by_new_slots(dialogue):

# 2. 
def get_sentence_components(turn):
    """
    return list of start and end indices of all terms(words can be multiple words )
    """
# 3. 
def replace(turn, start_idx, end_idx)
    # does old_value segment start_idx : end_idx correspond to a new slot_value v of key k some frame f?
    # if no, do nothing
    # if yes 
        # determine new value
        # depends whether slot is categorical or not
        # replace_utterance_and_span(turn, start_idx, end_idx, new_value)
        # replace_user_state(old_value, new_value, f, k)
        # for each affected frame f' that is linked in frame f
            # replace_user_state(old_value, new_value, f', k)
        # 


def get_status_diff(prev_state, new_state):
    state_update = new_state.copy()
    deleted_slots = []
    for slot_name, slot_value in state_update.items():
        if prev_state.get(slot_name, [None]) == slot_value:
            deleted_slots.append(slot_name)
    for slot_name in deleted_slots:
        del state_update[slot_name]
    return state_update, deleted_slots

def get_status_updates_from_dialogue(dialog):
    """
    dialogue: dict
    """
    prev_state = {}
    prev_service = ""
    status_updates_of_all_turns = []

    for turn in dialog["turns"]:
        status_updates_of_all_turns.append([])
        speaker = turn["speaker"]
        utterance = turn["utterance"]
        if speaker == "SYSTEM":
            status_updates_of_all_turns[-1].append( (speaker, utterance, turn["frames"][0], []) )
        else:
            new_state = None
            set_new_state = False
            for frame in turn["frames"]:
                if frame["service"] != prev_service:
                    new_state = frame["state"]["slot_values"]
                    set_new_state = True
                    status_update, deleted_slots = new_state, prev_state
                else:
                    status_update, deleted_slots = get_status_diff(prev_state, frame["state"]["slot_values"])
                    new_frame = frame.copy()
                    new_frame["state"]["slot_values"] = status_update
                    status_updates_of_all_turns[-1].append( (speaker, utterance, new_frame, deleted_slots ))
            if not set_new_state:
                new_state = turn["frames"]["state"]["slot_values"]
            prev_state = new_state

    return status_updates_of_all_turns

            

updates = get_status_updates_from_dialogue(orig_dialog[0])
pprint(updates)
                 


        
def process_dialogue(dialogue):
    status_updates_of_all_turns = get_status_updates_from_dialogue(dialogue)

    # add replacement

    new_dialogue = compose_dialogues_from_all_status_updates(status_updates_of_all_turns)
    return new_dialogue






def digit2str(x):
    x = x.split()
    for i in range(len(x)):
        if x[i].is_digit():
            x[i] = num2words(x[i])
    return " ".join(x)

is_categorical=dict()
for schema in orig_schema:
    service_name = schema['service_name']
    for slot in schema['slots']:
        slot_name = slot['name']
        is_categorical[(service_name, slot_name)] = slot['is_categorical']


# for dialog in orig_dialog:
#     for turn in dialog['turns']:
#         orig_utterance = turn['utterance']
#         new_frames = [frame.copy for frame in turn['frames']]

#         slot_spans = []
#         for frame_id, frame in enumerate(turn['frames']):
#             service_name = frame['service']
#             if frame.get('slots', None):
#                 # do not standardize non-categorical 
#                 slot_spans.append((frame['slots']['start'], frame['slots']['exclusive_end']))

#             if frame.get('state', None):
#                 for slot_name, slot_val in frame['state']['slot_values'].items():
#                     if is_categorical[(service_name, slot_name)]:
#                         new_frames[frame_id]['state']['slot_values'][0] = digit2str(slot_val[0])
                    
#             if frame.get('actions', None):
#                 for action_id, action in enumerate(frame['actions']):
#                     slot_name = action['slot']
#                     if slot_name in ['INFORM', 'OFFER', 'REQUEST']:
#                         slot_value = action['values'][0] if  action['values'] else None
#                         if is_categorical[(service_name, slot_name)] and slot_value:
#                             new_frames[frame_id]['actions'][action_id]['values'][0] = digit2str(slot_value)
            
#         # replace words in utterance








