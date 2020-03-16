import argparse
import json
import os
import re
import shutil
from os.path import exists, expanduser
from typing import List
from nemo.collections.nlp.data.datasets.datasets_utils import if_exist
from collections import defaultdict
from process_multiwoz import insertSpace


def get_domains(ontology: dict) -> List[str]:
    domains = []
    for k in ontology:
        domain = k.split("-")[0].lower()
        if domain not in domains:
            domains.append(domain)
    assert len(domains) == 7
    return domains

def get_domain_slots(ontology: dict) -> dict:
    domain_to_slots = defaultdict(dict)
    for k, v in ontology.items():
        domain, slot = k.split("-")
        slot = normalize(slot)
        slot = slot.split()
        if slot[0] == "book":
            slot = slot[1:]
        slot = "".join(slot)
        domain_to_slots[domain][slot] = v
        # fix arriveby -> arriveBy etc
    return domain_to_slots



def get_schema_slots(domain: str, domain_slots: dict) -> List[dict]:
    slots = domain_slots[domain] # dict: slot -> list of values
    res = list()
    for slot, values in slots.items():
        slot_entry = dict()
        slot_entry["name"] = slot
        slot_entry["description"] = f"{domain} {slot}"
        slot_entry["is_categorical"] = False # CHANGE?
        slot_entry["possible_values"] = values
        res.append(slot_entry)

    return res


def get_schema_intents(domain: str, domain_slots: dict) -> List[dict]:
    intents = ["request", "inform"]
    slots = domain_slots[domain].keys() # dict: slot -> list of values
    res = list()
    for intent in intents:
        intent_entry = dict()
        intent_entry["name"] = intent
        intent_entry["description"] = f"{domain} {intent}"
        intent_entry["is_transactional"] = True # CHANGE?
        intent_entry["required_slots"] = []
        intent_entry["optional_slots"] = dict([(slot, "dontcare") for slot in slots])
        res.append(intent_entry)

    return res

def get_domain_actions(domains: List[str], acts: dict) -> dict:
    """
        acts: dict over entire dialogue_acts.json
    """
    # without bookings
    domain_to_actions = defaultdict(list)
    dom_ints = [(x, diag[0]) for diag in acts.items() for turn in diag[1].values() for x in turn]
    for dom_int, id in dom_ints:
        try:
            dom, action = dom_int.lower().split("-")
        except:
            continue
        if dom in domains and action not in domain_to_actions[dom]:
            domain_to_actions[dom].append(action)
    return domain_to_actions


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in REPLACEMENTS:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def get_services_of_dialogue(dials: List[dict], dial_idx: int) -> List[str]:
    dialogue = dials[dial_idx]["dialogue"]
    services = set()
    for turn in dialogue:
        services.add(turn["domain"])
    return list(services)

def get_diag_sys_turn(turn: dict, turn_id: int, diag_acts: dict, prev_domain: str, possible_slots: dict)->dict:
    if turn["system_transcript"] == "":
        return None
    
    out_turn = dict()
    out_turn["speaker"] = "SYSTEM"
    out_turn["utterance"] = normalize(turn["system_transcript"])
    domain = prev_domain
    frames = [{}]
    service = f"{domain}_1"
    actions = list()
    turn_id = str(turn_id)
    slots = []
    if turn_id in diag_acts and diag_acts[turn_id] != "No Annotation":
        sys_acts = diag_acts[turn_id]
        for act, act_vals in sys_acts.items():
            act_name = normalize(act.split("-")[1]).upper()
            for slot_name, slot_val in act_vals:
                
                slot_name = normalize(slot_name)
                slot_val = normalize(slot_val)
                ##### TODO: fix and normlize slot_name with file
                if slot_name in SLOTNAME_REPLACEMENTS:
                    slot_name = SLOTNAME_REPLACEMENTS[slot_name]
                
                out_act = dict()
                out_act["act"] = act_name # filter ACTIONS if needed
                out_slot_name = ""
                out_slot_val = []
                if slot_name != "none" and slot_name in possible_slots:
                    out_slot_name = slot_name
                    if slot_val in ["none", "?"] :
                        out_slot_val = []
                    else:
                        out_slot_val = [slot_val]
                        m = re.match(slot_val, turn["system_transcript"], re.IGNORECASE)
                        if m is not None:
                            slots.append({"slot": out_slot_name, "start": m.span()[0], "exclusive_end": m.span()[1]})
                else:
                    print(f"NOT FOUND {slot_name}:{slot_val}")

                out_act["slot"] = out_slot_name
                out_act["values"] = out_slot_val

                # TODO add rule -based filter for actions and their values, e.g. inform cannot have empty out_slot_val
                if act_name == "REQUEST":
                    if out_slot_val: # nothing enters
                        continue
                elif act_name in ["INFORM", "RECOMMEND"]:
                    if not out_slot_val or not out_slot_name: 
                        print(f"#### {act_name}: {out_slot_name} {out_slot_val}")
                        continue

                if out_slot_val and not out_slot_name: # no enter
                    continue
                actions.append(out_act)

    frames[0]["actions"] = actions
    frames[0]["service"] = service
    frames[0]["slots"] = slots
    out_turn["frames"] = frames
    return out_turn

def clean_slot(slot: str)->str:
    slot = slot.split("-")[1]
    if "book" in slot:
        slot = slot.split()[1]
    return slot.strip()

def get_diag_user_turn(turn: dict)->dict:
    if turn["transcript"] == "":
        return None
    out_turn = dict()
    out_turn["speaker"] = "USER"
    out_turn["utterance"] = normalize(turn["transcript"])
    frames = list()
    dom = turn["domain"]
    
    frame = dict()
    frame["service"] = f"{dom}_1"
    new_slots = turn["turn_label"]
    frame["slots"] = []
    ## populate slot spans if possible
    for new_slot in new_slots:
        out_slot = dict()
        out_slot["slot"] = clean_slot(new_slot[0])
        slot_val = new_slot[1]
        m = re.match(slot_val, turn["transcript"], re.IGNORECASE)
        if m is None:
            continue
        out_slot["start"] = m.span()[0]
        out_slot["exclusive_end"] = m.span()[1]
        frame["slots"].append(out_slot)

    ## populate state based on state_belief
    all_slots = [x["slots"][0] for x in turn["belief_state"]] # list tuple
    frame["state"] = dict()
    frame["state"]["active_intent"] = f"find{dom}"
    frame["state"]["requested_slots"] = []
    frame["state"]["slot_values"] = dict()
    for s, v in all_slots:
        s = clean_slot(s)
        frame["state"]["slot_values"][s] = [v]
    frames.append(frame)
    out_turn["frames"] = frames
    return out_turn

def get_diag_turns(dials: List[dict], dial_idx: int, dialogue_acts: dict, domain_slots: dict) -> List[dict]:
    dialogue = dials[dial_idx]["dialogue"]
    out_turns = list()

    for in_turn_id, in_turn in enumerate(dialogue):
        sys_turn = None
        if in_turn_id > 0:
            prev_domain=dialogue[in_turn_id - 1]["domain"]
            sys_turn = get_diag_sys_turn(in_turn, in_turn_id, dialogue_acts, prev_domain=prev_domain, possible_slots=domain_slots[prev_domain])

        if sys_turn is not None:
            out_turns.append(sys_turn)
        user_turn = get_diag_user_turn(in_turn)
        if user_turn is not None:
            out_turns.append(user_turn)
    return out_turns




    return list(dict())

def main(args):

    ontology = json.load(open(args.source_ontology, 'r'))
    domains = get_domains(ontology)
    domain_slots = get_domain_slots(ontology)
    acts = json.load(open(args.source_acts, 'r'))
    domain_actions = get_domain_actions(domains, acts)
    source_dials = json.load(open(args.source_dials, 'r'))


    target_dials = list()
    for dial_idx, dial in enumerate(source_dials):
        new_diag = dict()
        new_diag["dialogue_id"] = dial["dialogue_idx"].split(".json")[0]
        new_diag["services"] = get_services_of_dialogue(source_dials, dial_idx)
        new_diag["turns"] = get_diag_turns(source_dials, dial_idx, acts[new_diag["dialogue_id"]], domain_slots)
        target_dials.append(new_diag)

    with open(args.target_dials, 'w') as target_dials_fp:
        json.dump(target_dials, target_dials_fp, indent=4)


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description='Create MultiWOZ Schemas')
    parser.add_argument("--source_ontology", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/ontology.json', type=str)
    parser.add_argument("--source_correct", default='mapping.pair', type=str)
    parser.add_argument("--source_slotname_correct", default='slot_name_mapping.txt', type=str)
    parser.add_argument("--source_acts", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/dialogue_acts.json', type=str)
    parser.add_argument("--source_dials", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/train_dials.json', type=str)
    parser.add_argument("--target_dials", default='train_dials_sgd.json', type=str)
    args = parser.parse_args()

    fin = open(args.source_correct, 'r')
    REPLACEMENTS = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        REPLACEMENTS.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

    fin = open(args.source_slotname_correct, 'r')
    SLOTNAME_REPLACEMENTS = {}
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split()
        SLOTNAME_REPLACEMENTS[tok_from.strip()] = tok_to.strip()

    main(args)

