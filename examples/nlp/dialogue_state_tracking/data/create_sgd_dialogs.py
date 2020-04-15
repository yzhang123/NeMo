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

"""
returns dict (domain, slot) -> value
"""

hotel_ranges = [
    "nigh",
    "moderate -ly priced",
    "bed and breakfast",
    "centre",
    "venetian",
    "intern",
    "a cheap -er hotel",
]
locations = ["gastropub", "la raza", "galleria", "gallery", "science", "m"]
detailed_hotels = ["hotel with free parking and free wifi", "4", "3 star hotel"]
areas = ["stansted airport", "cambridge", "silver street"]
attr_areas = ["norwich", "ely", "museum", "same area as hotel"]

def get_domain_slots(ontology: dict) -> dict:
    domain_to_slots = defaultdict(dict)
    for k, v in ontology.items():
        domain, slot = k.split("-")
        slot = normalize(slot)
        slot = slot.split()
        # if slot[0] == "book":
        #     slot = slot[1:]
        slot = "".join(slot)
        if slot == "":
            continue
        domain_slot_values = []
        for x in v:
            x =  normalize(x.strip().lower())
            x = GENERAL_TYPOS.get(x, x)
            # miss match slot and value
            if (
                (domain == "hotel" and slot == "type" and x in hotel_ranges)
                or (domain == "hotel" and slot == "internet" and x == "4")
                or (domain == "hotel" and slot == "pricerange" and x == "2")
                or (domain == "attraction" and slot == "type" and x in locations)
                or ("area" in slot and x in ["moderate"])
                or ("day" in slot and x == "t")
            ):
                continue  # x=none
            elif domain == "hotel" and slot == "type" and x in detailed_hotels:
                x = "hotel"
            elif domain == "hotel" and slot == "star" and x == "3 star hotel":
                x = "3"
            elif "area" in slot:
                if x == "no":
                    x = "north"
                elif x == "we":
                    x = "west"
                elif x == "cent":
                    x = "centre"
            elif "day" in slot:
                if x == "we":
                    x = "wednesday"
                elif x == "no":
                    x = "none"
            elif "price" in slot and x == "ch":
                x = "cheap"
            elif "internet" in slot and x == "free":
                x = "yes"

            # some out-of-define classification slot values
            if (domain == "restaurant" and slot == "area" and x in areas) or (
                domain == "attraction" and slot == "area" and x in attr_areas
            ):
                continue
            
            domain_slot_values.append(x)
        if domain_slot_values:
            domain_to_slots[domain][slot] = domain_slot_values

    return domain_to_slots


def get_schema_intents(domain: str, domain_slots: dict) -> List[dict]:
    intents = ["inform"]
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

def get_services_of_dialogue(dials: List[dict], dial_idx: int, used_domains: List[str]) -> List[str]:
    dialogue = dials[dial_idx]["dialogue"]
    services = set()
    for turn in dialogue:
        if turn["domain"] in used_domains:
            services.add(f"{turn['domain']}_1")
    return list(services)

def get_diag_sys_turn(turn: dict, turn_id: int, diag_acts: dict, prev_domain: str, possible_slots: dict)->dict:
    if turn["system_transcript"] == "":
        return None
    
    out_turn = dict()
    out_turn["speaker"] = "SYSTEM"
    turn["system_transcript"] = normalize(turn["system_transcript"])
    out_turn["utterance"] = turn["system_transcript"]
    domain = prev_domain
    frames = [{}]
    service = f"{domain}_1"
    actions = list()
    turn_id = str(turn_id)
    slots = []
    sys_acts = turn["system_acts"]
    for act_item in sys_acts:
        if isinstance(act_item, str):
            act_name = "REQUEST"
            slot_name = normalize(act_item)
            slot_val = []
        else:
            assert(isinstance(act_item, list))
            act_name = "INFORM"
            slot_name = normalize(act_item[0])
            slot_val = normalize(act_item[1])
        
            
        slot_name = SLOTNAME_REPLACEMENTS.get(slot_name, slot_name)
        
        out_act = dict()
        out_act["act"] = act_name # filter ACTIONS if needed
        out_slot_name = ""
        out_slot_val = []
        if slot_name != "none" and slot_name in possible_slots:
            out_slot_name = slot_name
            if not slot_val or slot_val in ["none", "?"] :
                out_slot_val = []
            else:
                out_slot_val = [slot_val]
                
                is_categorical = None
                for schema in SCHEMAS:
                    if schema["service_name"] == service:
                        for item in schema["slots"]:
                            if item["name"] == out_slot_name:
                                is_categorical = item["is_categorical"]
                if is_categorical:
                    continue
                try:
                    m = re.search(slot_val, turn["system_transcript"], re.IGNORECASE)
                except:
                    import ipdb; ipdb.set_trace()
                if m is not None:
                    if not  turn["system_transcript"][m.span()[0]:m.span()[1]] == slot_val.strip():
                        print(turn["system_transcript"][m.span()[0]:m.span()[1]] , ",", slot_val)
                        continue
                            
                    if len(turn["system_transcript"]) > m.span()[1] and turn["system_transcript"][m.span()[1]].isalnum():
                        print("ALPHANUM", turn["system_transcript"], slot_val)
                    
                        continue
                    if m.span()[0]>0 and turn["system_transcript"][m.span()[0]-1].isalnum():
                        print("ALPHANUM", turn["system_transcript"], slot_val)
                    
                        continue
                    slots.append({"slot": out_slot_name.strip(), "start": m.span()[0], "exclusive_end": m.span()[1]})
        else:
            # print(f"NOT FOUND {slot_name}:{slot_val}")
            continue

        out_act["slot"] = out_slot_name
        out_act["values"] = out_slot_val

        # TODO add rule -based filter for actions and their values, e.g. inform cannot have empty out_slot_val
        if act_name == "REQUEST":
            if out_slot_val: # nothing enters
                continue
        elif act_name in ["INFORM", "RECOMMEND"]:
            if not out_slot_val or not out_slot_name: 
                # print(f"#### {act_name}: {out_slot_name} {out_slot_val}")
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
    dom, slot = slot.split("-")
    if "book" in slot:
        slot = "".join(slot.split())
    return slot.strip(), dom

def get_diag_user_turn(turn: dict, domains: List[str])->dict:
    if turn["transcript"] == "":
        return None
    out_turn = dict()
    out_turn["speaker"] = "USER"
    turn["transcript"] = normalize(turn["transcript"])
    out_turn["utterance"] = turn["transcript"] 
    frames = list()

    for dom in domains:
        if not dom:
            continue
        frame = dict()
        frame["service"] = f"{dom}_1"

        frame["slots"] = []
        if turn["domain"] == dom: # current domain
            new_slots = turn["turn_label"]
            ## populate slot spans if possible
            for new_slot in new_slots:
                out_slot = dict()
                out_slot["slot"], _ = clean_slot(new_slot[0])
                is_categorical = None
                for schema in SCHEMAS:
                    if schema["service_name"] == frame["service"]:
                        for item in schema["slots"]:
                            if item["name"] == out_slot["slot"]:
                                is_categorical = item["is_categorical"]
                if is_categorical:
                    continue
                slot_val = new_slot[1]
                m = re.search(slot_val.strip(), turn["transcript"], re.IGNORECASE)
                if m is None:
                    continue
                
                if not turn["transcript"][m.span()[0]:m.span()[1]] == slot_val.strip():
                    print(turn["transcript"][m.span()[0]:m.span()[1]], ",", slot_val)
                    if "?" in slot_val:
                        import ipdb; ipdb.set_trace()
                    continue
                
                if len(turn["transcript"]) > m.span()[1] and turn["transcript"][m.span()[1]].isalnum():
                    print("ALPHANUM", turn["transcript"], slot_val)
                    continue
                
                if m.span()[0]>0 and turn["transcript"][m.span()[0]-1].isalnum():
                    print("ALPHANUM", turn["transcript"], slot_val)
                
                    continue
                out_slot["start"] = m.span()[0]
                out_slot["exclusive_end"] = m.span()[1]
                frame["slots"].append(out_slot)

        ## populate state based on state_belief
        all_slots = [x["slots"][0] for x in turn["belief_state"]] # list tuple
        frame["state"] = dict()
        frame["state"]["active_intent"] = f"{dom}_1"
        frame["state"]["requested_slots"] = []
        frame["state"]["slot_values"] = dict()
        for s, v in all_slots:
            s, d = clean_slot(s)
            if d != dom:
                continue
            if v != "none":
                frame["state"]["slot_values"][s] = [GENERAL_TYPOS.get(v, v)]
        frames.append(frame)
    out_turn["frames"] = frames
    return out_turn

def get_diag_turns(dials: List[dict], dial_idx: int, dialogue_acts: dict, domain_slots: dict) -> List[dict]:
    dialogue = dials[dial_idx]["dialogue"]
    out_turns = list()


    prev_domain = None
    for in_turn_id, in_turn in enumerate(dialogue):
        sys_turn = None
        user_turn = None
        current_domain = in_turn["domain"]
        if current_domain in domain_slots.keys():
            sys_turn = get_diag_sys_turn(in_turn, in_turn_id, dialogue_acts, prev_domain=prev_domain, possible_slots=domain_slots[prev_domain])

            if sys_turn is not None:
                out_turns.append(sys_turn)
            user_turn = get_diag_user_turn(in_turn, domains=list(set({current_domain, prev_domain})))
            if user_turn is not None:
                out_turns.append(user_turn)
            prev_domain = current_domain
    return out_turns




    return list(dict())

def main(args):

    ontology = json.load(open(args.source_ontology, 'r'))
    domains = get_domains(ontology)
    domain_slots = get_domain_slots(ontology)
    acts = json.load(open(args.source_acts, 'r'))
    domain_actions = get_domain_actions(domains, acts)
    source_dials = json.load(open(f"{args.source_dials}/{args.mode}_dials.json", 'r'))


    target_dials = list()
    diag_name_to_id = dict()
    diag_id_to_name = []
    for dial_idx, dial in enumerate(source_dials):
        new_diag = dict()
        diag_name = dial["dialogue_idx"].split(".json")[0]
        assert(diag_name not in diag_name_to_id)
        diag_name_to_id[diag_name] = len(diag_id_to_name)
        diag_id_to_name.append(diag_name)
        new_diag["dialogue_id"] = f"1_{len(diag_id_to_name):05d}"
        new_diag["services"] = get_services_of_dialogue(source_dials, dial_idx, domains)
        if not new_diag["services"]:
            continue
        new_diag["turns"] = get_diag_turns(source_dials, dial_idx, acts[diag_name], domain_slots)
        target_dials.append(new_diag)

    with open(f"{args.mode}_{args.target_dials}", 'w') as target_dials_fp:
        json.dump(target_dials, target_dials_fp, indent=4)


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description='Create MultiWOZ Schemas')
    parser.add_argument("--source_ontology", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/ontology.json', type=str)
    parser.add_argument("--source_multiwoz_mapping", default='multiwoz_mapping.pair', type=str)
    parser.add_argument("--source_general_typos", default='general_typo.json', type=str)
    parser.add_argument("--source_slotname_correct", default='slot_name_mapping.txt', type=str)
    parser.add_argument("--source_schemas", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/schemas.json', type=str)
    parser.add_argument("--source_acts", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/dialogue_acts.json', type=str)
    parser.add_argument("--source_dials", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/', type=str)
    parser.add_argument("--target_dials", default='dials_sgd.json', type=str)
    parser.add_argument("--mode", default='train', type=str)
    args = parser.parse_args()

    fin = open(args.source_multiwoz_mapping, 'r')
    REPLACEMENTS = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        REPLACEMENTS.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

    GENERAL_TYPOS = json.load(open(args.source_general_typos, 'r'))

    fin = open(args.source_slotname_correct, 'r')
    SLOTNAME_REPLACEMENTS = {}
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split()
        SLOTNAME_REPLACEMENTS[tok_from.strip()] = tok_to.strip()



    
    SCHEMAS = json.load(open(args.source_schemas, 'r'))

    main(args)

