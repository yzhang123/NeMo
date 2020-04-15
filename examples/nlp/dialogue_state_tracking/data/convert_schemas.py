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


def get_domains(ontology: dict) -> List[str]:
    domains = []
    for k in ontology:
        domain = k.split("-")[0].lower()
        if domain not in domains:
            domains.append(normalize(domain))
    assert len(domains) == 7
    return domains



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



def get_schema_slots(domain: str, domain_slots: dict) -> List[dict]:
    slots = domain_slots[domain] # dict: slot -> list of values
    res = list()
    for slot, values in slots.items():
        slot_entry = dict()
        slot_entry["name"] = slot
        slot_entry["description"] = f"{domain} {slot}"
        slot_entry["is_categorical"] = len(values)< 50 #slot in CATEGORICAL_SLOTS
        slot_entry["possible_values"] = values if slot_entry["is_categorical"] else []
        res.append(slot_entry)

    return res


def get_schema_intents(domain: str, domain_slots: dict) -> List[dict]:
    intents = [f"{domain}"]
    slots = domain_slots[domain].keys() # dict: slot -> list of values
    res = list()
    for intent in intents:
        intent_entry = dict()
        intent_entry["name"] = f"{intent}_1"
        intent_entry["description"] = f"{domain}"
        intent_entry["is_transactional"] = True # CHANGE?
        intent_entry["required_slots"] = []
        intent_entry["optional_slots"] = dict([(slot, "dontcare") for slot in slots])
        res.append(intent_entry)

    return res


def main(args):

    ontology = json.load(open(args.source_ontology, 'r'))
    # acts = json.load(open(args.source_acts, 'r'))
    domains = get_domains(ontology)
    domain_slots = get_domain_slots(ontology)

    with open(args.target_schema, 'w') as schema_fp:
        res = []
        for domain in domains:
            schema = {}
            schema["service_name"] = f"{domain}_1"
            schema["description"] = f"{domain}"
            schema_slots = get_schema_slots(domain, domain_slots)
            schema["slots"] = schema_slots
            schema_intents = get_schema_intents(domain, domain_slots)
            schema["intents"] = schema_intents
            res.append(schema)

        json.dump(res, schema_fp, indent=4)


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description='Create MultiWOZ Schemas')
    parser.add_argument("--source_multiwoz_mapping", default='multiwoz_mapping.pair', type=str)
    parser.add_argument("--source_general_typos", default='general_typo.json', type=str)
    parser.add_argument("--source_categorical_slots", default='categorical_slots.json', type=str)
    parser.add_argument("--source_ontology", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/ontology.json', type=str)
    # parser.add_argument("--source_acts", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/dialogue_acts.json', type=str)
    parser.add_argument("--target_schema", default='schemas.json', type=str)
    args = parser.parse_args()

    fin = open(args.source_multiwoz_mapping, 'r')
    REPLACEMENTS = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        REPLACEMENTS.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))
    
    CATEGORICAL_SLOTS = json.load(open(args.source_categorical_slots, 'r'))["categorical_slots"]
    GENERAL_TYPOS = json.load(open(args.source_general_typos, 'r'))

    main(args)

