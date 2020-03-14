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
    intents = [f"find{domain}"]
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


def main(args):

    ontology = json.load(open(args.source_ontology, 'r'))
    acts = json.load(open(args.source_acts, 'r'))
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
    parser.add_argument("--source_ontology", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/ontology.json', type=str)
    parser.add_argument("--source_acts", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/dialogue_acts.json', type=str)
    parser.add_argument("--target_schema", default='/home/yzhang/data/nlp/MULTIWOZ2.1_bak/schemas.json', type=str)
    args = parser.parse_args()

    fin = open('mapping.pair', 'r')
    REPLACEMENTS = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        REPLACEMENTS.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

    main(args)

