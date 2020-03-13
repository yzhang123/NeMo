import argparse
import json
import os
import re
import shutil
from os.path import exists, expanduser
from typing import List
from nemo.collections.nlp.data.datasets.datasets_utils import if_exist
from collections import defaultdict


def get_domains(ontology: dict) -> List[str]:
    domains = []
    for k in ontology:
        domain = k.split("-")[0].lower()
        if domain not in domains:
            domains.append(domain)
    assert len(domains) == 7
    return domains


def get_domain_intents(domains: List[str], acts: dict) -> dict:
    """
        acts: dict over entire dialogue_acts.json
    """

    domain_to_intents = defaultdict(list)
    dom_ints = [(x, diag[0]) for diag in acts.items() for turn in diag[1].values() for x in turn]
    for dom_int, id in dom_ints:
        try:
            dom, intent = dom_int.lower().split("-")
        except:
            continue
        if dom in domains and intent not in domain_to_intents[dom]:
            domain_to_intents[dom].append(intent)
    return domain_to_intents


def get_domain_slots(ontology: dict) -> dict:
    domain_to_slots = defaultdict(dict)
    for k, v in ontology.items():
        domain, slot = k.split("-")
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
        slot_entry["is_categorical"] = True # CHANGE?
        slot_entry["possible_values"] = values
        res.append(slot_entry)

    return res


def get_schema_intents(domain: str, domain_intents: dict, domain_slots: dict) -> List[dict]:
    intents = domain_intents[domain] # list
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
    domain_intents = get_domain_intents(domains, acts)

    with open(args.target_schema, 'w') as schema_fp:
        res = []
        for domain in domains:
            schema = {}
            schema["service_name"] = f"{domain}_1"
            schema["description"] = f"{domain}"
            schema_slots = get_schema_slots(domain, domain_slots)
            schema["slots"] = schema_slots
            schema_intents = get_schema_intents(domain, domain_intents, domain_slots)
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

    main(args)

