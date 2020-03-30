import os
import sys
import argparse
import json

parser = argparse.ArgumentParser(description='extract text data`')
parser.add_argument("--dataset", choices=["multiwoz", "sgd", "atis", "snip"], type=str, nargs='+')
parser.add_argument("--multiwoz_dir", default=None, type=str)
parser.add_argument("--sgd_dir", default=None, type=str)
parser.add_argument("--atis_dir", default=None, type=str)
parser.add_argument("--snip_dir", default=None, type=str)
parser.add_argument("--target_file", default=None, type=str)


args = parser.parse_args()
dialogues = []
if args.multiwoz_dir is not None:
    filepath = os.path.join(args.multiwoz_dir, "train_dials.json")
    dialogues_json = json.load(open(filepath, 'r'))
    for dialogue_json in dialogues_json:
        dialogue = []
        for turn in dialogue_json["dialogue"]:
            sys_txt = turn['system_transcript']
            user_txt = turn['transcript']
            sys_txt = sys_txt.strip()
            user_txt = user_txt.strip()
            if sys_txt and sys_txt[-1] not in [".", "?"]:
                sys_txt += " ."
            if user_txt and user_txt[-1] not in [".", "?"]:
                user_txt += " ."
            dialogue.append(f"{sys_txt} {user_txt}")
        dialogues.append(dialogue)

if args.sgd_dir is not None:
    dirpath = os.path.join(args.sgd_dir, "train")
    filepaths = []
    for file in os.listdir(dirpath):
        if file.endswith(".json") and file.startswith("dialogue"):
            filepaths.append(os.path.join(dirpath, file))
    for filepath in filepaths:
        dialogues_json = json.load(open(filepath, 'r'))
        for dialogue_json in dialogues_json:
            dialogue = []
            sys_txt = ""
            user_txt = ""
            for turn in dialogue_json["turns"]:
                if turn["speaker"] == "SYSTEM":
                    sys_txt = turn["utterance"]
                if turn["speaker"] == "USER":
                    user_txt = turn["utterance"]
                    sys_txt = sys_txt.strip()
                    user_txt = user_txt.strip()
                    dialogue.append(f"{sys_txt} {user_txt}")
                    sys_txt = ""
                    user_txt = ""
            dialogues.append(dialogue)

with open(args.target_file, 'w') as out_fp:
    text="\n\n".join(["\n".join(x) for x in dialogues])
    out_fp.write(text)

    







