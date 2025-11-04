import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools
import torch
#import hf_olmo
import random
import numpy as np
import time
import pickle
import pandas as pd
import ast
import re
from difflib import SequenceMatcher
import unicodedata


option_id_to_letter = {
    '0': 'A',
    '1': 'B',
    '2': 'C',
    '3': 'D',
    '4': 'E'
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        print('No GPU available, using the CPU instead.')


def load_dataset(args):
    if args.dataset == "musique":
        df = pd.read_csv('../data/echoqa_dataset/musique_data.csv')
        query_list = df['query'].tolist()
        query_mod_list = []

        for query in query_list:
            parts = query.strip().split('\n')
            if not parts:
                query_mod_list.append(query)
                continue

            question = parts[0].strip()
            options_list = [line.strip() for line in parts[1:]]

            option_list_mod = []
            for option in options_list:
                # keep only lines that look like "0. " ... "4. "
                if not option or not re.match(r'^[0-4]\.\s', option):
                    continue

                # split on "digit + dot + space"
                chunks = [p.strip() for p in re.split(r"\d\. ", option) if p.strip()]

                if len(chunks) > 1:
                    # line contains multiple options on one line, e.g. "0. a 1. b 2. c"
                    start_idx = int(option[0])
                    parts_mod = [
                        f"{option_id_to_letter[str(start_idx + i)]}. {txt}"
                        for i, txt in enumerate(chunks)
                    ]
                    option_mod = "\n".join(parts_mod)
                else:
                    # simple case: a single option like "0. something"
                    option_mod = option_id_to_letter[option[0]] + option[1:]

                option_list_mod.append(option_mod)

            options_str = '\n'.join(option_list_mod)
            query_mod = question + ('\n' + options_str if options_str else '')
            query_mod_list.append(query_mod)

        df['query'] = query_mod_list
        df.to_csv('../data/echoqa_dataset/musique_data.csv', index=False)



def main(args):

    load_dataset(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
