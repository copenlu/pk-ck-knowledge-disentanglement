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


def main(args):
    model_name = args.model_name.strip().split('/')[-1]
    
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_both_wo_instruction_cot.json", 'r') as f:
        all_input_cot = json.load(f)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_both_wo_instruction_no_explanation.json", 'r') as f:
        all_input_non_cot = json.load(f)

    context_list, query_list, cot_answer_list, non_cot_answer_list = [], [], [], []

    diff_count = 0
    for idx in range(len(all_input_cot)):
        try:
            context_cot = all_input_cot[idx]['context'].strip()
            context_non_cot = all_input_non_cot[idx]['context'].strip()
            assert context_cot == context_non_cot

            query_cot = all_input_cot[idx]['query'].strip()
            query_non_cot = all_input_non_cot[idx]['query'].strip()
            assert query_cot == query_non_cot

            context = context_cot
            query = query_cot
            cot_answer = all_input_cot[idx]['cot_ans']
            non_cot_answer = all_input_non_cot[idx]['both_wo_instruction_ans']

            if cot_answer != non_cot_answer:
                diff_count += 1
            
            context_list.append(context)
            query_list.append(query)
            cot_answer_list.append(cot_answer)
            non_cot_answer_list.append(non_cot_answer)
        except:
            continue

    df = pd.DataFrame({'context': context_list, 'query': query_list, 'cot_answer': cot_answer_list, 'non_cot_answer': non_cot_answer_list})
    df.to_csv(f"{args.data_dir}{args.dataset}/{model_name}/projection_data_cot.csv", index=False)
    print(diff_count)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
