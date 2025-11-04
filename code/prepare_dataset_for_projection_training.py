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
    
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_prior_only_no_explanation.json", 'r') as f:
        all_input_prior_only = json.load(f)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_context_only_no_explanation.json", 'r') as f:
        all_input_context_only = json.load(f)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/all_input_both_w_instruction_no_explanation.json", 'r') as f:
        all_input_both_w_instruction = json.load(f)

    context_list, query_list, answer_list, ctx_answer_list, prior_answer_list, weight_context_list, weight_prior_list = [], [], [], [], [], [], []

    supportive_num = 0 
    complementary_num = 0
    conflicting_num = 0
    irrlevant_num = 0
    outlier_num = 0

    knowledge_intearction_dict = {
        'supportive': [],
        'complementary': [],
        'conflicting': [],
        'irrelevant': [],
        'noise': []
    }

    for idx in range(len(all_input_both_w_instruction)):
        try:
            context_both_w_instruction = all_input_both_w_instruction[idx]['context'].strip()
            context_prior_only = all_input_prior_only[idx]['context'].strip()
            context_context_only = all_input_context_only[idx]['context'].strip()
            assert context_both_w_instruction == context_prior_only == context_context_only

            query_both_w_instruction = all_input_both_w_instruction[idx]['query'].strip()
            query_prior_only = all_input_prior_only[idx]['query'].strip()
            query_context_only = all_input_context_only[idx]['query'].strip()
            assert query_both_w_instruction == query_prior_only == query_context_only

            context = context_both_w_instruction
            query = query_both_w_instruction
            answer = all_input_both_w_instruction[idx]['both_w_instruction_ans'].strip()
            ctx_answer = all_input_context_only[idx]['context_only_ans'].strip()
            prior_answer = all_input_prior_only[idx]['prior_only_ans'].strip()

            unk_to_id = 'E'
            if args.dataset == 'openbookqa' or args.dataset == 'musique':
                query_split_list = query.strip().split('\n')
                for item in query_split_list[1:]:
                    if 'unknown' in item.lower():
                        unk_to_id = item[0]

            if ctx_answer == prior_answer:
                if ctx_answer == answer:
                    weight_context = weight_prior = 1.0
                    supportive_num += 1
                    knowledge_intearction_dict['supportive'].append(idx)
                else:
                    weight_context = weight_prior = 0.0 
                    outlier_num += 1
                    knowledge_intearction_dict['noise'].append(idx)
            else:
                if ctx_answer == answer:
                    weight_context = 1.0
                    weight_prior = 0.0
                    conflicting_num += 1
                    knowledge_intearction_dict['conflicting'].append(idx)
                elif prior_answer == answer:
                    weight_context = 0.0
                    weight_prior = 1.0
                    if ctx_answer.lower() == 'unknown' or ctx_answer == unk_to_id:
                        irrlevant_num += 1
                        knowledge_intearction_dict['irrelevant'].append(idx)
                    else:
                        conflicting_num += 1
                        knowledge_intearction_dict['conflicting'].append(idx)
                else:
                    weight_context = weight_prior = 0.0
                    complementary_num += 1
                    knowledge_intearction_dict['complementary'].append(idx)
            
            context_list.append(context)
            query_list.append(query)
            answer_list.append(answer)
            ctx_answer_list.append(ctx_answer)
            prior_answer_list.append(prior_answer)
            weight_context_list.append(weight_context)
            weight_prior_list.append(weight_prior)
        except:
            continue

    df = pd.DataFrame({'context': context_list, 'query': query_list, 'answer': answer_list, 'ctx_answer': ctx_answer_list, 'prior_answer': prior_answer_list, 'weight_context': weight_context_list, 'weight_prior': weight_prior_list})
    df.to_csv(f"{args.data_dir}{args.dataset}/{model_name}/projection_data.csv", index=False)
    with open(f"{args.data_dir}{args.dataset}/{model_name}/knowledge_intearction_dict_no_explanation.json", 'w') as f:
        json.dump(knowledge_intearction_dict, f, indent=4)
    print("supportive:", supportive_num, "| conflicting:", conflicting_num, "| complementary:", complementary_num, "| irrelevant:", irrlevant_num, "| outlier:", outlier_num)



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
