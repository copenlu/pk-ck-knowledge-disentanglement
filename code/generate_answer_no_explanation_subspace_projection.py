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
from nnpatch.subspace import LowRankOrthogonalProjection


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

    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_hs_{args.prompt_mode}_no_explanation.pkl", 'rb') as f:
        all_pred_hs = pickle.load(f)

    proj_rank1 = LowRankOrthogonalProjection.from_pretrained("jkminder/CTXPRIOR-Projection-Mistral-7B-Instruct-v0.3-L16").to(args.device)
    proj_rank2 = LowRankOrthogonalProjection.from_pretrained("../context-vs-prior-finetuning/analysis/projections/Mistral-7B-Instruct-v0.3-L16").to(args.device)

    with open(f"{args.data_dir}{args.dataset}/{model_name}/knowledge_intearction_dict_no_explanation.json", "r") as f:
        knowledge_intearction_dict = json.load(f)

    df = pd.read_csv(f"{args.data_dir}{args.dataset}/{model_name}/projection_data.csv")

    ctx_weight_list = df['weight_context'].tolist()
    prior_weight_list = df['weight_prior'].tolist()

    example_idx_to_knowledge_interaction = {}

    all_proj_rank1, all_proj_rank2 = [], []
    supp_proj_rank1, supp_proj_rank2 = [], []
    ctx_conf_proj_rank1, ctx_conf_proj_rank2 = [], []
    prior_conf_proj_rank1, prior_conf_proj_rank2 = [], []
    comp_proj_rank1, comp_proj_rank2 = [], []
    irr_proj_rank1, irr_proj_rank2 = [], []

    for key in knowledge_intearction_dict.keys():
        example_idx_list = knowledge_intearction_dict[key]
        for example_idx in example_idx_list:
            example_idx_to_knowledge_interaction[int(example_idx)] = key

    all_proj_rank1, all_proj_rank2 = [], []
    supp_proj_rank1, supp_proj_rank2 = [], []
    ctx_conf_proj_rank1, ctx_conf_proj_rank2 = [], []
    prior_conf_proj_rank1, prior_conf_proj_rank2 = [], []
    comp_proj_rank1, comp_proj_rank2 = [], []
    irr_proj_rank1, irr_proj_rank2 = [], []

    for example_idx, example in enumerate(all_pred_hs):
        try:
            ans_hs = torch.tensor(example
            ).to(args.device)

            explanation_rank1_contrib = ans_hs @ proj_rank1.weight
            explanation_rank2_contrib = ans_hs @ proj_rank2.weight

            all_proj_rank1.append(explanation_rank1_contrib.detach().cpu().numpy())
            all_proj_rank2.append(explanation_rank2_contrib.detach().cpu().numpy())
            cat = example_idx_to_knowledge_interaction[example_idx]

            if cat == 'supportive':
                supp_proj_rank1.append(explanation_rank1_contrib.detach().cpu().numpy())
                supp_proj_rank2.append(explanation_rank2_contrib.detach().cpu().numpy())
            elif cat == 'conflicting':
                if float(ctx_weight_list[example_idx]) == 1.0:
                    ctx_conf_proj_rank1.append(explanation_rank1_contrib.detach().cpu().numpy())
                    ctx_conf_proj_rank2.append(explanation_rank2_contrib.detach().cpu().numpy())
                elif float(prior_weight_list[example_idx]) == 1.0:
                    prior_conf_proj_rank1.append(explanation_rank1_contrib.detach().cpu().numpy())
                    prior_conf_proj_rank2.append(explanation_rank2_contrib.detach().cpu().numpy())
            elif cat == 'complementary':
                comp_proj_rank1.append(explanation_rank1_contrib.detach().cpu().numpy())
                comp_proj_rank2.append(explanation_rank2_contrib.detach().cpu().numpy())
            elif cat == 'irrelevant':
                irr_proj_rank1.append(explanation_rank1_contrib.detach().cpu().numpy())
                irr_proj_rank2.append(explanation_rank2_contrib.detach().cpu().numpy())
        except:
            continue


    #Write results
    os.makedirs(f"{args.data_dir}{args.dataset}/{model_name}", exist_ok=True)
    os.makedirs(f"{args.output_dir}{args.dataset}/{model_name}", exist_ok=True)

    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_proj_rank1_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(all_proj_rank1, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/all_proj_rank2_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(all_proj_rank2, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/supp_proj_rank1_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(supp_proj_rank1, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/supp_proj_rank2_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(supp_proj_rank2, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/ctx_conf_proj_rank1_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(ctx_conf_proj_rank1, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/ctx_conf_proj_rank2_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(ctx_conf_proj_rank2, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/prior_conf_proj_rank1_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(prior_conf_proj_rank1, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/prior_conf_proj_rank2_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(prior_conf_proj_rank2, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/comp_proj_rank1_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(comp_proj_rank1, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/comp_proj_rank2_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(comp_proj_rank2, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/irr_proj_rank1_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(irr_proj_rank1, f)
    with open(f"{args.output_dir}{args.dataset}/{model_name}/irr_proj_rank2_{args.prompt_mode}_no_explanation.pkl", 'wb') as f:
        pickle.dump(irr_proj_rank2, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_w_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
