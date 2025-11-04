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
import matplotlib.pyplot as plt
import matplotlib as mpl


def main(args):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],  # replace with "Helvetica" if available
        "font.size": 8,          # base font
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,       # high-res output
    })

    #Write results
    model_name = args.model_name.strip().split('/')[-1]
    
    prompt_mode_list = ['prior_only', 'context_only', 'both_w_instruction']

    data = []
    for prompt_mode in prompt_mode_list:
        with open(f"{args.output_dir}{args.dataset}/{model_name}/all_pred_prob_{prompt_mode}_no_explanation.pkl", 'rb') as f:
            all_pred_prob = pickle.load(f)
        data.append(all_pred_prob)

    # --- Create figure (single-column friendly size: ~3.5–4.5 inches wide) ---
    fig_w, fig_h = 4.2, 3.0  # inches
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    # Boxplot styling
    box = ax.boxplot(
        data,
        labels=["Prior only", "Context only", "Both"],
        showmeans=False,
        showfliers=False,  # cleaner for papers
        #widths=0.6,
        patch_artist=False,  # keep simple line art
        # boxprops=dict(linewidth=1.5),
        # whiskerprops=dict(linewidth=1.5),
        # capprops=dict(linewidth=1.5),
        # medianprops=dict(linewidth=1.8),
    )

    # Axes labels/title
    ax.set_ylabel("Answer probability")
    ax.set_xlabel("Intent")
    #ax.set_title(f"Prediction probabilities across prompt modes ({model_name})", pad=8)

    # Grid (y-only, light)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: y-limits (uncomment if your probabilities are in [0,1])
    #ax.set_ylim(0.0, 1.0)

    # Tight layout & save
    fig.tight_layout()
    fig.savefig(f"{args.output_dir}{args.dataset}/{model_name}/pred_prob_box_plot_{args.dataset}.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    #print(f"Saved figure to: {out_path}")
    

        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model : meta-llama/Llama-Guard-3-8B ,
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="strategyqa")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instrcution", default="both")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    #set_seed(args)

    main(args)
