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
import matplotlib.pyplot as plt

data_to_paper_name = {
    "strategyqa": "StrategyQA",
    "basefakepedia": "BaseFakepedia",
    "multihopfakepedia": "MultiHopFakepedia",
    "openbookqa": "OpenBookQA",
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

# ---- Helpers ----
def shannon_entropy_bits(p, eps=1e-12):
    """
    Compute Shannon entropy (in bits) for a 1D probability vector.
    Robust to tiny/zero values; normalizes if not exactly summing to 1.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return -np.sum(p * np.log2(p))


def main(args):

    model_name = args.model_name.strip().split('/')[-1]

    dataset_list = ["strategyqa", "basefakepedia", "multihopfakepedia", "openbookqa"]

    # ---- ICLR-friendly styling ----
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    all_entropy_groups = []   # list of [dataset][quartile]
    quartile_labels = None

    # ---- Collect entropies for each dataset ----
    for dataset in dataset_list:

        with open(f"{args.data_dir}{dataset}/{model_name}/all_input_{args.prompt_mode}_nle.json", 'r') as f:
            all_input = json.load(f)

        probs_list = []
        for example in all_input:
            explanation_tokens_info = example['explanation_tokens']
            try:
                explanation_prob = torch.stack(
                    [torch.tensor(seq_step["prob"], dtype=torch.float32) for seq_step in explanation_tokens_info],
                    dim=0
                ).numpy()
                probs_list.append(explanation_prob)
            except:
                continue

        lengths = np.array([len(arr) for arr in probs_list])
        order = np.argsort(lengths)
        quartile_indices = np.array_split(order, 4)

        entropy_groups = []
        labels = []
        for i, idxs in enumerate(quartile_indices):
            if len(idxs) == 0:
                entropy_groups.append([])
                labels.append(f"Q{i+1}")
                continue
            lens = lengths[idxs]
            ents = [shannon_entropy_bits(probs_list[j]) for j in idxs]
            entropy_groups.append(ents)
            labels.append(f"Q{i+1} [{lens.min()}–{lens.max()}]")

        all_entropy_groups.append(entropy_groups)
        if quartile_labels is None:
            quartile_labels = labels

    # ---- Plot grouped boxplots ----
    n_datasets = len(dataset_list)
    n_quartiles = 4
    positions = np.arange(1, n_quartiles + 1)  # base x for quartiles
    width = 0.15  # spacing between datasets in each quartile

    colors = plt.cm.Set2.colors  # nice categorical colors

    for d, dataset in enumerate(dataset_list):
        # shift each dataset’s boxes slightly
        pos = positions + (d - n_datasets/2) * width + width/2
        bp = ax.boxplot(
            all_entropy_groups[d],
            positions=pos,
            widths=width*0.9,
            patch_artist=True,
            showfliers=False,
        )
        # color the boxes
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[d % len(colors)])
            patch.set_alpha(0.7)
        # add dataset name for legend (use a proxy artist)
        ax.plot([], c=colors[d % len(colors)], label=data_to_paper_name.get(dataset, dataset))

    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_xlabel("Length Quartiles (by rank)")
    ax.set_xticks(positions)
    ax.set_xticklabels(quartile_labels, rotation=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout(pad=0.6)
    out_pdf = os.path.join(
        args.output_dir, f"entropy_quartiles_all_datasets_{model_name}_nle.pdf"
    )
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved combined plot to: {out_pdf}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_w_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)

    main(args)
