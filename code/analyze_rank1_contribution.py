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
from nnpatch.subspace import LowRankOrthogonalProjection, BinaryHook
import matplotlib.pyplot as plt

data_to_paper_name = {'strategyqa': 'StrategyQA', 
                    'basefakepedia': 'BaseFakepedia', 
                    'multihopfakepedia': 'MultihopFakepedia', 
                    'openbookqa': 'OpenBookQA'}


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
    # Models / tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model_name = args.model_name.strip().split('/')[-1]

    dataset_list = ['strategyqa', 'basefakepedia', 'multihopfakepedia', 'openbookqa']

    # ---- ICLR-friendly styling ----
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,           # good for 10pt papers
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,       # TrueType fonts (embed as text)
        "ps.fonttype": 42
    })

    # One-column ICLR width is 5.5 in; pick a compact height
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), constrained_layout=False)
    axes = axes.flatten()
    all_handles, all_labels = [], []

    for idx, dataset in enumerate(dataset_list):
        # --- load ---
        with open(f"{args.output_dir}{dataset}/{model_name}/all_pred_hs_{args.prompt_mode}_no_explanation.pkl", "rb") as f:
            all_pred_hs = pickle.load(f)
        all_proj_hs = torch.tensor(all_pred_hs).to(args.device)

        if 'llama' in args.model_name:
            proj = LowRankOrthogonalProjection.from_pretrained(
                "jkminder/CTXPRIOR-Projection-Meta-Llama-3.1-8B-Instruct-L16"
            ).to(model.device)
        elif 'gemma' in args.model_name:
            proj = LowRankOrthogonalProjection.from_pretrained(
                "jkminder/CTXPRIOR-Projection-gemma-2-9b-it-L27"
            ).to(model.device)
        elif 'mistral' in args.model_name:
            proj = LowRankOrthogonalProjection.from_pretrained(
                "jkminder/CTXPRIOR-Projection-Mistral-7B-Instruct-v0.3-L16"
            ).to(model.device)

        rank1_contrib = torch.matmul(all_proj_hs, proj.weight).detach().cpu().numpy()

        with open(f"{args.data_dir}{dataset}/{model_name}/knowledge_intearction_dict_no_explanation.json", "r") as f:
            knowledge_intearction_dict = json.load(f)

        data = {k: rank1_contrib[v] for k, v in knowledge_intearction_dict.items()}

        # --- prepare arrays ---
        def to_finite_array(x):
            a = np.asarray(x, dtype=float).ravel()
            return a[np.isfinite(a)]

        arrays = {k: to_finite_array(v) for k, v in data.items() if to_finite_array(v).size > 0}
        if not arrays:
            raise ValueError(f"No finite values to plot for dataset {dataset}")

        all_vals = np.concatenate(list(arrays.values()))
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        pad = 0.05 * max(1e-12, x_max - x_min)
        grid = np.linspace(x_min - pad, x_max + pad, 512)

        # --- KDE helpers ---
        def silverman_bandwidth(x):
            x = np.asarray(x, dtype=float).ravel()
            n = x.size
            if n < 2:
                return 0.1 if n == 0 else 1.06 * (np.std(x) + 1e-12)
            std = np.std(x, ddof=1)
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            sigma = min(std, iqr / 1.34) if (std > 0 and iqr > 0) else max(std, iqr / 1.34, 1e-12)
            h = 0.9 * sigma * n ** (-1/5)
            if not np.isfinite(h) or h <= 0:
                rng = np.max(x) - np.min(x)
                h = max(1e-3, 0.1 * (rng if rng > 0 else 1.0))
            return h

        def kde_gaussian(x, grid, h):
            x = np.asarray(x, dtype=float).ravel()
            if x.size == 0:
                return np.zeros_like(grid)
            diff = (grid[:, None] - x[None, :]) / h
            K = np.exp(-0.5 * diff * diff)
            const = 1.0 / (np.sqrt(2 * np.pi) * h * x.size)
            return const * K.sum(axis=1)

        # --- plot this dataset ---
        ax = axes[idx]
        handles = []
        labels = []
        for cat, vals in arrays.items():
            h = silverman_bandwidth(vals)
            dens = kde_gaussian(vals, grid, h)
            line, = ax.plot(grid, dens)#, label=f"{cat} (h={h:.3g})")
            ax.fill_between(grid, dens, alpha=0.15)
            handles.append(line)
            labels.append(cat)
        ax.set_title(data_to_paper_name[dataset])
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2, linewidth=0.5)
        # Keep legends compact; move them inside the axes
        # ax.legend(loc="upper right", frameon=False, handlelength=1.0)

        # Collect handles/labels once (first subplot)
        if idx == 0:
            all_handles, all_labels = handles, labels

        # Clean shared layout
        # bottom row → xlabel
        if idx >= 2:
            ax.set_xlabel("Subspace component")
        else:
            ax.set_xlabel("")

        # left column → ylabel
        if idx % 2 == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")

    # --- Shared Legend (bottom of entire figure) ---
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_labels),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.2,
        bbox_to_anchor=(0.5, -0.10)
    )

    # Tight layout for ICLR column width; minimal padding
    plt.tight_layout(pad=0.6, w_pad=0.6, h_pad=0.6)

    # Save a SINGLE vector PDF (good for LaTeX: \includegraphics[width=\linewidth]{...})
    out_pdf = f"{args.output_dir}rank1_proj_contrib_{model_name}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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
