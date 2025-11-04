import json
import os
from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer  # not used here
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

data_to_paper_name = {
    'strategyqa': 'StrategyQA',
    'basefakepedia': 'BaseFakepedia',
    'multihopfakepedia': 'MultihopFakepedia',
    'openbookqa': 'OpenBookQA'  # fixed typo
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if ("cuda" in args.device) and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # deterministic + benchmark True is contradictory → make benchmark False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f'There are {torch.cuda.device_count()} GPU(s) available. Using {args.device}.')
    else:
        print('No GPU available or CUDA not requested, using the CPU instead.')

# ---- Core: compute SVD of between-variant means ----
def between_svd_last_token(hidden_by_variant, center=True):
    """
    hidden_by_variant: list of np.ndarray, length V (e.g., 5 variants)
        each array is shape (N_v, d) with last-layer hidden of the answer token
    Returns:
        U: (d, V), S: (V,), Vt: (V, V), M: (d, V), mu_mat: (d, V), grand_mu: (d,)
    """
    # guard: ensure no empty variant
    for idx, X in enumerate(hidden_by_variant):
        if X is None or len(X) == 0:
            raise ValueError(f"Variant {idx} has no examples (empty array).")

    # per-variant means (d,)
    mus = [X.mean(axis=0) for X in hidden_by_variant]
    mu_mat = np.stack(mus, axis=1)            # (d, V)
    if center:
        grand_mu = mu_mat.mean(axis=1, keepdims=True)  # (d, 1)
        M = mu_mat - grand_mu                          # (d, V)
    else:
        grand_mu = np.zeros((mu_mat.shape[0], 1), dtype=mu_mat.dtype)
        M = mu_mat
    U, S, Vt = np.linalg.svd(M, full_matrices=False)   # S desc
    return U, S, Vt, M, mu_mat, grand_mu.squeeze()

def explained_variance_from_S(S):
    """Cumulative fraction of between-condition energy explained by top-k."""
    total = float((S**2).sum())
    if total == 0.0:
        return np.zeros_like(S)
    return np.cumsum(S**2) / total

# ---- Permutation test for significance of singular values ----
def perm_test_sigmas_last_token(hidden_by_variant, n_perm=1000, seed=0):
    """
    Shuffles variant labels across examples to form a null for singular values.
    Returns dict with observed S, null_s (n_perm, V), and p-values (V,)
    """
    rng = np.random.default_rng(seed)
    V = len(hidden_by_variant)

    # Build concatenated pool and label vector
    X_all = np.concatenate(hidden_by_variant, axis=0)      # (N_tot, d)
    labels = np.concatenate([
        np.full(len(X), v, dtype=int) for v, X in enumerate(hidden_by_variant)
    ])

    # Observed sigmas
    _, S_obs, _, *_ = between_svd_last_token(hidden_by_variant)

    # Null distribution
    null_s = np.zeros((n_perm, len(S_obs)), dtype=S_obs.dtype)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        perm_groups = [X_all[perm == v] for v in range(V)]
        # guard: if any group becomes empty due to extreme imbalance (rare), resample
        if any(len(g) == 0 for g in perm_groups):
            # skip this permutation; keep zeros (conservative)
            continue
        _, S_null, _, *_ = between_svd_last_token(perm_groups)
        s = np.zeros_like(S_obs)
        s[:len(S_null)] = S_null  # pad if numerical rank < V
        null_s[i] = s

    # One-sided p-values: P(null >= observed)
    # if a null row is all zeros (skipped), it won't inflate p unfairly
    pvals = (null_s >= S_obs[None, :]).mean(axis=0)
    return {"S_obs": S_obs, "null_sigmas": null_s, "pvals": pvals}

def compute_svd_stats(dataset, model_name, output_dir, prompt_mode_list):
    X = []
    for pm in prompt_mode_list:
        path = os.path.join(output_dir, dataset, model_name,
                            f"all_pred_hs_{pm}_no_explanation.pkl")
        with open(path, "rb") as f:
            arr = pickle.load(f)
        arr = np.array(arr)
        assert isinstance(arr, np.ndarray) and arr.ndim == 2
        X.append(arr)

    U, S, Vt, M, mu_mat, grand = between_svd_last_token(X, center=True)
    ev = explained_variance_from_S(S)
    return S, ev

def main(args):

    # ---- ICLR-friendly styling ----
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })


    datasets = ["strategyqa", "basefakepedia", "multihopfakepedia", "openbookqa"]
    models = ["Meta-Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-7B-Instruct-v0.3"]
    prompt_mode_list = [
        # "prior_wo_context",
        "prior_only",
        "context_only",
        "both_w_instruction",
        # "both_wo_instruction"
    ]

    n_rows, n_cols = len(datasets), len(models)
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows), sharey=True)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), constrained_layout=False, sharex=True, sharey=True)
    axes = axes.flatten()

    lines_for_legend = []
    labels_for_legend = []

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, model in enumerate(models):
            try:
                _, S = compute_svd_stats(ds, model, args.output_dir, prompt_mode_list)
            except FileNotFoundError as e:
                print(f"[WARN] Missing files for {ds} | {model}: {e}")
                continue
            except Exception as e:
                print(f"[WARN] Skipping {ds} | {model} due to error: {e}")
                continue

            x = np.arange(1, len(S) + 1)
            line = ax.plot(x, S, marker="o", linestyle="-", label=model)[0]
            # Title for each subplot = dataset name
            ax.set_title(data_to_paper_name[ds])
            # collect legend handle only once per model (from first subplot)
            if i == 0:
                lines_for_legend.append(line)
                labels_for_legend.append(model)

        # Clean shared layout
        # bottom row → xlabel
        if i >= 2:
            ax.set_xlabel("rank(r)")
        else:
            ax.set_xlabel("")

        # left column → ylabel
        if i % 2 == 0:
            ax.set_ylabel("EVr")
        else:
            ax.set_ylabel("")
        ax.grid(True, alpha=0.3)

    # single shared legend on top
    fig.legend(lines_for_legend, labels_for_legend, loc="lower center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{args.output_dir}svd_scree_2x2_datasets_3models.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")  # path only
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # parser.add_argument("--dataset", type=str, default="strategyqa")  # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    # parser.add_argument("--prompt_mode", type=str, default="both_w_instruction",
    #                     help="prior_wo_context/context_only/prior_only/both_w_instruction/both_wo_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    args = parser.parse_args()
    print(args)

    set_seed(args)
    main(args)