#!/usr/bin/env python3
import os, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------- Pretty names ----------
DATA_TO_PAPER = {
    'strategyqa': 'StrategyQA',
    'basefakepedia': 'BaseFakepedia',
    'multihopfakepedia': 'MultihopFakepedia',
    'openbookqa': 'OpenBookQA'
}

# ---------- small util ----------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

# ---------- IO ----------
def load_matrix(path):
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array at {path}, got shape {arr.shape}")
    return arr

# ---------- Linear algebra ----------
def svd_centered(X, center=True):
    """SVD on (optionally) row-centered X (N,d). Return singular values S and Vt."""
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S, Vt

def cum_explained_variance(S):
    lam = S**2
    tot = lam.sum()
    if tot <= 0:
        return np.zeros_like(S)
    return np.cumsum(lam) / tot

def effective_rank_entropy(S):
    lam = S**2
    tot = lam.sum()
    if tot <= 0:
        return 0.0
    p = lam / tot
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))

def participation_ratio(S):
    lam = S**2
    num = lam.sum()**2
    den = (lam**2).sum()
    return float(num / den) if den > 0 else 0.0

# ---------- Permutation test (sign-flip) ----------
def perm_test_sigmas_diff(D, n_perm=1000, seed=0, center=True):
    """
    D: (N, d) difference matrix (CoT - NoCoT). Null: random sign flips.
    Returns dict with observed S, null sigmas, and p-values per component.
    """
    rng = np.random.default_rng(seed)
    S_obs, _ = svd_centered(D, center=center)
    r = len(S_obs)
    null_s = np.zeros((n_perm, r), dtype=S_obs.dtype)

    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(D.shape[0], 1))
        S_null, _ = svd_centered(D * signs, center=center)
        s = np.zeros_like(S_obs)
        s[:len(S_null)] = S_null
        null_s[i] = s

    pvals = (null_s >= S_obs[None, :]).mean(axis=0)
    return {"S_obs": S_obs, "null_sigmas": null_s, "pvals": pvals}

# ---------- Core ----------
def analyze_cot_space(cot_mat, nocot_mat, center=True, n_perm=1000, seed=0, alpha=0.05):
    if cot_mat.shape != nocot_mat.shape:
        raise ValueError(f"Shape mismatch: CoT {cot_mat.shape} vs NoCoT {nocot_mat.shape}")
    D = cot_mat - nocot_mat
    S, _ = svd_centered(D, center=center)
    ev = cum_explained_variance(S)
    perms = perm_test_sigmas_diff(D, n_perm=n_perm, seed=seed, center=center)
    pvals = perms["pvals"]
    k_sig = int(np.sum(pvals < alpha))
    return {
        "S": S, "ev": ev, "pvals": pvals, "k_sig": k_sig,
        "r_eff": effective_rank_entropy(S),
        "pr": participation_ratio(S),
    }

def main(args):
    model_name = args.model_name.strip().split('/')[-1]
    # Plot style
    plt.rcParams.update({
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
    })

    datasets = ["strategyqa", "basefakepedia", "multihopfakepedia", "openbookqa"]

    # 2×2 grid, single model curve per subplot
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), constrained_layout=False, sharex=False, sharey=True)
    axes = axes.flatten()

    lines, labels = [], []
    summary_rows = []

    for i, ds in enumerate(datasets):
        ax = axes[i]
        base = os.path.join(args.output_dir, ds, model_name)

        cot_path   = os.path.join(base, f"all_pred_hs_{args.prompt_mode}_cot.pkl")
        nocot_path = os.path.join(base, f"all_pred_hs_{args.prompt_mode}_no_explanation.pkl")

        if not os.path.exists(cot_path) or not os.path.exists(nocot_path):
            print(f"[WARN] Missing: {cot_path} or {nocot_path}")
            ax.set_title(DATA_TO_PAPER.get(ds, ds))
            ax.set_xlabel("rank r")
            if i % 2 == 0: ax.set_ylabel("Cumulative explained variance")
            ax.grid(alpha=0.3, linewidth=0.5)
            continue

        C = load_matrix(cot_path)
        N = load_matrix(nocot_path)

        res = analyze_cot_space(
            C, N,
            center=not args.no_center,
            n_perm=args.n_perm,
            seed=args.seed,
            alpha=args.alpha
        )

        x = np.arange(1, len(res["ev"]) + 1)
        line = ax.plot(x, res["ev"], marker="o", linestyle="-", label=args.model_name)[0]

        if i == 0:
            lines.append(line)
            labels.append(args.model_name)

        if res["k_sig"] > 0:
            ax.axvline(res["k_sig"], linestyle="--", linewidth=0.8, alpha=0.6)

        summary_rows.append({
            "dataset": ds, "model": args.model_name,
            "k_sig": res["k_sig"], "r_eff": round(res["r_eff"], 2),
            "PR": round(res["pr"], 2)
        })

        ax.set_title(DATA_TO_PAPER.get(ds, ds))
        ax.set_xlabel("rank r")
        if i % 2 == 0:
            ax.set_ylabel("Cumulative explained variance")
        ax.grid(alpha=0.3, linewidth=0.5)

    # Legend (single entry, but keeps figure consistent)
    if lines:
        fig.legend(lines, labels, loc="lower center", ncol=len(labels),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.8)
    plt.subplots_adjust(bottom=0.16)

    out_pdf = os.path.join(args.output_dir, f"cot_minus_nocot_scree_2x2_{args.model_name}.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print("\nSuggested CoT-space rank by permutation test (p < {:.2f}):".format(args.alpha))
    for row in summary_rows:
        print("{dataset:>16} | {model:<28} | k_sig={k_sig:<2} | r_eff={r_eff:<5} | PR={PR:<5}".format(**row))
    print(f"\nSaved plot to: {out_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_wo_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--no_center", action="store_true",
                    help="if set, do NOT mean-center ΔH before SVD")
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    main(args)
