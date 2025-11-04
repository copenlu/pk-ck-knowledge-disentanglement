#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from statistics import mode


data_to_paper_name = {
    "strategyqa": "StrategyQA",
    "basefakepedia": "BaseFakepedia",
    "multihopfakepedia": "MultiHopFakepedia",
    "openbookqa": "OpenBookQA",
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_positionwise_stats(seqs_rank2, seqs_rank1):
    """
    Given a list of sequences `seqs`, where each sequence is an array-like of shape [T, 2]
    with columns [memory, context], compute the per-position average and std of the absolute
    contributions for both memory and context across sequences that have that position.
    Returns: avg_context, std_context, avg_memory, std_memory (each shape [max_len])
    """
    if not seqs_rank2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_len = max(len(seq) for seq in seqs_rank2)
    avg_context_rank2, std_context_rank2 = [], []
    avg_memory_rank2, std_memory_rank2 = [], []

    avg_context_rank1, std_context_rank1 = [], []
    avg_memory_rank1, std_memory_rank1 = [], []

    rank1_len_list = [len(seq) for seq in seqs_rank1]
    rank2_len_list = [len(seq) for seq in seqs_rank2]


    for pos in range(max_len):
        # take only sequences long enough
        ctx_vals_rank2 = [abs(seq[pos][1]) / (abs(seq[pos][1]) + abs(seq[pos][0])) for seq in seqs_rank2 if len(seq) > pos] 
        mem_vals_rank2 = [abs(seq[pos][0]) / (abs(seq[pos][1]) + abs(seq[pos][0])) for seq in seqs_rank2 if len(seq) > pos]

        ctx_vals_rank1 = [1.0 if seq[pos] <= 0.0 else 0.0  for seq in seqs_rank1 if len(seq) > pos]
        mem_vals_rank1 = [0.0 if seq[pos] <= 0.0 else 1.0 for seq in seqs_rank1 if len(seq) > pos]

        # guard if somehow empty (shouldn't happen if there's at least one seq)
        if len(ctx_vals_rank2) == 0:
            avg_context_rank2.append(0.0)
            std_context_rank2.append(0.0)
            avg_memory_rank2.append(0.0)
            std_memory_rank2.append(0.0)
        else:
            avg_context_rank2.append(float(np.mean(ctx_vals_rank2)))
            std_context_rank2.append(float(np.std(ctx_vals_rank2, ddof=0)))
            avg_memory_rank2.append(float(np.mean(mem_vals_rank2)))
            std_memory_rank2.append(float(np.std(mem_vals_rank2, ddof=0)))

        if len(ctx_vals_rank1) == 0:
            avg_context_rank1.append(0.0)
            std_context_rank1.append(0.0)
            avg_memory_rank1.append(0.0)
            std_memory_rank1.append(0.0)
        else:
            avg_context_rank1.append(float(np.mean(ctx_vals_rank1)))
            std_context_rank1.append(float(np.std(ctx_vals_rank1, ddof=0)))
            avg_memory_rank1.append(float(np.mean(mem_vals_rank1)))
            std_memory_rank1.append(float(np.std(mem_vals_rank1, ddof=0)))

    return (
        np.array(avg_context_rank2),
        np.array(std_context_rank2),
        np.array(avg_memory_rank2),
        np.array(std_memory_rank2),
        np.array(avg_context_rank1),
        np.array(std_context_rank1),
        np.array(avg_memory_rank1),
        np.array(std_memory_rank1),
        rank2_len_list
    )


def smooth(arr, window=7):
    """
    Centered moving average with edge-padding.
    'window' must be odd; set to 1 to disable smoothing.
    """
    arr = np.asarray(arr)
    window = int(window)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def main(args):
    model_name = args.model_name.strip().split("/")[-1]

    dataset_list = ["strategyqa", "basefakepedia", "multihopfakepedia", "openbookqa"]

    # ---- ICLR-friendly styling ----
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,  # good for 10pt papers
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,  # TrueType fonts (embed as text)
            "ps.fonttype": 42,
        }
    )

    # One-column ICLR width is ~5.5 in; pick a compact height
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), constrained_layout=False)
    axes = axes.flatten()

    for idx, dataset in enumerate(dataset_list):
        base = os.path.join(args.output_dir, dataset, model_name)

        # Only files actually used by this plot:
        # We compute the dynamics from the rank-2 projection over *all* examples.
        all_rank2_path = os.path.join(
            base, f"comp_proj_rank2_{args.prompt_mode}_nle.pkl"
        )
        all_rank1_path = os.path.join(
            base, f"comp_proj_rank1_{args.prompt_mode}_nle.pkl"
        )

        if not os.path.exists(all_rank2_path):
            raise FileNotFoundError(
                f"Missing file: {all_rank2_path}\n"
                "If you intended to plot subsets (supp/comp/etc.), add that logic or create those files."
            )

        all_proj_rank2 = load_pickle(all_rank2_path)
        all_proj_rank1 = load_pickle(all_rank1_path)

        # Compute per-position stats
        avg_ctx, std_ctx, avg_mem, std_mem, avg_ctx_r1, std_ctx_r1, avg_mem_r1, std_mem_r1, rank2_len_list = compute_positionwise_stats(all_proj_rank2, all_proj_rank1)

        # Smoothing
        win = args.smooth_window
        avg_ctx_s = smooth(avg_ctx, win)
        std_ctx_s = smooth(std_ctx, win)
        avg_mem_s = smooth(avg_mem, win)
        std_mem_s = smooth(std_mem, win)

        avg_ctx_s_r1 = smooth(avg_ctx_r1, win)
        std_ctx_s_r1 = smooth(std_ctx_r1, win)
        avg_mem_s_r1 = smooth(avg_mem_r1, win)
        std_mem_s_r1 = smooth(std_mem_r1, win)

        # X axis (1-indexed steps)
        max_len = len(avg_ctx_s)
        x = np.arange(1, max_len + 1)

        ax = axes[idx]

        # Context line + SD band
        ax.plot(x, avg_ctx_s, label="CK")
        ax.fill_between(
            x, avg_ctx_s - std_ctx_s, avg_ctx_s + std_ctx_s, alpha=0.2, linewidth=0
        )

        # Memory line + SD band
        ax.plot(x, avg_mem_s, label="PK")
        ax.fill_between(
            x, avg_mem_s - std_mem_s, avg_mem_s + std_mem_s, alpha=0.2, linewidth=0
        )

        # ax.plot(x, avg_ctx_s_r1, label="CR1")
        # ax.fill_between(
        #     x, avg_ctx_s_r1 - std_ctx_s_r1, avg_ctx_s_r1 + std_ctx_s_r1, alpha=0.2, linewidth=0
        # )

        # ax.plot(x, avg_mem_s_r1, label="MR1")
        # ax.fill_between(
        #     x, avg_mem_s_r1 - std_mem_s_r1, avg_mem_s_r1 + std_mem_s_r1, alpha=0.2, linewidth=0
        # )

        # --- NEW: vertical lines at mean & mode ---
        mean_val = np.mean(rank2_len_list)
        mode_val = mode(rank2_len_list)

        # Mean line (red, dashed)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.0,
                   label="Mean length")

        # Mode line (blue, dotted)
        if mode_val is not None:
            ax.axvline(mode_val, color="blue", linestyle=":", linewidth=1.0,
                       label="Mode length")

        ax.set_title(data_to_paper_name.get(dataset, dataset))
        # bottom row → xlabel
        if idx >= 2:
            ax.set_xlabel("Sequence step")
        else:
            ax.set_xlabel("")

        # left column → ylabel
        if idx % 2 == 0:
            ax.set_ylabel("PK-CK Contribution")
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.2, linewidth=0.5)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),  # small negative or positive to fine-tune
        ncol=len(labels),             # put all entries in one row
        frameon=False
    )

    # Adjust subplot layout so bottom has room for legend
    plt.tight_layout(pad=0.6, w_pad=0.6, h_pad=0.6)
    plt.subplots_adjust(bottom=0.12)  # positive fraction of figure height

    out_pdf = os.path.join(
        args.output_dir, f"comp_rank2_proj_contrib_pk_ck_dynamics_{model_name}_abs.pdf"
    )
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    #                     help="Device (cuda or cpu)")
    # parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_w_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    parser.add_argument("--smooth_window", type=int, default=9,
                        help="Odd window size for centered moving average (1 = no smoothing)")
    args = parser.parse_args()
    print(args)

    # set_seed(args)

    main(args)
