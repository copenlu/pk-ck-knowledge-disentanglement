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
        ctx_vals_rank2 = [seq[pos][1] for seq in seqs_rank2 if len(seq) > pos] 
        mem_vals_rank2 = [seq[pos][0] for seq in seqs_rank2 if len(seq) > pos]

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
            "font.size": 8,  # good for 10pt papers
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,  # TrueType fonts (embed as text)
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.5, 2.5))  # wider to fit multiple datasets

    # Collect data: one PK and CK list per dataset
    pk_all, ck_all = [], []
    labels = []

    for dataset in dataset_list:
        base = os.path.join(args.output_dir, dataset, model_name)

        all_rank2_path = os.path.join(
            base, f"all_proj_rank2_{args.prompt_mode}_no_explanation.pkl"
        )

        if not os.path.exists(all_rank2_path):
            print(f"[warn] Skipping missing file: {all_rank2_path}")
            continue

        all_proj_rank2 = load_pickle(all_rank2_path)

        pk_list, ck_list = [], []
        for item in all_proj_rank2:
            pk_list.append(abs(float(item[0])))
            ck_list.append(abs(float(item[1])))

        if len(pk_list) == 0 or len(ck_list) == 0:
            continue

        pk_all.append(pk_list)
        ck_all.append(ck_list)
        labels.append(data_to_paper_name[dataset])

    if len(pk_all) == 0:
        raise RuntimeError("No PK/CK values found to plot.")

    # --- Make grouped boxplot ---
    positions = []
    data = []
    xticks = []
    for i, (pk_list, ck_list) in enumerate(zip(pk_all, ck_all)):
        pos_pk = 2 * i + 1
        pos_ck = 2 * i + 2
        positions.extend([pos_pk, pos_ck])
        data.extend([pk_list, ck_list])
        xticks.append((pos_pk + pos_ck) / 2.0)  # center for dataset label

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        whis=1.5,
    )

    # Styling
    colors = ["#1f77b4", "#ff7f0e"]  # PK blue, CK orange
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % 2])
        box.set_alpha(0.3)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_linewidth(1.2)
    for mean in bp["means"]:
        mean.set_linewidth(1.0)

    # Axis labels
    ax.set_ylabel("Knowledge Contribution")
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=20)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Add legend manually
    handles = [
        plt.Line2D([0], [0], color=colors[0], lw=4, alpha=0.5),
        plt.Line2D([0], [0], color=colors[1], lw=4, alpha=0.5),
    ]
    ax.legend(handles, ["PK", "CK"], loc="upper right", frameon=False)

    # Layout & save
    plt.tight_layout(pad=0.6)
    out_pdf = os.path.join(
        args.output_dir, f"rank2_proj_contrib_answer_token_grouped_{model_name}_no_explanation.pdf"
    )
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved plot to: {out_pdf}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
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
