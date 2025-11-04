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
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Collect per-dataset values for both variants
    data_by_dataset = []   # list of dicts in the found order
    labels = []            # pretty dataset labels for x-axis

    for dataset in dataset_list:
        base = os.path.join(args.output_dir, dataset, model_name)

        # File paths
        path_cot = os.path.join(base, f"all_proj_rank2_{args.prompt_mode}_cot.pkl")
        path_noexp = os.path.join(base, f"all_proj_rank2_both_w_instruction_no_explanation.pkl")

        # Load if present
        if not os.path.exists(path_cot) and not os.path.exists(path_noexp):
            print(f"[warn] Skipping dataset (no files): {dataset}")
            continue

        pk_cot, ck_cot = [], []
        pk_noexp, ck_noexp = [], []

        if os.path.exists(path_cot):
            seqs_cot = load_pickle(path_cot)
            for item in seqs_cot:
                pk_cot.append(abs(float(item[0])))
                ck_cot.append(abs(float(item[1])))
        else:
            print(f"[warn] Missing CoT file: {path_cot}")

        if os.path.exists(path_noexp):
            seqs_noexp = load_pickle(path_noexp)
            for item in seqs_noexp:
                pk_noexp.append(abs(float(item[0])))
                ck_noexp.append(abs(float(item[1])))
        else:
            print(f"[warn] Missing No-Explanation file: {path_noexp}")

        # Only include dataset if we have at least one variant with data
        if any(len(x) > 0 for x in [pk_cot, ck_cot, pk_noexp, ck_noexp]):
            data_by_dataset.append(
                {
                    "name": data_to_paper_name.get(dataset, dataset),
                    "PK_CoT": pk_cot,
                    "CK_CoT": ck_cot,
                    "PK_NoExp": pk_noexp,
                    "CK_NoExp": ck_noexp,
                }
            )
            labels.append(data_to_paper_name.get(dataset, dataset))

    if not data_by_dataset:
        raise RuntimeError("No PK/CK values found for any dataset/variant.")

    # Figure sizing: a bit wider per dataset group so labels don't cram
    fig_width = max(4.5, 1.6 * len(data_by_dataset) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 2.6))

    # Prepare boxplot data and positions
    # Order within each dataset group: [PK CoT, CK CoT, PK NoExp, CK NoExp]
    box_data = []
    positions = []
    xticks = []
    xticklabels = []

    group_spacing = 5.0           # space per dataset
    offsets = np.array([0.8, 1.6, 2.6, 3.4])  # within-group positions for the 4 boxes

    for i, entry in enumerate(data_by_dataset):
        gbase = i * group_spacing
        # Gather in fixed order, even if some lists are empty
        box_data.extend([
            entry["PK_CoT"], entry["CK_CoT"], entry["PK_NoExp"], entry["CK_NoExp"]
        ])
        positions.extend(list(gbase + offsets))
        # Center tick under the group
        xticks.append(gbase + offsets.mean())
        xticklabels.append(entry["name"])

    # Create the boxplot
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        whis=1.5,
    )

    # Styling: color → PK/CK, hatch → Variant
    # Index pattern repeats every 4 boxes per dataset: 0:PK CoT, 1:CK CoT, 2:PK NoExp, 3:CK NoExp
    color_map = {0: "#1f77b4", 1: "#ff7f0e", 2: "#1f77b4", 3: "#ff7f0e"}  # PK blue, CK orange
    hatch_map = {0: "///", 1: "///", 2: "", 3: ""}                       # CoT hatched, NoExp solid

    for i, box in enumerate(bp["boxes"]):
        k = i % 4
        box.set_facecolor(color_map[k])
        box.set_alpha(0.30)
        box.set_hatch(hatch_map[k])

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_linewidth(1.2)
    for mean in bp["means"]:
        mean.set_linewidth(1.0)

    # Axes / ticks
    ax.set_ylabel("Knowledge Contribution")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=20)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Legends: (1) PK/CK by color, (2) Variant by hatch
    import matplotlib.patches as mpatches
    color_handles = [
        mpatches.Patch(facecolor="#1f77b4", alpha=0.5, label="PK"),
        mpatches.Patch(facecolor="#ff7f0e", alpha=0.5, label="CK"),
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="CoT"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="Standard Prompting"),
    ]
    # Put both legends in one, side-by-side
    from itertools import chain
    both_handles = list(chain(color_handles, hatch_handles))
    ax.legend(both_handles, [h.get_label() for h in both_handles],
              loc="upper left", frameon=False, ncol=2)

    # Layout & save
    plt.tight_layout(pad=0.6)
    out_pdf = os.path.join(
        args.output_dir, f"rank2_proj_contrib_answer_token_grouped_{model_name}_cot_noexp.pdf"
    )
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved plot to: {out_pdf}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Meta-Llama-3.1-8B-Instruct / google/gemma-2-9b-it / mistralai/Mistral-7B-Instruct-v0.3
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    #                     help="Device (cuda or cpu)")
    # parser.add_argument("--dataset", type=str, default="strategyqa") # strategyqa / basefakepedia / multihopfakepedia / openbookqa / musique
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--prompt_mode", type=str, help = "prior_wo_context/prior_only/context_only/both_w_instruction/both_wo_instruction", default="both_wo_instruction")
    parser.add_argument("--output_dir", type=str, default="../results/")
    parser.add_argument("--smooth_window", type=int, default=9,
                        help="Odd window size for centered moving average (1 = no smoothing)")
    args = parser.parse_args()
    print(args)

    # set_seed(args)

    main(args)
