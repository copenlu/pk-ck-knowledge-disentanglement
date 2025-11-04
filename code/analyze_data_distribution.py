#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ==== Fill in your preferred model names here ====
model_names = ["Llama-3.1-8B", "Gemma-2 9B", "Mistral-v0.3 7B"]

# ==== Data (counts) ====
# Order: [supportive, conflicting, complementary, irrelevant, noise]
data = {
    "StrategyQA": {
        model_names[0]: [1756, 487, 5,   0,   41],
        model_names[1]: [1809, 294, 12,  60,  114],
        model_names[2]: [1609, 627, 7,   5,   41],
    },
    "OpenBookQA": {
        model_names[0]: [4258, 2764, 79,  530, 159],
        model_names[1]: [3927, 2303, 69,  1417, 74],
        model_names[2]: [6105, 1275, 34,  233,  143],
    },
    "BaseFakepedia": {
        model_names[0]: [1838, 4147, 82,  1,   20],
        model_names[1]: [2323, 3700, 57,  0,   8],
        model_names[2]: [4587, 1430, 28,  0,   43],
    },
    "MultiHopFakepedia": {
        model_names[0]: [972,  3604, 129, 5,   44],
        model_names[1]: [974,  3381, 396, 0,   3],
        model_names[2]: [3508, 1144, 54,  0,   48],
    },
}

categories = ["supportive", "conflicting", "complementary", "irrelevant", "noise"]
datasets_order = ["StrategyQA", "OpenBookQA", "BaseFakepedia", "MultiHopFakepedia"]

# ---- Styling (compact, paper-friendly) ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), constrained_layout=False)
axes = axes.flatten()

n_cat = len(categories)
x = np.arange(n_cat)

n_models = len(model_names)
bar_width = 0.22
# center the triplet around each category
offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * (bar_width + 0.02)

for ax, ds in zip(axes, datasets_order):
    # collect per-model arrays in a fixed model order
    model_series = [np.array(data[ds][m]) for m in model_names]

    # plot
    for i, (mname, series) in enumerate(zip(model_names, model_series)):
        ax.bar(x + offsets[i], series, width=bar_width, label=mname)

    ax.set_title(ds)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

# one common legend at the bottom in a single row
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
           ncol=len(model_names), frameon=False)

plt.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.8)
plt.subplots_adjust(bottom=0.16)  # room for bottom legend

# Save (optional)
plt.savefig("../results/category_counts_by_model.pdf", bbox_inches="tight", pad_inches=0.1)
# plt.show()
