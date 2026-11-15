"""
Stage 2 figures: gate weights vs PC1/PC2 and paired learning curves.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/stage2"); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]
EXPERTS = ["EUR_viprs", "AFR_viprs", "EAS_viprs", "SAS_viprs", "AMR_viprs"]

# 1) Gate weights vs PC1 and PC2 (replicate 0, full training size, validation individuals)
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True, constrained_layout=True)
for c, s in enumerate(SCENARIOS):
    g = pd.read_csv(OUT / s / "rep0" / "n2000" / "validation_gate_weights.csv")
    for r, pc in enumerate(["PC1", "PC2"]):
        for e in EXPERTS:
            axes[r, c].scatter(g[pc], g[e], s=4, alpha=0.5, label=e.replace("_viprs", ""))
        axes[r, c].set_xlabel(pc); axes[0, c].set_title(s)
    axes[0, c].set_ylabel("gate weight")
    axes[1, c].set_ylabel("gate weight")
axes[0, 0].legend(markerscale=3, fontsize=8)
fig.savefig(FIG / "gate_weights_vs_pcs.png", dpi=200); plt.close(fig)


# 2) Gate weights: mean weight per expert in deciles of PC1 / PC2
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True, constrained_layout=True)
for c, s in enumerate(SCENARIOS):
    g = pd.read_csv(OUT / s / "rep0" / "n2000" / "validation_gate_weights.csv")
    for r, pc in enumerate(["PC1", "PC2"]):
        bins = pd.qcut(g[pc], 10, duplicates="drop")
        grp = g.groupby(bins, observed=True)
        x = grp[pc].mean().values
        for e in EXPERTS:
            axes[r, c].plot(x, grp[e].mean().values, marker="o", label=e.replace("_viprs", ""))
        axes[r, c].axhline(0.2, color="k", ls=":", lw=0.8)      # equal mixture = 0.2 each
        axes[r, c].set_xlabel(f"{pc} (decile means)")
    axes[0, c].set_title(s); axes[0, c].set_ylabel("mean gate weight"); axes[1, c].set_ylabel("mean gate weight")
axes[0, 0].legend(fontsize=8)
fig.savefig(FIG / "binned_gate_weights_vs_pcs.png", dpi=200); plt.close(fig)


# 3) Learning curves: mean incremental R^2 by method, and paired comparison:
#    ΔR² = R²_MoE - R²_baseline
df = pd.read_csv(OUT / "all_validation_metrics.csv")
sizes = sorted(df.n_train.unique())
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True, constrained_layout=True)
for ax, s in zip(axes, SCENARIOS):
    d = df[df.scenario == s]
    for m, g in d.groupby("method"):
        p = g.pivot(index="replicate", columns="n_train", values="r2")
        ax.errorbar(sizes, p.mean(), p.std(ddof=1) / np.sqrt(len(p)), marker="o", capsize=3, label=m)
    ax.set_title(s); ax.set_xscale("log"); ax.set_xlabel("training size"); ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
axes[0].set_ylabel("validation incremental R² (mean ± SE, 10 replicates)"); axes[0].legend(fontsize=8)
fig.savefig(FIG / "learning_curves.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True, constrained_layout=True)
for ax, s in zip(axes, SCENARIOS):
    d = df[df.scenario == s]
    wide = {m: g.pivot(index="replicate", columns="n_train", values="r2") for m, g in d.groupby("method")}
    for base in ["multiprs", "best_single", "genotype_ridge"]:
        diff = wide["stage2_moe"] - wide[base]                      # paired within replicate
        ax.errorbar(sizes, diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)),
                    marker="o", capsize=3, label=f"MoE − {base}")
    ax.axhline(0, color="k", lw=0.8); ax.set_title(s); ax.set_xscale("log")
    ax.set_xlabel("training size"); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
axes[0].set_ylabel("paired ΔR² (mean ± SE)"); axes[0].legend(fontsize=8)
fig.savefig(FIG / "paired_learning_curves.png", dpi=200); plt.close(fig)
print("Saved to", FIG.resolve())