"""
Stage 1 figures/tables
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/stage1")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]

mean = {s: pd.read_csv(OUT / s / "transfer_mean.csv", index_col=0) for s in SCENARIOS}
se = {s: pd.read_csv(OUT / s / "transfer_se.csv", index_col=0) for s in SCENARIOS}
vmax = max(m.to_numpy().max() for m in mean.values())   # same colour scale everywhere


def draw(ax, s):
    m, e = mean[s], se[s]
    im = ax.imshow(m.to_numpy(), cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
    ax.set_xlabel("Evaluation cohort"); ax.set_ylabel("Source PRS")
    ax.set_title(s)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{v:.3f}\n±{e.iat[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if v < 0.6 * vmax else "black")
    return im


def figure(scens, name, nrows, ncols, size):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True)

    im = None
    for ax, s in zip(axes.ravel() if hasattr(axes, "ravel") else [axes], scens):
        im = draw(ax, s)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Incremental R² (mean over 10 replicates)")

    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


figure(["shared_effects", "ancestry_dependent"], "transfer_shared_vs_ancestry.png", 1, 2, (12, 4.5))
figure(SCENARIOS, "transfer_all_scenarios.png", 2, 2, (12, 9))

# "mean ± SE" tables, one CSV per scenario
for s in SCENARIOS:
    tab = mean[s].map(lambda v: f"{v:.3f}") + " ± " + se[s].map(lambda v: f"{v:.3f}")
    tab.to_csv(FIG / f"transfer_table_{s}.csv")
    print(s, "\n", tab.to_string(), "\n")
print("Saved to", FIG.resolve())