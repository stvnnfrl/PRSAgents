"""Stage 2 summary in one run: (1) file/fit checks, (2) EM convergence, (3) result tables.

Run from the repo root:  python custom_stage2_summary.py
Writes to outputs/stage2/: checks.csv, baseline_table_r2.csv, baseline_table_mse.csv,
contrast_tables.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("outputs/stage2")
SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]
SIZES = [250, 500, 1000, 2000]
EXPERTS = [f"{p}_viprs" for p in ["EUR", "AFR", "EAS", "SAS", "AMR"]]
METHODS = ["best_single", "multiprs", "genotype_ridge", "stage2_moe"]


def check_runs():
    """Every run: config, gate weights (finite, >=0, sum to 1), log-likelihood, 1,200 test predictions."""
    rows = []
    for s in SCENARIOS:
        for r in range(10):
            ref_ids = None
            for n in SIZES:
                d = OUT / s / f"rep{r}" / f"n{n}"
                row = dict(scenario=s, replicate=r, n_train=n, error="")
                try:
                    cfg = json.loads((d / "stage2_config.json").read_text())
                    ll = pd.read_csv(d / "moe_training_loglik.csv").training_log_likelihood.to_numpy()
                    g = pd.read_csv(d / "validation_gate_weights.csv")[EXPERTS].to_numpy()
                    row["config_ok"] = (cfg["n_train"] == n and cfg["experts"] == EXPERTS
                                        and cfg["n_pc"] == 10 and cfg["l2_gate"] == 0.01)
                    row["loglik_finite"] = bool(np.isfinite(ll).all())
                    row["loglik_decreases"] = int((np.diff(ll) < -1e-6 * np.abs(ll[:-1])).sum())
                    row["em_hit_cap"] = bool(cfg["n_iter"] >= cfg["n_em"])
                    row["gate_ok"] = bool(np.isfinite(g).all() and g.min() >= 0
                                          and np.abs(g.sum(1) - 1).max() < 1e-6 and len(g) == 800)
                    if n == 2000:   # relative size of the last EM-iteration increase (convergence)
                        row["last_rel_increase"] = float((ll[-2] - ll[-3]) / abs(ll[-3]))
                    ok = True
                    for m in METHODS:
                        p = pd.read_csv(d / f"{m}_predictions.csv")
                        ok &= (list(p.columns) == ["iid", "prediction"] and len(p) == 1200
                               and p.iid.is_unique and bool(np.isfinite(p.prediction).all()))
                        ids = set(p.iid)
                        if ref_ids is None:
                            ref_ids = ids
                        ok &= (ids == ref_ids)
                    row["predictions_ok"] = bool(ok)
                except Exception as e:
                    row["error"] = repr(e)[:120]
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "checks.csv", index=False)
    flag = lambda c: ~df[c].fillna(False).astype(bool)
    print("== 1. Checks ==")
    print("runs checked:", len(df), "| errors:", int((df.error != "").sum()))
    for c in ["config_ok", "loglik_finite", "gate_ok", "predictions_ok"]:
        print(f"{c}: failures =", int(flag(c).sum()))
    print("runs with a log-likelihood decrease:", int((df.loglik_decreases > 0).sum()))
    print("runs that hit the EM iteration cap:", int(df.em_hit_cap.sum()))
    bad = df[(df.error != "") | flag("config_ok") | flag("gate_ok") | flag("predictions_ok")]
    if len(bad):
        print("Problem runs:\n", bad[["scenario", "replicate", "n_train", "error"]].to_string(index=False))
    print("\n== 2. EM convergence (n=2000, last-iteration relative increase; tolerance is 1e-6) ==")
    print(df[df.n_train == 2000].groupby("scenario").last_rel_increase.agg(["median", "min", "max"])
          .map("{:.1e}".format).to_string())


def result_tables():
    """Table 1 (each method's score, mean ± SE) and Table 2 (paired MoE - baseline), all sizes."""
    df = pd.read_csv(OUT / "all_validation_metrics.csv")
    d = df[df.n_train == 2000]
    mean_se = lambda x: f"{x.mean():.4f} ± {x.std(ddof=1) / np.sqrt(len(x)):.4f}"
    for metric in ["r2", "mse"]:
        t = d.groupby(["scenario", "method"])[metric].apply(mean_se).unstack()
        t.to_csv(OUT / f"baseline_table_{metric}.csv")
        print(f"\n== 3. Table 1, {metric}, n=2000 (mean ± SE) ==\n", t.to_string())
    res = pd.read_csv(OUT / "paired_contrasts.csv")
    res["cell"] = res.mean_gain.map("{:+.4f}".format) + " ± " + res.se.map("{:.4f}".format)
    res.to_csv(OUT / "contrast_tables.csv", index=False)
    for metric in ["r2", "mse"]:
        for n in SIZES:
            t = res[(res.metric == metric) & (res.n_train == n) & (res.scenario != "pooled")] \
                .pivot(index="scenario", columns="baseline", values="cell")
            print(f"\n== Table 2, MoE - baseline, {metric}, n={n} ==\n", t.to_string())


if __name__ == "__main__":
    check_runs()
    result_tables()