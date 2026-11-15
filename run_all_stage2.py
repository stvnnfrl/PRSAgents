"""
Stage 2 on all scenarios x 10 replicates x training sizes, then paired contrasts.
"""
import io, contextlib
from pathlib import Path
import pandas as pd
from stage2_baselines import run
from paired_contrasts import summarize

SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]
SIZES = [250, 500, 1000, 2000]
OUT = Path("outputs/stage2")

for s in SCENARIOS:
    for r in range(10):
        scores = f"outputs/stage1/{s}/rep{r}/target_scores.csv"
        for n in SIZES:
            out = OUT / s / f"rep{r}" / f"n{n}"
            if (out / "validation_metrics.csv").exists():
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                run(f"data/{s}", out, scores, None if n == 2000 else n, r)
            print(s, r, n, "done", flush=True)

# Collect every validation_metrics.csv into one table
rows = []
for s in SCENARIOS:
    for r in range(10):
        for n in SIZES:
            m = pd.read_csv(OUT / s / f"rep{r}" / f"n{n}" / "validation_metrics.csv")
            for _, x in m.iterrows():
                rows.append(dict(scenario=s, replicate=r, n_train=n, method=x.method,
                                 r2=x.validation_incremental_r2, mse=x.validation_mse))
df = pd.DataFrame(rows)
df.to_csv(OUT / "all_validation_metrics.csv", index=False)

# Paired contrasts: stage2_moe minus each baseline, same replicates
paired_comparison_results = []
for n in SIZES:
    d = df[df.n_train == n]
    for base in ["multiprs", "best_single", "genotype_ridge"]:
        for metric in ["r2", "mse"]:
            t = summarize(d.rename(columns={metric: "score"})
                          [["scenario", "replicate", "method", "score"]], base, "stage2_moe")
            t.insert(0, "n_train", n); t.insert(1, "baseline", base); t.insert(2, "metric", metric)
            paired_comparison_results.append(t)
paired_comparison_results = pd.concat(paired_comparison_results)
paired_comparison_results.to_csv(OUT / "paired_contrasts.csv", index=False)
print(paired_comparison_results[paired_comparison_results.n_train == 2000].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))