"""Stage 1 on all 4 scenarios x 10 replicates, then mean/SE of transfer matrices."""
import io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
from fit_stage1 import main

SCENARIOS = ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]
OUT = Path("outputs/stage1")

for s in SCENARIOS:
    for r in range(10):
        out = OUT / s / f"rep{r}"
        if (out / "target_scores.csv").exists():      # skip finished runs (resumable)
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            main(f"data/{s}", r, out)
        print(s, r, "done", flush=True)

for s in SCENARIOS:
    files = [OUT / s / f"rep{r}" / "transfer_matrix.csv" for r in range(10)]
    mats = np.stack([pd.read_csv(f, index_col=0).to_numpy() for f in files])
    ref = pd.read_csv(files[0], index_col=0)
    mean = pd.DataFrame(mats.mean(0), ref.index, ref.columns)
    se = pd.DataFrame(mats.std(0, ddof=1) / np.sqrt(10), ref.index, ref.columns)
    mean.to_csv(OUT / s / "transfer_mean.csv")
    se.to_csv(OUT / s / "transfer_se.csv")
    print(s, "\n", mean.round(3).to_string(), "\n")