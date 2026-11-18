"""
Verify Stage 3 incumbent baseline against saved Stage 2 MoEPRS predictions.

Computes:
    X = max_i |y_hat_i^(Stage 2) - y_hat_i^(Stage 3)|

Usage (run via CLI with required arguments; do not run directly without flags):
    python check_incumbent_stage2.py data/shared_effects --replicate 0 \
        --scores "outputs/stage1/shared_effects/rep0/target_scores.csv" \
        --stage2-preds "outputs/stage2/shared_effects/rep0/n2000/stage2_moe_predictions.csv"

PowerShell (single line):
    python check_incumbent_stage2.py data/shared_effects --replicate 0 --scores "outputs\stage1\shared_effects\rep0\target_scores.csv" --stage2-preds "outputs\stage2\shared_effects\rep0\n2000\stage2_moe_predictions.csv"
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "moeprs.py").exists() and (ROOT.parent / "moeprs.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from harness import Dataset
from moeprs import MoEPRS
from stage2_baselines import load_experts
from stages.stage1_moeprs import build_moeprs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "data",
        nargs="?",
        default="data/shared_effects",
        help="Scenario path (e.g., data/shared_effects)",
    )
    ap.add_argument("--replicate", type=int, default=0, help="Replicate index (0-9)")
    ap.add_argument(
        "--scores",
        default=None,
        help="Optional Stage 1 target_scores.csv (must match Stage 2 run)",
    )
    ap.add_argument(
        "--stage2-preds",
        required=True,
        help="Path to saved Stage 2 predictions CSV (columns: iid, prediction)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Tolerance threshold for numerical agreement",
    )
    args = ap.parse_args()

    # 1. Load the exact same replicate and expert scores (5 experts via load_experts)
    d = load_experts(Dataset(args.data, args.replicate), args.scores)
    S_train, Z_train, C_train, y_train, _ = d.part("train")

    # 2. Refit Stage 3 baseline with exact Stage 2 defaults:
    #    MoEPRS(n_pc=10, l2_gate=1e-2, n_em=40, tol=1e-6, seed=0, gate="linear", expert="linear")
    stage3_predict = build_moeprs(
        S_train,
        Z_train,
        C_train,
        y_train,
        n_pc=10,
        l2_gate=0.01,
        n_em=40,
        tol=1e-6,
        seed=0,
        gate="linear",
        expert="linear",
    )

    # 3. Load saved Stage 2 predictions and match the split by `iid`
    df_s2 = pd.read_csv(args.stage2_preds)
    if not {"iid", "prediction"}.issubset(df_s2.columns):
        raise ValueError(
            f"Expected columns {{'iid', 'prediction'}} in {args.stage2_preds}, "
            f"got {list(df_s2.columns)}"
        )

    saved_iids = df_s2["iid"].to_numpy()
    if np.array_equal(np.sort(saved_iids), np.sort(d.idx_test)):
        split = "test"
    elif np.array_equal(np.sort(saved_iids), np.sort(d.idx_val)):
        split = "val"
    else:
        raise ValueError(
            "Saved `iid` values do not match either `test` or `val` indices "
            f"for {args.data} replicate={args.replicate}."
        )

    S_eval, Z_eval, C_eval, _, eval_iids = d.part(split)
    yhat_s3 = np.asarray(stage3_predict(S_eval, Z_eval, C_eval), dtype=float)

    # Align predictions by `iid` in case row order differs
    s3_by_iid = pd.Series(yhat_s3, index=eval_iids)
    yhat_s3_aligned = s3_by_iid.loc[saved_iids].to_numpy()
    yhat_s2 = df_s2["prediction"].to_numpy(dtype=float)

    # 4. Compare prediction arrays and report X = max_i |y_hat_s2 - y_hat_s3|
    abs_diff = np.abs(yhat_s2 - yhat_s3_aligned)
    X = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    print(f"Scenario        : {d.scenario} (replicate={d.replicate})")
    print(f"Matched split   : {split} (n={len(saved_iids)} individuals)")
    print(f"Experts shape   : {S_train.shape[1]} experts")
    print(f"Max abs diff (X): {X:.6e}")
    print(f"Mean abs diff   : {mean_diff:.6e}")

    if X == 0.0:
        print("Status          : EXACT MATCH (X == 0)")
    elif X <= args.tol:
        print(f"Status          : NUMERICAL AGREEMENT (X <= {args.tol:.1e})")
    else:
        print(f"Status          : MISMATCH (X > {args.tol:.1e})")
        sys.exit(1)


if __name__ == "__main__":
    main()